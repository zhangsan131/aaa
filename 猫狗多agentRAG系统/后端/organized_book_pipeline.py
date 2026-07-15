from __future__ import annotations

"""书籍处理整理版。

目标：
- PDF / EPUB / TXT 统一转为结构化 JSON 知识树
- 尽量保留章节层级、页码和段落信息
- 降低误删正文的概率，为后续多粒度检索做准备
"""

import argparse
import copy
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def _check_dependency(module_name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _check_paddle_gpu_available() -> bool:
    try:
        import paddle

        return bool(paddle.is_compiled_with_cuda())
    except Exception:
        return False


def _check_poppler_available() -> bool:
    return bool(shutil.which("pdftoppm") or shutil.which("pdfinfo"))


def _check_pdf2image_available() -> bool:
    return _check_dependency("pdf2image")


def _check_paddleocr_available() -> bool:
    return _check_dependency("paddleocr")


def _check_ebooklib_available() -> bool:
    return _check_dependency("ebooklib")


_PADDLE_OCR_CACHE = {"gpu": None, "cpu": None}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def _page_contains_any_marker(text: str, markers: List[str]) -> bool:
    normalized = _normalize_text(text)
    return any(_normalize_text(marker) in normalized for marker in markers)


# ============================================================
# Part 1: Source -> MK
# ============================================================


def _find_source_files(root_folder: str) -> List[Path]:
    root = Path(root_folder)
    if not root.exists():
        return []
    files = []
    files.extend(root.rglob("*.pdf"))
    files.extend(root.rglob("*.epub"))
    files.extend(root.rglob("*.txt"))
    return sorted(files)


def _render_pdf_to_images(pdf_path: str):
    from pdf2image import convert_from_path, pdfinfo_from_path

    dpi_candidates = [220, 180, 140] if _check_paddle_gpu_available() else [180, 140, 110]
    info = pdfinfo_from_path(pdf_path)
    total_pages = int(info.get("Pages", 0))
    if total_pages <= 0:
        return []

    images = []
    for page_num in range(1, total_pages + 1):
        page_image = None
        last_error = None
        for dpi in dpi_candidates:
            try:
                page_images = convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    fmt="png",
                    use_pdftocairo=True,
                    first_page=page_num,
                    last_page=page_num,
                    thread_count=1,
                    timeout=60,
                )
                if page_images:
                    page_image = page_images[0]
                    break
            except Exception as e:
                last_error = e
        if page_image is None:
            print(f"    [WARN] 第 {page_num} 页渲染失败: {last_error}")
            continue
        images.append(page_image)
    return images


def _ocr_image_with_paddle(image, use_gpu: bool) -> str:
    from paddleocr import PaddleOCR
    import numpy as np

    global _PADDLE_OCR_CACHE
    cache_key = "gpu" if use_gpu else "cpu"
    if _PADDLE_OCR_CACHE.get(cache_key) is None:
        _PADDLE_OCR_CACHE[cache_key] = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            use_gpu=use_gpu,
            show_log=False,
        )
    ocr = _PADDLE_OCR_CACHE[cache_key]
    image_np = np.array(image)
    result = ocr.ocr(image_np, cls=True)
    lines = []
    if result:
        for page in result:
            for item in page:
                try:
                    text = item[1][0]
                    if text:
                        lines.append(text)
                except Exception:
                    continue
    return "\n".join(lines)


def _score_page_text(text: str, markers: List[str]) -> int:
    normalized = _normalize_text(text)
    score = 0
    if any(_normalize_text(marker) in normalized for marker in markers):
        score += 4
    score += min(len(text) // 80, 3)
    score += min(len(re.findall(r"[一二三四五六七八九十0-9]", text)), 2)
    score += 1 if re.search(r"第[一二三四五六七八九十0-9]+章|chapter\s*\d+", text, re.IGNORECASE) else 0
    return score


def _find_start_page(images, use_gpu: bool, start_markers: List[str]) -> int:
    best_page = 1
    best_score = -1
    for idx, image in enumerate(images, start=1):
        if image.mode != "RGB":
            image = image.convert("RGB")
        try:
            text = _ocr_image_with_paddle(image, use_gpu=use_gpu)
        except Exception as e:
            print(f"    [WARN] 起始页检测失败 第 {idx} 页: {type(e).__name__}: {e}")
            continue
        score = _score_page_text(text, start_markers)
        if score > best_score:
            best_score = score
            best_page = idx
    return max(1, best_page)


def _split_ocr_page_text(text: str) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if not text:
        return []
    parts: List[str] = []
    buffer: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if buffer:
                parts.append("\n".join(buffer).strip())
                buffer = []
            continue
        if re.match(r"^(第\s*[一二三四五六七八九十百千零〇两\d]+\s*章|Chapter\s*\d+|Part\s*[IVXLC0-9]+)", line, re.IGNORECASE):
            if buffer:
                parts.append("\n".join(buffer).strip())
                buffer = []
            parts.append(line)
            continue
        buffer.append(line)
    if buffer:
        parts.append("\n".join(buffer).strip())
    return [p for p in parts if p]


def _extract_pdf_markdown(pdf_path: str) -> str:
    if not _check_pdf2image_available():
        raise RuntimeError("当前环境未安装 pdf2image")
    if not _check_paddleocr_available():
        raise RuntimeError("当前环境未安装 paddleocr")
    if not _check_poppler_available():
        raise RuntimeError("未找到 poppler 命令，请确认已加入系统环境变量")

    start_markers = ["目录", "目 录", "Contents", "第一章", "第1章", "Chapter 1", "Chap ter 1"]
    end_markers = ["参考文献", "参考资料", "附录", "索引", "后记", "后 记", "致谢", "作者简介"]
    use_gpu = _check_paddle_gpu_available()
    print(f"    OCR 模式: {'GPU' if use_gpu else 'CPU'}")

    images = _render_pdf_to_images(pdf_path)
    if not images:
        return ""

    # 起始页只用于跳过明显的封面；保留前一页避免 OCR 对章节首页误判。
    start_page = max(1, _find_start_page(images, use_gpu=use_gpu, start_markers=start_markers) - 1)
    print(f"    解析范围: 第 {start_page} 页到第 {len(images)} 页（总页数 {len(images)}）")

    parts = []
    for idx, image in enumerate(images, start=1):
        if idx < start_page:
            continue

        print(f"    识别第 {idx}/{len(images)} 页")
        if image.mode != "RGB":
            image = image.convert("RGB")
        try:
            text = _ocr_image_with_paddle(image, use_gpu=use_gpu)
        except Exception as e:
            print(f"    [WARN] 第 {idx} 页 OCR 失败: {type(e).__name__}: {e}")
            text = ""
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            continue
        # 参考文献等标记只作为页面元数据保留，不中断解析，以免误删正文附录。
        page_parts = _split_ocr_page_text(text) or [text]
        paragraph_blocks = [f"## 段落 {part_index}\n\n{part}" for part_index, part in enumerate(page_parts, start=1)]
        marker_note = "\n\n> 页面可能包含尾部材料标记" if _page_contains_any_marker(text, end_markers) else ""
        parts.append(f"# 第 {idx} 页\n\n" + "\n\n".join(paragraph_blocks) + marker_note)

    return "\n\n".join(parts).strip()


def _strip_txt_noise(text: str) -> str:
    """仅清理确定无语义的行；章节标题、短术语和列表项均应保留。"""
    cleaned: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            cleaned.append("")
            continue
        if re.fullmatch(r"(?:[-_—·.。]|\d{1,4})", s):
            continue
        if re.fullmatch(r"(?:第?\s*\d+\s*页?)", s, re.IGNORECASE):
            continue
        cleaned.append(s)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def _extract_txt_markdown(txt_path: str) -> str:
    text = ""
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            text = Path(txt_path).read_text(encoding=enc, errors="ignore")
            break
        except Exception:
            continue
    if not text:
        return ""

    # TXT 的目录、参考文献提示可能也是正文的一部分，统一保留并依赖后续结构识别。
    body = _strip_txt_noise(text)
    if not body:
        return ""
    blocks = [block.strip() for block in re.split(r"\n\s*\n", body) if block.strip()]
    return "# 第 1 页\n\n" + "\n\n".join(
        f"## 段落 {index}\n\n{block}" for index, block in enumerate(blocks, start=1)
    )


def _sanitize_folder_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "未命名"
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:80] or "未命名"


def _book_output_dir(out_dir: Path, source_path: Path) -> Path:
    return out_dir / _sanitize_folder_name(source_path.stem)


def _source_outputs_exist(source_path: Path, out_dir: Path) -> bool:
    book_dir = _book_output_dir(out_dir, source_path)
    if source_path.suffix.lower() == ".epub":
        return (book_dir / f"{source_path.stem}.json").exists()
    return (book_dir / f"{source_path.stem}.mk").exists() and (book_dir / f"{source_path.stem}.json").exists()


def _mk_for_source_file(source_path: Path, out_dir: Path) -> Path:
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        mk_content = _extract_pdf_markdown(str(source_path))
    elif suffix == ".txt":
        mk_content = _extract_txt_markdown(str(source_path))
    else:
        raise ValueError(f"不支持的文件类型: {source_path.name}")

    book_dir = _book_output_dir(out_dir, source_path)
    book_dir.mkdir(parents=True, exist_ok=True)
    mk_path = book_dir / f"{source_path.stem}.mk"
    with open(mk_path, "w", encoding="utf-8") as f:
        f.write(mk_content)
    return mk_path


def _is_noise_text(text: str) -> bool:
    s = re.sub(r"\s+", "", text or "")
    if not s:
        return True
    if len(s) < 4:
        return True
    if re.fullmatch(r"[\W_]+", s):
        return True
    if re.fullmatch(r"\d+", s):
        return True
    noise_patterns = [
        r"目录", r"contents?", r"copyright", r"isbn", r"版权所有", r"图书在版编目",
        r"参考文献", r"附录", r"索引", r"作者简介", r"致谢", r"前言", r"序言", r"导言", r"引言",
    ]
    return any(re.search(p, s, re.IGNORECASE) for p in noise_patterns)


def _is_heading_like(text: str) -> bool:
    s = re.sub(r"\s+", " ", (text or "")).strip()
    if not s:
        return False
    heading_patterns = [
        r"^第\s*[一二三四五六七八九十百千零〇两\d]+\s*章(?:\b|\s*[:：、-]?)",
        r"^第\s*[一二三四五六七八九十百千零〇两\d]+\s*节(?:\b|\s*[:：、-]?)",
        r"^Chapter\s*\d+(?:\b|\s*[:：、-]?)",
        r"^Part\s*[IVXLC0-9]+(?:\b|\s*[:：、-]?)",
        r"^Section\s*\d+(?:\b|\s*[:：、-]?)",
        r"^前言$", r"^序言$", r"^导言$", r"^引言$",
    ]
    if any(re.search(p, s, re.IGNORECASE) for p in heading_patterns):
        return True
    return len(s) <= 24 and bool(re.search(r"[一二三四五六七八九十百千零〇两\d]", s))


def _normalize_heading_title(text: str) -> str:
    s = re.sub(r"\s+", " ", (text or "")).strip()
    s = re.sub(r"^(Chapter|Section|Part)\s*", lambda m: m.group(1) + " ", s, flags=re.IGNORECASE)
    return s


def _should_keep_epub_title(title: str, content_len: int = 0) -> bool:
    s = re.sub(r"\s+", "", title or "")
    if not s:
        return False
    if _is_noise_text(s):
        return False
    if _is_heading_like(s):
        return True
    if title.strip() in {"前言", "序言", "导言", "引言"}:
        return True
    # EPUB 的导航标题通常没有正文长度；短且非噪声的标题同样值得保留。
    return len(s) <= 60 and (content_len >= 20 or len(s) <= 30)


def _epub_to_json(source_path: Path, out_dir: Path) -> Path:
    book_dir = _book_output_dir(out_dir, source_path)
    book_dir.mkdir(parents=True, exist_ok=True)
    json_path = book_dir / f"{source_path.stem}.json"
    raw_json_path = book_dir / f"{source_path.stem}.raw.json"
    if json_path.exists() and raw_json_path.exists():
        return json_path

    if not _check_ebooklib_available():
        raise RuntimeError("当前环境未安装 ebooklib")

    from ebooklib import epub as epub_lib
    from bs4 import BeautifulSoup

    book = epub_lib.read_epub(str(source_path))
    root = {"title": source_path.stem, "level": 0, "type": "book", "content": "", "children": []}
    stack = [root]

    def push_heading(level: int, title: str):
        normalized = _normalize_heading_title(title)
        while len(stack) > 1 and stack[-1]["level"] >= level:
            stack.pop()
        node = {"title": normalized, "level": level, "type": "heading", "content": "", "children": []}
        stack[-1]["children"].append(node)
        stack.append(node)
        return node

    def add_paragraph(text: str):
        if not text or _is_noise_text(text):
            return
        parent = stack[-1]
        node = {"title": "", "level": parent["level"] + 1, "type": "paragraph", "content": text, "children": []}
        parent["children"].append(node)

    for item in book.get_items():
        if item.get_type() != 9:
            continue
        soup = BeautifulSoup(item.content, "html.parser")
        for elem in soup.find_all(["h1", "h2", "h3", "h4", "p"]):
            text = elem.get_text(" ", strip=True)
            if not text:
                continue
            if elem.name in ("h1", "h2", "h3", "h4"):
                if _should_keep_epub_title(text):
                    push_heading(int(elem.name[1]), text)
                continue
            add_paragraph(text)

    def prune_noise_paragraph_nodes(node: dict) -> list[dict]:
        pruned_children = []
        for child in node.get("children", []):
            child["children"] = prune_noise_paragraph_nodes(child)
            content = str(child.get("content", "")).strip()
            is_noise_paragraph = (
                child.get("type") == "paragraph"
                and (not content or re.fullmatch(r"[\W_\d]+", content))
                and len(content) < 12
            )
            if not is_noise_paragraph:
                pruned_children.append(child)
        return pruned_children

    raw_root = copy.deepcopy(root)
    raw_json_path.write_text(json.dumps(raw_root, ensure_ascii=False, indent=2), encoding="utf-8")
    root["children"] = prune_noise_paragraph_nodes(root)
    json_path.write_text(json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def batch_parse_sources_to_mk(source_folder: str, parse_doc_folder: str) -> List[Path]:
    source_files = _find_source_files(source_folder)
    if not source_files:
        print(f"未找到支持的文件: {source_folder}")
        return []

    out_dir = Path(parse_doc_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"发现 {len(source_files)} 个文件")
    print(f"pdf2image: {'已安装' if _check_pdf2image_available() else '未安装'}")
    print(f"PaddleOCR: {'已安装' if _check_paddleocr_available() else '未安装'}")
    print(f"ebooklib: {'已安装' if _check_ebooklib_available() else '未安装'}")
    print(f"Paddle GPU: {'可用' if _check_paddle_gpu_available() else '不可用，使用 CPU'}")
    print(f"Poppler: {'已检测到' if _check_poppler_available() else '未检测到'}")

    generated: List[Path] = []
    for source_path in source_files:
        try:
            if _source_outputs_exist(source_path, out_dir):
                print(f"已存在，跳过: {source_path.name}")
                continue
            print(f"开始解析: {source_path.name}")
            if source_path.suffix.lower() == ".epub":
                json_path = _epub_to_json(source_path, out_dir)
                generated.append(json_path)
                print(f"已保存: {json_path}")
            else:
                mk_path = _mk_for_source_file(source_path, out_dir)
                generated.append(mk_path)
                print(f"已保存: {mk_path}")
        except Exception as e:
            print(f"解析失败 {source_path.name}: {e}")
    return generated


# ============================================================
# Part 2: MK -> JSON chapter tree
# ============================================================


CHAPTER_HEADER_RE = re.compile(r"^第\s*(\d+)\s*章\s*$")
CHAPTER_WITH_TITLE_RE = re.compile(r"^第\s*(\d+)\s*章(?:\s*|\s*[:：、-]?)\s*([\u4e00-\u9fffA-Za-z0-9《》“”\(\)（）·、，。！？!?—\-\s]{1,80})$")
PAGE_MARK_RE = re.compile(r"^#\s*第\s*(\d+)\s*页\s*$")
SPLIT_CHAPTER_NUMBER_RE = re.compile(r"^\d+$")
CHINESE_CHAPTER_RE = re.compile(r"^第([一二三四五六七八九十百千零〇两]+)章$")
CHINESE_CHAPTER_WITH_TITLE_RE = re.compile(r"^第([一二三四五六七八九十百千零〇两]+)章(?:\s*|\s*[:：、-]?)\s*([\u4e00-\u9fffA-Za-z0-9《》“”\(\)（）·、，。！？!?—\-\s]{1,80})$")

CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

OCR_NUMBER_FIXES = {
    8: (5, 6, 9),
    9: (6, 8),
    6: (8, 9),
    5: (8,),
}

MAX_CHILD_CONTENT_LEN = 500
A2_PREFIX_LEN = 50


@dataclass
class ChapterSegment:
    chapter_number: int
    title: str = ""
    body: list[str] = field(default_factory=list)


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def collapse_body_lines(lines: list[str]) -> str:
    return "".join(line.strip() for line in lines if line.strip())


def chinese_to_int(text: str) -> int | None:
    if not text:
        return None
    if text in CHINESE_DIGITS:
        return CHINESE_DIGITS[text]
    if text == "十":
        return 10
    if "十" in text:
        left, right = text.split("十", 1)
        tens = CHINESE_DIGITS.get(left, 1) if left else 1
        ones = CHINESE_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


def parse_chapter_number(line: str) -> int | None:
    match = CHAPTER_HEADER_RE.match(line)
    if match:
        return int(match.group(1))
    match = CHINESE_CHAPTER_RE.match(line)
    if match:
        return chinese_to_int(match.group(1))
    match = CHAPTER_WITH_TITLE_RE.match(line)
    if match:
        return int(match.group(1))
    match = CHINESE_CHAPTER_WITH_TITLE_RE.match(line)
    if match:
        return chinese_to_int(match.group(1))
    return None


def normalize_expected_number(previous_number: int | None, observed_number: int) -> int | None:
    if previous_number is None:
        return observed_number
    expected = previous_number + 1
    if observed_number == expected:
        return observed_number
    if observed_number in OCR_NUMBER_FIXES and expected in OCR_NUMBER_FIXES[observed_number]:
        return expected
    return None


def split_long_content(content: str) -> list[str]:
    if len(content) <= MAX_CHILD_CONTENT_LEN:
        return [content]
    chunks: list[str] = []
    start = 0
    while start < len(content):
        end = min(start + MAX_CHILD_CONTENT_LEN, len(content))
        chunk = content[start:end]
        if start > 0:
            chunk = content[max(0, start - A2_PREFIX_LEN):end]
        chunks.append(chunk)
        if end == len(content):
            break
        start = end
    return chunks


def build_section_node(title: str, content: str, level: int = 2, node_type: str = "section") -> dict:
    return {"title": title, "level": level, "type": node_type, "content": content, "children": []}


def build_chapter_node(chapter_number: int, title: str, body_lines: list[str]) -> dict:
    content = collapse_body_lines(body_lines)
    chunks = split_long_content(content)
    children = [build_section_node(f"第{chapter_number}章-A{index}", chunk) for index, chunk in enumerate(chunks, start=1)]
    return {"title": title or f"第{chapter_number}章", "level": 1, "type": "chapter", "content": "", "children": children}


def extract_chapter_title(line: str) -> str | None:
    match = CHAPTER_WITH_TITLE_RE.match(line)
    if match:
        return match.group(2).strip()
    match = CHINESE_CHAPTER_WITH_TITLE_RE.match(line)
    if match:
        return match.group(2).strip()
    return None


def build_book_root(mk_path: Path) -> dict:
    return {"title": mk_path.stem, "level": 0, "type": "book", "content": "", "children": []}


def should_start_new_chapter(line: str, previous_number: int | None) -> int | None:
    parsed = parse_chapter_number(line)
    if parsed is None:
        return None
    if previous_number is None:
        return parsed
    if parsed == previous_number:
        return None
    if parsed == previous_number + 1:
        return parsed
    return normalize_expected_number(previous_number, parsed)


def _annotate_node_paths(node: dict, book_title: str, parent_path: Optional[List[str]] = None, parent_id: str = "root", counters: Optional[dict] = None) -> None:
    parent_path = parent_path or []
    counters = counters or {"chapter": 0, "section": 0, "paragraph": 0}
    title = str(node.get("title", "")).strip()
    node_type = str(node.get("type", "")).strip()
    current_path = parent_path + ([title] if title else [])
    node["title_path"] = " > ".join([book_title] + current_path) if current_path else book_title
    node["node_id"] = f"{parent_id}:{node_type}:{len(current_path)}:{title[:20]}"
    node["parent_id"] = parent_id
    if node_type == "chapter":
        counters["chapter"] += 1
        node["chapter_index"] = counters["chapter"]
    elif node_type == "section":
        counters["section"] += 1
        node["section_index"] = counters["section"]
    elif node_type == "paragraph":
        counters["paragraph"] += 1
        node["paragraph_index"] = counters["paragraph"]
    for child in node.get("children", []):
        if isinstance(child, dict):
            _annotate_node_paths(child, book_title, current_path, node["node_id"], counters)


def build_chapter_tree_from_mk(mk_path: Path, json_path: Optional[Path] = None) -> Path:
    json_path = json_path or mk_path.with_suffix(".json")
    lines = read_lines(mk_path)
    book = build_book_root(mk_path)
    title_map: dict[int, dict] = {}

    current: ChapterSegment | None = None
    pending_number: int | None = None

    def create_chapter_node(number: int, chapter_title: str | None = None) -> dict:
        node = {"title": chapter_title or f"第{number}章", "level": 1, "type": "chapter", "content": "", "children": []}
        title_map[number] = node
        book["children"].append(node)
        return node

    def flush_current() -> None:
        nonlocal current
        if current is None:
            return
        built = build_chapter_node(current.chapter_number, current.title, current.body)
        current_node = title_map.get(current.chapter_number)
        if current_node is not None:
            current_node.update({"title": built["title"], "content": built["content"], "children": built["children"]})
        current = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if PAGE_MARK_RE.match(line):
            continue
        if SPLIT_CHAPTER_NUMBER_RE.match(line):
            pending_number = int(line)
            continue
        if line == "第" and pending_number is not None:
            continue
        if line == "章" and pending_number is not None:
            candidate = pending_number
            pending_number = None
            existing = title_map.get(candidate)
            if existing is not None:
                flush_current()
                current = ChapterSegment(chapter_number=candidate, title=str(existing.get("title", f"第{candidate}章")))
                continue
            chapter_number = should_start_new_chapter(f"第{candidate}章", current.chapter_number if current else None)
            if chapter_number is not None:
                flush_current()
                current = ChapterSegment(chapter_number=chapter_number)
                create_chapter_node(chapter_number)
            continue

        parsed_line_number = parse_chapter_number(line)
        if parsed_line_number is not None:
            chapter_title = extract_chapter_title(line) or f"第{parsed_line_number}章"
            existing = title_map.get(parsed_line_number)
            if existing is not None:
                existing["title"] = chapter_title
                flush_current()
                current = ChapterSegment(chapter_number=parsed_line_number, title=chapter_title)
                continue

            chapter_number = should_start_new_chapter(line, current.chapter_number if current else None)
            if chapter_number is not None:
                flush_current()
                current = ChapterSegment(chapter_number=chapter_number, title=chapter_title)
                create_chapter_node(chapter_number, chapter_title)
                continue

            if current is not None:
                current.body.append(line)
            pending_number = None
            continue

        pending_number = None
        if current is None:
            continue
        current.body.append(line)

    flush_current()
    book["children"] = [child for child in book["children"] if isinstance(child, dict)]
    _annotate_node_paths(book, book.get("title", mk_path.stem))
    json_path.write_text(json.dumps(book, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


# ============================================================
# CLI
# ============================================================


def _run_one_click_pipeline(source_folder: str, out_folder: str) -> None:
    source_files = _find_source_files(source_folder)
    if not source_files:
        print(f"未找到支持的文件: {source_folder}")
        return

    out_dir = Path(out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    for source_path in source_files:
        try:
            if _source_outputs_exist(source_path, out_dir):
                print(f"已存在，跳过: {source_path.name}")
                continue

            print(f"开始解析: {source_path.name}")
            if source_path.suffix.lower() == ".epub":
                json_path = _epub_to_json(source_path, out_dir)
                print(f"已生成 JSON: {json_path}")
            else:
                mk_path = _mk_for_source_file(source_path, out_dir)
                print(f"已保存 MK: {mk_path}")
                json_path = build_chapter_tree_from_mk(mk_path)
                print(f"已生成 JSON: {json_path}")
        except Exception as e:
            print(f"处理失败 {source_path.name}: {e}")


def _iter_supported_files(folder: str) -> list[Path]:
    return _find_source_files(folder)


def main() -> None:
    parser = argparse.ArgumentParser(description="整理版书籍处理工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    full_parser = subparsers.add_parser("run-all", help="一次运行：PDF / EPUB / TXT -> MK -> JSON")
    full_parser.add_argument("--source-folder", default=r"D:\Desktop\books")
    full_parser.add_argument("--out-folder", default=r"D:\Desktop\解析文档")

    pdf_parser = subparsers.add_parser("parse-pdf", help="仅批量把 PDF 转成 .mk")
    pdf_parser.add_argument("--source-folder", default=r"D:\Desktop\books")
    pdf_parser.add_argument("--out-folder", default=r"D:\Desktop\解析文档")

    tree_parser = subparsers.add_parser("build-tree", help="从 .mk 构建章节树 JSON")
    tree_parser.add_argument("--mk-path", default=str(Path(r"D:\Desktop\openai\动物百科多agent项目\解析文档\别跟狗争老大.mk")))
    tree_parser.add_argument("--json-path", default="")

    args = parser.parse_args()

    if args.command == "run-all":
        _run_one_click_pipeline(args.source_folder, args.out_folder)
        return

    if args.command == "parse-pdf":
        pdf_only = [p for p in _iter_supported_files(args.source_folder) if p.suffix.lower() == ".pdf"]
        if not pdf_only:
            print(f"未找到 PDF 文件: {args.source_folder}")
            return
        batch_parse_sources_to_mk(args.source_folder, args.out_folder)
        return

    if args.command == "build-tree":
        mk_path = Path(args.mk_path)
        json_path = Path(args.json_path) if args.json_path else None
        out = build_chapter_tree_from_mk(mk_path, json_path)
        print(f"已生成: {out}")
        return


if __name__ == "__main__":
    main()
