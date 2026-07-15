"""JSON 入库版书籍工具。

职责：
- 读取 `解析文档` 下的 JSON
- 将 JSON 树展开为多粒度 Chunk
- 构建知识库（向量检索 + BM25）
- 提供 search_pdf / read_page / answer_question

说明：
- 不再包含 PDF / EPUB / TXT 预处理
- 重点保留章节层级、父子关系与检索增强元数据
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

load_dotenv()

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except Exception:
    SENTENCE_TRANSFORMER_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ============================================================
# 配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PARSE_DOC_DIR = (BASE_DIR / ".." / ".." / "解析文档").resolve()
CACHE_FILE = BASE_DIR / "knowledge_cache.pkl"
VECTOR_INDEX_FILE = BASE_DIR / "vector_index.pkl"
BM25_INDEX_FILE = BASE_DIR / "bm25_index.pkl"
FAILED_CHUNKS_FILE = BASE_DIR / "failed_chunks.json"
FAISS_DIR = Path(r"D:\Desktop\FIASS")
FAISS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = r"C:\Users\ls\.cache\sentence_transformers\text2vec-base-chinese"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
SUMMARY_MAX_LEN = 160
SUMMARY_BATCH_SIZE = 3
CACHE_SCHEMA_VERSION = 2


# ============================================================
# 工具函数
# ============================================================


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def _generate_summary(text: str, max_len: int = SUMMARY_MAX_LEN) -> str:
    """提取前一至两句作为检索特征，而不是展示用的机械截断。"""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_len:
        return text
    sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])\s*", text) if s.strip()]
    selected: List[str] = []
    for sentence in sentences:
        candidate = " ".join(selected + [sentence])
        if selected and len(candidate) > max_len:
            break
        selected.append(sentence)
        if len(candidate) >= max_len * 0.65:
            break
    if selected:
        return " ".join(selected)[:max_len].rstrip()
    return text[:max_len].rstrip() + "..."


def _combine_for_index(chunk: "Chunk") -> str:
    parts = [
        f"书名：{chunk.book_title}",
        f"别名：{'、'.join(chunk.book_aliases) if chunk.book_aliases else ''}",
        f"路径：{chunk.title_path}",
        f"章节：{chunk.section_title}",
        f"类型：{chunk.chunk_type}",
        f"摘要：{chunk.summary}",
        f"正文：{chunk.content}",
    ]
    return "\n".join([p for p in parts if p and not p.endswith("：")])


def _extract_proper_nouns(text: str) -> List[str]:
    candidates = ["阿拉伯野猫", "利比亚猫", "新月沃地", "幼态持续", "neotenization", "身体屏障法", "狗的秘密", "猫的秘密"]
    return [c for c in candidates if c in text]


def _extract_numeric_facts(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"\d+(?:\.\d+)?%",
        r"\d+(?:\.\d+)?\s*[倍]",
        r"\d+(?:\.\d+)?\s*(?:℃|°[CcFf]?|度|小时|分钟|秒钟|秒|天|年|岁|个月|月|次|公斤|kg|克|g|米|m|厘米|cm|毫米|mm|升|L|毫升|ml)",
        r"\d+(?:\.\d+)?\s*[-~—–至到]\s*\d+(?:\.\d+)?",
    ]
    found = []
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            val = m.group().strip()
            if val not in found:
                found.append(val)
    return ", ".join(found[:10])


def _infer_book_aliases(book_title: str, file_name: str) -> List[str]:
    aliases = []
    for candidate in [book_title, Path(file_name).stem, re.sub(r"\s*[（(].*$", "", book_title or "").strip()]:
        candidate = (candidate or "").strip()
        if candidate and candidate not in aliases:
            aliases.append(candidate)
    return aliases


def _repair_mojibake(value: Any) -> Any:
    """修复上游解析器将 UTF-8 文本经 latin-1 错解后再次写出的常见乱码。"""
    if isinstance(value, str) and _looks_like_mojibake(value):
        for source_encoding in ("latin1", "cp1252"):
            try:
                repaired = value.encode(source_encoding).decode("utf-8")
                if not _looks_like_mojibake(repaired):
                    return repaired
            except UnicodeError:
                continue
    if isinstance(value, list):
        return [_repair_mojibake(item) for item in value]
    if isinstance(value, dict):
        return {key: _repair_mojibake(item) for key, item in value.items()}
    return value


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return _repair_mojibake(json.load(f))


def _iter_json_files(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        return []
    return sorted([p for p in root_dir.rglob("*.json") if not p.name.endswith(".raw.json")])


def _source_manifest(json_root: Path) -> str:
    """用源文件路径、大小和修改时间识别过期索引。"""
    digest = hashlib.sha256()
    for path in _iter_json_files(json_root):
        stat = path.stat()
        digest.update(f"{path.relative_to(json_root)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def _looks_like_mojibake(value: str) -> bool:
    return value.count("�") >= 2 or "Ã" in value or ("�" in value and any(ord(char) > 127 for char in value))


def _json_has_content(node: Dict[str, Any]) -> bool:
    """接受不同解析器的树形节点；只要任意后代带正文即可索引。"""
    if str(node.get("content", "")).strip():
        return True
    return any(isinstance(ch, dict) and _json_has_content(ch) for ch in (node.get("children", []) or []))


def _pick_title_path(node: Dict[str, Any], fallback: str) -> str:
    return str(node.get("title_path") or fallback or "").strip()


def _make_chunk_uid(file_name: str, node_id: str, chunk_index: int, chunk_type: str) -> str:
    return f"{file_name}::{node_id}::{chunk_type}::{chunk_index}"


def _node_text(node: Dict[str, Any]) -> str:
    content = str(node.get("content", "") or "").strip()
    title = str(node.get("title", "") or "").strip()
    if title and content:
        return f"{title}\n{content}"
    return title or content


def _sliding_windows(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    windows: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(text.rfind(mark, start + size // 2, end) for mark in "。！？；")
            if boundary >= start + size // 2:
                end = boundary + 1
        windows.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return windows


# ============================================================
# Chunk
# ============================================================


@dataclass
class Chunk:
    content: str
    file_name: str
    book_title: str
    page_num: int
    title_path: str
    content_type: str
    chunk_index: int
    chunk_uid: str = ""
    tags: List[str] = field(default_factory=list)
    summary: str = ""
    section_title: str = ""
    doc_type: str = "JSON"
    entity_tags: List[str] = field(default_factory=list)
    token_count: int = 0
    parent_id: Optional[str] = None
    node_id: str = ""
    chunk_type: str = "paragraph"
    hierarchy_level: int = 0
    siblings_ids: List[str] = field(default_factory=list)
    numeric_summary: str = ""
    summary_tags: List[str] = field(default_factory=list)
    concept_tags: List[str] = field(default_factory=list)
    proper_nouns: List[str] = field(default_factory=list)
    book_aliases: List[str] = field(default_factory=list)
    block_coords: Optional[List] = None
    index_text: str = ""
    source_file: str = ""
    source_node_path: str = ""
    children_count: int = 0

    def __post_init__(self):
        if not self.summary:
            self.summary = _generate_summary(self.content)
        if not self.section_title and self.title_path:
            self.section_title = self.title_path.split(" > ")[-1]
        if not self.token_count:
            self.token_count = len(list(jieba.cut(self.content)))
        if not self.index_text:
            self.index_text = _combine_for_index(self)
        if not self.source_file:
            self.source_file = self.file_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "file_name": self.file_name,
            "book_title": self.book_title,
            "page_num": self.page_num,
            "title_path": self.title_path,
            "content_type": self.content_type,
            "chunk_index": self.chunk_index,
            "chunk_uid": self.chunk_uid,
            "tags": self.tags,
            "summary": self.summary,
            "section_title": self.section_title,
            "doc_type": self.doc_type,
            "entity_tags": self.entity_tags,
            "token_count": self.token_count,
            "parent_id": self.parent_id,
            "node_id": self.node_id,
            "chunk_type": self.chunk_type,
            "hierarchy_level": self.hierarchy_level,
            "siblings_ids": self.siblings_ids,
            "numeric_summary": self.numeric_summary,
            "summary_tags": self.summary_tags,
            "concept_tags": self.concept_tags,
            "proper_nouns": self.proper_nouns,
            "book_aliases": self.book_aliases,
            "block_coords": self.block_coords,
            "index_text": self.index_text,
            "source_file": self.source_file,
            "source_node_path": self.source_node_path,
            "children_count": self.children_count,
        }


# ============================================================
# JSON -> Chunk
# ============================================================


def _walk_json_tree(node: Dict[str, Any], file_name: str, book_title: str, parent_path: Optional[List[str]] = None, parent_id: str = "root", depth: int = 0) -> List[Dict[str, Any]]:
    parent_path = parent_path or []
    title = str(node.get("title", "") or "").strip()
    node_type = str(node.get("type", "") or "").strip() or "paragraph"
    current_path = parent_path + ([title] if title else [])
    title_path = " > ".join([book_title] + current_path) if current_path else book_title
    node_id = str(node.get("node_id") or f"{parent_id}:{node_type}:{len(current_path)}:{title[:24]}")

    current = {
        "node_id": node_id,
        "parent_id": parent_id,
        "title": title,
        "title_path": title_path,
        "type": node_type,
        "level": int(node.get("level", depth) or depth),
        "content": str(node.get("content", "") or "").strip(),
        "children": [],
        "file_name": file_name,
        "book_title": book_title,
        "source_node_path": title_path,
        "depth": depth,
    }
    children = node.get("children", []) or []
    for child in children:
        if isinstance(child, dict):
            current["children"].append(_walk_json_tree(child, file_name, book_title, current_path, node_id, depth + 1))
    current["children_count"] = len(current["children"])
    return current


def _collect_nodes(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes = [tree]
    for child in tree.get("children", []):
        nodes.extend(_collect_nodes(child))
    return nodes


def _build_chunks_from_json_tree(node: Dict[str, Any], file_name: str, book_title: str, chunks: Optional[List[Chunk]] = None) -> Tuple[List[Chunk], int]:
    """从树构造段落、节摘要和章节摘要，并以节点 ID 保留稳定关系。"""
    chunks = chunks or []
    flat_nodes = _collect_nodes(node)
    chunk_index = len(chunks)
    sibling_map: DefaultDict[str, List[str]] = defaultdict(list)
    for item in flat_nodes:
        sibling_map[str(item.get("parent_id", ""))].append(str(item.get("node_id", "")))

    def descendant_text(item: Dict[str, Any]) -> str:
        texts = [_node_text(item)] if _node_text(item) else []
        for child in item.get("children", []):
            texts.append(descendant_text(child))
        return "\n".join(text for text in texts if text).strip()

    for item in flat_nodes:
        node_type = str(item.get("type", "paragraph"))
        if node_type == "book":
            continue
        # 不同解析器会将正文标为 paragraph 或 section；只有叶子正文节点才写入索引，避免章节全文重复。
        is_leaf_content = bool(str(item.get("content", "")).strip()) and not item.get("children")
        if node_type != "paragraph" and not is_leaf_content:
            continue
        content = _node_text(item)
        if not content:
            continue
        title_path = _pick_title_path(item, book_title)
        hierarchy_level = int(item.get("level", 0) or 0)
        node_id = str(item.get("node_id", ""))
        parent_id = str(item.get("parent_id", "")) or None
        for window_index, window in enumerate(_sliding_windows(content)):
            window_node_id = f"{node_id}:window:{window_index}"
            chunk = Chunk(
                content=window, file_name=file_name, book_title=book_title,
                page_num=int(item.get("page_num", 1) or 1), title_path=title_path,
                content_type=node_type, chunk_index=chunk_index,
                chunk_uid=_make_chunk_uid(file_name, window_node_id, chunk_index, "evidence_window"),
                section_title=title_path.split(" > ")[-1] if title_path else book_title,
                parent_id=parent_id, node_id=window_node_id, chunk_type="evidence_window",
                hierarchy_level=hierarchy_level,
                siblings_ids=[], numeric_summary=_extract_numeric_facts(window),
                proper_nouns=_extract_proper_nouns(window + " " + title_path),
                book_aliases=_infer_book_aliases(book_title, file_name), source_file=file_name,
                source_node_path=title_path, children_count=0,
            )
            chunks.append(chunk)
            chunk_index += 1
    return chunks, chunk_index


def load_json_books(json_root: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in _iter_json_files(json_root):
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        if not _json_has_content(data):
            continue
        book_title = str(data.get("title") or path.stem).strip() or path.stem
        file_name = path.name
        tree = _walk_json_tree(data, file_name=file_name, book_title=book_title)
        book_chunks, _ = _build_chunks_from_json_tree(tree, file_name=file_name, book_title=book_title, chunks=[])
        chunks.extend(book_chunks)

    # 段落摘要保持轻量；昂贵的 LLM 摘要仅应在需要时针对章节/小节离线生成。
    return chunks


# ============================================================
# 检索库
# ============================================================


class VectorStore:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.index = None
        self.chunks: List[Chunk] = []
        self.dimension = 0

    def load_model(self):
        if self.model is not None or not SENTENCE_TRANSFORMER_AVAILABLE:
            return
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"[WARN] 向量模型加载失败: {e}")
            self.model = None

    def build(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.load_model()
        if self.model is None or not chunks:
            self.embeddings = None
            self.index = None
            return
        texts = [c.index_text or _combine_for_index(c) for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        self.embeddings = np.array(embeddings, dtype="float32")
        self.dimension = self.embeddings.shape[1]
        if FAISS_AVAILABLE:
            try:
                if len(chunks) < 1000:
                    self.index = faiss.IndexFlatIP(self.dimension)
                else:
                    nlist = min(int(len(chunks) ** 0.5), 256)
                    quantizer = faiss.IndexFlatIP(self.dimension)
                    self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                    self.index.train(self.embeddings)
                self.index.add(self.embeddings)
                print(f"[OK] FAISS 向量库构建完成: {len(chunks)} 条, 维度 {self.dimension}")
            except Exception as e:
                print(f"[WARN] FAISS 构建失败，回退到 numpy：{e}")
                self.index = None

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.model is None or self.embeddings is None or not len(self.chunks):
            return []
        query_emb = self.model.encode([query], normalize_embeddings=True)
        query_emb = np.array(query_emb, dtype="float32")
        if self.index is not None and FAISS_AVAILABLE:
            try:
                if hasattr(self.index, "nprobe") and hasattr(self.index, "nlist"):
                    self.index.nprobe = min(16, getattr(self.index, "nlist", 16))
                k = min(top_k, len(self.chunks))
                distances, indices = self.index.search(query_emb, k)
                pairs = list(zip(distances[0], indices[0]))
            except Exception:
                pairs = []
        else:
            scores = np.dot(self.embeddings, query_emb.reshape(1, -1).T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            pairs = [(scores[idx], idx) for idx in top_indices]

        results = []
        for score, idx in pairs:
            if idx < 0 or score <= 0:
                continue
            c = self.chunks[idx]
            results.append({
                "chunk": c.to_dict(),
                "score": float(score),
                "source": f"[来源: {c.file_name} 第{c.page_num}页]",
                "book_title": c.book_title,
                "page_num": c.page_num,
                "title_path": c.title_path,
                "content_type": c.content_type,
                "tags": c.tags,
                "summary": c.summary,
                "content": c.content,
                "chunk_uid": c.chunk_uid,
                "node_id": c.node_id,
                "parent_id": c.parent_id,
                "chunk_type": c.chunk_type,
                "hierarchy_level": c.hierarchy_level,
            })
        return results

    def save(self, path: str):
        faiss_path = str(FAISS_DIR / "vector_index.faiss")
        data = {
            "model_name": self.model_name,
            "chunks": [c.to_dict() for c in self.chunks],
            "embeddings": self.embeddings,
            "dimension": self.dimension,
            "faiss_path": faiss_path,
        }
        if self.index is not None and FAISS_AVAILABLE:
            try:
                faiss.write_index(self.index, faiss_path)
                print(f"[OK] FAISS 索引已保存: {faiss_path}")
            except Exception as e:
                print(f"[WARN] FAISS 索引保存失败: {e}")
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = cls(data["model_name"])
        store.chunks = [Chunk(**d) for d in data["chunks"]]
        store.embeddings = data.get("embeddings")
        store.dimension = data.get("dimension", 0)
        faiss_path = data.get("faiss_path", str(FAISS_DIR / "vector_index.faiss"))
        if FAISS_AVAILABLE and faiss_path and Path(faiss_path).exists():
            try:
                store.index = faiss.read_index(faiss_path)
                print(f"[OK] FAISS 索引已加载: {faiss_path}")
            except Exception as e:
                print(f"[WARN] FAISS 索引加载失败: {e}")
                store.index = None
        store.load_model()
        return store


class BM25Store:
    def __init__(self):
        self.bm25 = None
        self.chunks: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []

    def build(self, chunks: List[Chunk]):
        self.chunks = chunks
        if not chunks:
            self.bm25 = None
            self.tokenized_corpus = []
            return
        self.tokenized_corpus = [list(jieba.cut(c.index_text or _combine_for_index(c))) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(list(jieba.cut(query)))
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            c = self.chunks[idx]
            results.append({
                "chunk": c.to_dict(),
                "score": float(scores[idx]),
                "source": f"[来源: {c.file_name} 第{c.page_num}页]",
                "book_title": c.book_title,
                "page_num": c.page_num,
                "title_path": c.title_path,
                "content_type": c.content_type,
                "summary": c.summary,
                "content": c.content,
                "chunk_uid": c.chunk_uid,
                "node_id": c.node_id,
                "parent_id": c.parent_id,
                "chunk_type": c.chunk_type,
                "hierarchy_level": c.hierarchy_level,
            })
        return results

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "chunks": [c.to_dict() for c in self.chunks],
                "tokenized_corpus": self.tokenized_corpus,
            }, f)

    @classmethod
    def load(cls, path: str) -> "BM25Store":
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = cls()
        store.chunks = [Chunk(**d) for d in data["chunks"]]
        store.tokenized_corpus = data["tokenized_corpus"]
        store.bm25 = BM25Okapi(store.tokenized_corpus)
        return store


def rrf_fusion(vector_results: List[Dict], bm25_results: List[Dict], top_k: int = 10, k: int = 60) -> List[Dict]:
    score_map: Dict[str, float] = {}
    result_map: Dict[str, Dict] = {}
    for rank, r in enumerate(vector_results):
        key = r.get("chunk_uid") or r.get("chunk", {}).get("chunk_uid")
        if not key:
            continue
        score_map[key] = score_map.get(key, 0) + 1.0 / (k + rank + 1)
        result_map[key] = r
    for rank, r in enumerate(bm25_results):
        key = r.get("chunk_uid") or r.get("chunk", {}).get("chunk_uid")
        if not key:
            continue
        score_map[key] = score_map.get(key, 0) + 1.0 / (k + rank + 1)
        result_map.setdefault(key, r)
    return [result_map[k_] for k_ in sorted(score_map, key=score_map.get, reverse=True)[:top_k]]


class KnowledgeBase:
    def __init__(self):
        self.vector_store: Optional[VectorStore] = None
        self.bm25_store: Optional[BM25Store] = None
        self.all_chunks: List[Chunk] = []
        self.page_chunks: Dict[Tuple[str, int], List[Chunk]] = {}
        self.node_index: Dict[str, Chunk] = {}
        self.parent_map: Dict[str, List[str]] = defaultdict(list)
        self.child_map: Dict[str, List[str]] = defaultdict(list)
        self.term_index: Dict[str, List[int]] = defaultdict(list)
        self.title_index: Dict[str, List[int]] = defaultdict(list)

    def _rebuild_indexes(self):
        self.page_chunks.clear()
        self.node_index.clear()
        self.parent_map.clear()
        self.child_map.clear()
        self.term_index.clear()
        self.title_index.clear()
        for idx, c in enumerate(self.all_chunks):
            self.page_chunks.setdefault((c.file_name, c.page_num), []).append(c)
            self.node_index[c.node_id or c.chunk_uid] = c
            if c.parent_id:
                self.parent_map[c.node_id or c.chunk_uid].append(c.parent_id)
                self.child_map[c.parent_id].append(c.node_id or c.chunk_uid)
            terms = [c.book_title, c.section_title, c.title_path, c.summary, c.numeric_summary, " ".join(c.proper_nouns), " ".join(c.concept_tags), c.content]
            for text in terms:
                for term in set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text or "")):
                    self.term_index[_normalize_text(term)].append(idx)
            for title in [c.book_title, c.section_title, c.title_path] + list(c.book_aliases):
                title_norm = _normalize_text(title)
                if title_norm:
                    self.title_index[title_norm].append(idx)

    def build(self, chunks: List[Chunk]):
        self.all_chunks = sorted(chunks, key=lambda c: (c.file_name, c.page_num, c.chunk_index))
        self.vector_store = VectorStore()
        self.vector_store.build(self.all_chunks)
        self.bm25_store = BM25Store()
        self.bm25_store.build(self.all_chunks)
        self._rebuild_indexes()

    def save(self, cache_path: str = str(CACHE_FILE), vector_path: str = str(VECTOR_INDEX_FILE), bm25_path: str = str(BM25_INDEX_FILE)):
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source_manifest": _source_manifest(PARSE_DOC_DIR),
            "knowledge_base": [c.to_dict() for c in self.all_chunks],
        }
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        if self.vector_store:
            self.vector_store.save(vector_path)
        if self.bm25_store:
            self.bm25_store.save(bm25_path)

    def load_from_cache(self, cache_path: str = str(CACHE_FILE), vector_path: str = str(VECTOR_INDEX_FILE), bm25_path: str = str(BM25_INDEX_FILE)):
        if not (Path(cache_path).exists() and Path(vector_path).exists() and Path(bm25_path).exists()):
            raise FileNotFoundError("知识库索引文件不完整，请运行 import_books.py 重建")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        chunks = data.get("knowledge_base", [])
        if data.get("schema_version") != CACHE_SCHEMA_VERSION or data.get("source_manifest") != _source_manifest(PARSE_DOC_DIR):
            raise RuntimeError("知识库源文件已更新或缓存版本过期，请运行 import_books.py 重建")
        if not chunks or any(_looks_like_mojibake(str(chunk.get("book_title", ""))) for chunk in chunks):
            raise RuntimeError("知识库缓存包含乱码或为空，请运行 import_books.py 重建")
        self.all_chunks = [Chunk(**d) for d in chunks]
        self.vector_store = VectorStore.load(vector_path)
        self.bm25_store = BM25Store.load(bm25_path)
        self._rebuild_indexes()

    def _expand_context(self, chunk: Chunk) -> Dict[str, Any]:
        # 主索引只保存小证据窗口；命中后补齐同段相邻窗口和层级标题。
        siblings = [candidate for candidate in self.all_chunks if candidate.parent_id == chunk.parent_id and candidate.file_name == chunk.file_name]
        siblings.sort(key=lambda candidate: candidate.chunk_index)
        current_index = next((index for index, candidate in enumerate(siblings) if candidate.chunk_uid == chunk.chunk_uid), -1)
        neighbors = siblings[max(0, current_index - 1):current_index + 2] if current_index >= 0 else []
        return {
            "parent_context": chunk.title_path,
            "neighbor_context": "\n\n".join(item.content for item in neighbors if item.chunk_uid != chunk.chunk_uid),
            "parent_title": chunk.title_path,
        }

    def search_pdf(self, query: str, top_k: int = 10, numeric_intent: bool = False, relevant_units: set = None) -> str:
        vector_results = self.vector_store.search(query, top_k=top_k * 3) if self.vector_store else []
        bm25_results = self.bm25_store.search(query, top_k=top_k * 3) if self.bm25_store else []
        fused = rrf_fusion(vector_results, bm25_results, top_k=top_k * 3)
        results = []
        for r in fused:
            c_dict = r.get("chunk", {})
            c = Chunk(**c_dict)
            ctx = self._expand_context(c)
            results.append({
                "rank": len(results) + 1,
                "content": r.get("content", c.content),
                "file_name": c.file_name,
                "book_title": c.book_title,
                "page_num": c.page_num,
                "title_path": c.title_path,
                "section_title": c.section_title,
                "content_type": c.content_type,
                "tags": c.tags,
                "summary": c.summary,
                "score": round(float(r.get("score", 0)), 4),
                "chunk_index": c.chunk_index,
                "content_length": len(c.content),
                "numeric_summary": c.numeric_summary,
                "summary_tags": c.summary_tags,
                "concept_tags": c.concept_tags,
                "proper_nouns": c.proper_nouns,
                "book_aliases": c.book_aliases,
                "source": r.get("source", f"[来源: {c.file_name} 第{c.page_num}页]"),
                "chunk_uid": c.chunk_uid,
                "node_id": c.node_id,
                "parent_id": c.parent_id,
                "chunk_type": c.chunk_type,
                "hierarchy_level": c.hierarchy_level,
                "parent_context": ctx["parent_context"],
                "neighbor_context": ctx["neighbor_context"],
                "retrieval_source": "hybrid_rrf",
            })
        return json.dumps({"results": results[:top_k], "count": len(results[:top_k]), "query": query}, ensure_ascii=False, indent=2)

    def search_by_numeric(self, query: str, top_k: int = 10, relevant_units: set = None) -> List[Dict[str, Any]]:
        relevant_units = relevant_units or set()
        results = []
        for chunk in self.all_chunks:
            numeric_summary = chunk.numeric_summary or _extract_numeric_facts(chunk.content)
            has_relevant_unit = any(unit in chunk.content or unit in numeric_summary for unit in relevant_units)
            if has_relevant_unit or numeric_summary:
                score = 0.0
                if numeric_summary:
                    score += 1.0
                if has_relevant_unit:
                    score += 2.0
                if query and query.lower() in chunk.content.lower():
                    score += 1.5
                results.append({
                    "content": chunk.content,
                    "file_name": chunk.file_name,
                    "book_title": chunk.book_title,
                    "page_num": chunk.page_num,
                    "title_path": chunk.title_path,
                    "section_title": chunk.section_title,
                    "content_type": chunk.content_type,
                    "tags": chunk.tags,
                    "summary": chunk.summary,
                    "score": score,
                    "chunk_index": chunk.chunk_index,
                    "content_length": len(chunk.content),
                    "numeric_summary": numeric_summary,
                    "summary_tags": chunk.summary_tags,
                    "concept_tags": chunk.concept_tags,
                    "proper_nouns": chunk.proper_nouns,
                    "book_aliases": chunk.book_aliases,
                    "chunk_uid": chunk.chunk_uid,
                    "node_id": chunk.node_id,
                    "parent_id": chunk.parent_id,
                    "chunk_type": chunk.chunk_type,
                    "hierarchy_level": chunk.hierarchy_level,
                    "source": f"[来源: {chunk.file_name} 第{chunk.page_num}页]",
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def read_page(self, file_name: str, page_num: int) -> str:
        page_data = self.page_chunks.get((file_name, page_num), [])
        if not page_data:
            page_data = [c for c in self.all_chunks if c.file_name == file_name]
        if not page_data:
            return json.dumps({"error": f"未找到 {file_name} 的内容"}, ensure_ascii=False)
        texts = [c.content for c in page_data]
        full_content = "\n\n".join(texts)
        result = {
            "file_name": file_name,
            "book_title": page_data[0].book_title if page_data else "",
            "page_num": page_num,
            "title_paths": sorted(set(c.title_path for c in page_data if c.title_path)),
            "section_titles": sorted(set(c.section_title for c in page_data if c.section_title)),
            "tags": sorted(set(t for c in page_data for t in c.tags)),
            "chunk_indices": sorted(set(c.chunk_index for c in page_data)),
            "texts": texts,
            "tables": [],
            "images": [],
            "full_content": full_content,
            "page_summary": _generate_summary(full_content, 160),
            "has_table": False,
            "has_image": False,
            "chunk_count": len(page_data),
            "is_fallback_file_view": page_num <= 1,
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    def answer_question(self, query: str, top_k: int = 5) -> str:
        if not OPENAI_AVAILABLE:
            return json.dumps({"error": "OpenAI 客户端不可用，无法调用 LLM"}, ensure_ascii=False)
        search_result = json.loads(self.search_pdf(query, top_k=top_k))
        if not search_result.get("results"):
            return json.dumps({"answer": "未在知识库中找到相关内容。", "citations": [], "search_results": []}, ensure_ascii=False)
        client = OpenAI()
        context_parts = ["=== 检索结果 ==="]
        for r in search_result["results"]:
            context_parts.append(
                f"[{r['source']}] {r['file_name']}\n书名: {r['book_title']}\n标题: {r.get('title_path', '')}\n内容: {r['content']}"
            )
        system_prompt = "你是一个专业的知识问答助手。请只基于上下文回答，不能编造。回答要中文、清晰，并尽量附带来源。"
        user_prompt = f"问题：{query}\n\n上下文：\n" + "\n\n".join(context_parts)
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=1000,
                temperature=0.3,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            return json.dumps({"error": f"LLM 调用失败: {e}"}, ensure_ascii=False)
        return json.dumps({"answer": answer, "citations": [r["source"] for r in search_result["results"][:top_k]], "search_results": search_result["results"]}, ensure_ascii=False, indent=2)


# ============================================================
# 主流程
# ============================================================


def import_books():
    print("=" * 60)
    print("书籍导入工具 - JSON 入库")
    print("=" * 60)
    json_root = PARSE_DOC_DIR
    json_files = _iter_json_files(json_root)
    print(f"[INFO] 发现 {len(json_files)} 个 JSON 文件")
    if not json_files:
        print(f"[WARN] 未找到可入库的 JSON 文件: {json_root}")
        return

    chunks = load_json_books(json_root)
    print(f"[INFO] 展开得到 {len(chunks)} 个切片")
    if not chunks:
        print("[WARN] 没有可入库切片")
        return

    kb = KnowledgeBase()
    kb.build(chunks)
    kb.save()
    print("[OK] 知识库已保存")

    if kb.all_chunks:
        try:
            test = json.loads(kb.search_pdf("狗狗训练", top_k=3))
            print(f"[TEST] search_pdf 返回 {len(test.get('results', []))} 条")
        except Exception:
            pass


if __name__ == "__main__":
    import_books()
