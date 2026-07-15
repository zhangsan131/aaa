import argparse
import asyncio
import csv
import json
import os
import re
import uuid
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# 强制离线模式，避免 HuggingFace 网络连接问题（使用本地缓存的 embedding 模型）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class EvalItem:
    id: str
    question: str
    answer: str
    category: str = ""
    metadata: Optional[Dict[str, Any]] = None
    claims: Optional[List[str]] = None
    gold_evidence: Optional[List[Dict[str, Any]]] = None
    gold_evidence_groups: Optional[List[Dict[str, Any]]] = None


@dataclass
class EvalResult:
    id: str
    question: str
    ground_truth: str
    system_answer: str
    agent_type: str
    session_id: str
    retrieved_contexts: List[str]
    raw_response: Dict[str, Any]
    claims: Optional[List[str]] = None
    gold_evidence: Optional[List[Dict[str, Any]]] = None
    gold_evidence_groups: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None


BLOCK_RE = re.compile(
    r"(?ms)^\s*(\d+)\.?\s*(.+?)\n\s*答[:：]\s*(.+?)(?=^\s*\d+\.?\s*.+?\n\s*答[:：]\s*|\Z)"
)

# 新格式：问题后跟空行再跟答案（无"答："前缀）
BLOCK_RE_V2 = re.compile(
    r"(?ms)^\s*(\d+)\.?\s*(.+?)\n\n(.+?)(?=^\s*\d+\.?\s*.+?\n\n|\Z)"
)


def parse_jsonl_file(path: Path) -> List[EvalItem]:
    items: List[EvalItem] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL 第 {line_number} 行格式错误: {exc}") from exc
        question = str(data.get("question", "")).strip()
        answer = str(data.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError(f"JSONL 第 {line_number} 行缺少 question 或 answer")
        items.append(EvalItem(id=str(data.get("id") or f"q{line_number:03d}"), question=question, answer=answer, category=str(data.get("category", "")), metadata=data.get("metadata"), claims=data.get("claims"), gold_evidence=data.get("gold_evidence"), gold_evidence_groups=data.get("gold_evidence_groups")))
    if not items:
        raise ValueError(f"未从 JSONL 解析出题目: {path}")
    return items


def parse_txt_file(path: Path) -> List[EvalItem]:
    text = path.read_text(encoding="utf-8-sig")
    items: List[EvalItem] = []
    # 优先尝试旧格式（有"答："前缀）
    for match in BLOCK_RE.finditer(text):
        idx = match.group(1).strip()
        question = match.group(2).strip()
        answer = match.group(3).strip().replace("\n", " ")
        items.append(EvalItem(id=f"q{int(idx):03d}", question=question, answer=answer))
    # 旧格式没匹配到，尝试新格式（问题+空行+答案）
    if not items:
        for match in BLOCK_RE_V2.finditer(text):
            idx = match.group(1).strip()
            question = match.group(2).strip()
            answer = match.group(3).strip().replace("\n", " ")
            items.append(EvalItem(id=f"q{int(idx):03d}", question=question, answer=answer))
    if not items:
        raise ValueError(f"未从文件中解析出题目: {path}")
    return items


def write_jsonl(path: Path, items: List[EvalItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def resolve_embedding_model() -> str:
    return os.getenv("RAGAS_EMBEDDING_MODEL", "shibing624/text2vec-base-chinese")


def resolve_eval_provider() -> str:
    return os.getenv("RAGAS_EVAL_PROVIDER", "deepseek")


def resolve_eval_model() -> str:
    return os.getenv("RAGAS_EVAL_MODEL", "deepseek-v4-pro")


def resolve_ragas_timeout() -> int:
    return int(os.getenv("RAGAS_TIMEOUT", "300"))


def resolve_ragas_max_retries() -> int:
    return int(os.getenv("RAGAS_MAX_RETRIES", "5"))


def resolve_ragas_max_wait() -> int:
    return int(os.getenv("RAGAS_MAX_WAIT", "120"))


def resolve_ragas_max_workers() -> int:
    return int(os.getenv("RAGAS_MAX_WORKERS", "6"))


def resolve_ragas_batch_size() -> int:
    return int(os.getenv("RAGAS_BATCH_SIZE", "10"))


def resolve_ragas_llm_max_tokens() -> int:
    return int(os.getenv("RAGAS_LLM_MAX_TOKENS", "4096"))


def resolve_eval_base_url() -> str:
    provider = resolve_eval_provider().lower()
    if provider == "deepseek":
        return os.getenv("RAGAS_EVAL_BASE_URL", "https://api.deepseek.com")
    return os.getenv("RAGAS_EVAL_BASE_URL", "")


def resolve_eval_api_key() -> str:
    provider = resolve_eval_provider().lower()
    if provider == "deepseek":
        return os.getenv("RAGAS_EVAL_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
    return os.getenv("RAGAS_EVAL_API_KEY", os.getenv("OPENAI_API_KEY", ""))


async def login_and_get_token(client: httpx.AsyncClient, base_url: str, username: str, password: str) -> str:
    payload = {"username": username, "password": password}
    resp = await client.post(f"{base_url.rstrip('/')}/auth/login", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    token = data.get("token", "")
    if not token:
        raise ValueError("登录成功但未返回 token")
    return token


async def ask_backend(client: httpx.AsyncClient, base_url: str, token: str, question: str, session_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"message": question, "session_id": session_id}
    async with client.stream("POST", f"{base_url.rstrip('/')}/chat", headers=headers, json=payload, timeout=120) as resp:
        resp.raise_for_status()
        chunks: List[Dict[str, Any]] = []
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if not data:
                continue
            try:
                chunks.append(json.loads(data))
            except Exception:
                continue
    answer_parts = [c.get("content", "") for c in chunks if c.get("type") == "content"]
    agent_type = next((c.get("agent_type", "") for c in chunks if c.get("type") == "agent"), "")
    done = next((c for c in chunks if c.get("type") == "done"), {})
    structured = done.get("retrieved_contexts_structured", []) if isinstance(done, dict) else []
    if not structured:
        raise ValueError("后端未返回 retrieved_contexts_structured，无法进行标准化评测")
    retrieved_contexts = [_normalize_structured_context(ctx) or "" for ctx in structured]
    retrieved_contexts = [ctx for ctx in retrieved_contexts if ctx]
    return {
        "system_answer": "".join(answer_parts).strip(),
        "agent_type": agent_type,
        "session_id": done.get("session_id", session_id),
        "raw_response": {"events": chunks},
        "retrieved_contexts": retrieved_contexts,
        "retrieved_contexts_structured": structured,
    }


def _normalize_structured_context(ctx: Any) -> Optional[str]:
    if not isinstance(ctx, dict):
        return None
    book_title = str(ctx.get("book_title", "") or "").strip()
    section_title = str(ctx.get("section_title", "") or "").strip()
    title_path = str(ctx.get("title_path", "") or "").strip()
    page_num = ctx.get("page_num", "")
    content = str(ctx.get("content", "") or ctx.get("evidence_text", "") or ctx.get("retrieval_text", "") or "").strip()
    summary = str(ctx.get("summary", "") or "").strip()
    concept_tags = ctx.get("concept_tags", []) or []
    proper_nouns = ctx.get("proper_nouns", []) or []
    book_aliases = ctx.get("book_aliases", []) or []
    numeric_summary = str(ctx.get("numeric_summary", "") or "").strip()

    lines = [
        f"书名：{book_title}" if book_title else "书名：未知",
        f"章节：{title_path or '未分类'}",
        f"小节：{section_title or '未分类'}",
        f"页码：{page_num}",
        f"概念标签：{'、'.join(concept_tags) if concept_tags else '无'}",
        f"专有名词：{'、'.join(proper_nouns) if proper_nouns else '无'}",
        f"书名别名：{'、'.join(book_aliases) if book_aliases else '无'}",
        f"数字摘要：{numeric_summary or '无'}",
        f"摘要：{summary or '无'}",
        f"摘录：{content[:500] or '无'}",
    ]
    return "\n".join(lines)


def _extract_contexts_from_events(events: List[Dict[str, Any]], done_event: Optional[Dict[str, Any]] = None) -> List[str]:
    """已废弃的旧兼容路径，仅保留函数占位，避免误用。"""
    return []


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def extract_book_titles_from_contexts(contexts: List[str]) -> List[str]:
    titles = []
    for ctx in contexts or []:
        # 匹配 《...》 格式
        for m in re.finditer(r"《([^》]+)》", ctx):
            title = m.group(1).strip()
            if title and title not in titles:
                titles.append(title)
        # 匹配 书名：... 或 书名: ... 格式（支持多种行尾符）
        for m in re.finditer(r"(?m)^书名[：:]\s*([^\r\n]+)", ctx):
            title = m.group(1).strip()
            if title and title != "未知" and title not in titles:
                titles.append(title)
        # 匹配 book_title: ... 格式（英文冒号）
        for m in re.finditer(r"(?m)^book_title[：:]\s*([^\r\n]+)", ctx, re.IGNORECASE):
            title = m.group(1).strip()
            if title and title not in titles:
                titles.append(title)
    return titles


def extract_key_terms(text: str) -> List[str]:
    text = text or ""
    terms = [
        "阿拉伯野猫", "利比亚猫", "新月沃地", "驯化", "立尾", "相互理毛", "幼态持续", "neotenization",
        "身体屏障法", "不要重复命令", "阿尔法翻滚", "支配", "短期记忆", "重复命令", "狗的秘密", "别跟狗争老大",
        "猫的秘密", "我的第一本养猫书", "DK猫咪百科", "猫咪心事", "如何养好你的狗狗", "犬病诊断与治疗",
        "体温", "相似度", "百分比", "DNA", "基因", "寿命", "体重", "小时", "分钟", "天", "猫语", "尾巴", "竖尾",
    ]
    return [t for t in terms if t in text]


def _extract_missing_terms(question: str, retrieved_contexts: List[str], ground_truth: str) -> List[str]:
    target_terms = extract_key_terms(question + " " + ground_truth)
    ctx_text = "\n".join(retrieved_contexts or [])
    return [term for term in target_terms if term not in ctx_text]


def _extract_retrieval_signals(contexts: List[str]) -> Dict[str, Any]:
    ctx_text = "\n".join(contexts or [])
    titles = extract_book_titles_from_contexts(contexts)
    signals = {
        "titles": titles,
        "has_numeric": bool(re.search(r"\d+(?:\.\d+)?%|\d+(?:\.\d+)?|约\s*\d+|大约\s*\d+", ctx_text)),
        "has_definition": any(k in ctx_text for k in ["定义", "是指", "意味着", "表示"]),
        "has_training": any(k in ctx_text for k in ["训练", "口令", "惩罚", "奖励", "正向", "身体语言"]),
        "has_social": any(k in ctx_text for k in ["尾巴", "目光", "气味", "社交", "打招呼"]),
        "has_origin": any(k in ctx_text for k in ["驯化", "祖先", "起源", "演化"]),
    }
    return signals


def infer_topic(question: str) -> str:
    q = question or ""
    if "猫" in q:
        return "cat"
    if "狗" in q or "犬" in q:
        return "dog"
    return "other"


# ============ 语义相似度（embedding 余弦相似度） ============
_emb_model_cache = None


def _resolve_embedding_device() -> str:
    """优先使用 GPU；若不可用则回退 CPU。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return os.getenv("RAGAS_EMBEDDING_DEVICE", "cpu")


def _get_embedding_model():
    """延迟加载 embedding 模型单例，优先使用 GPU 复用已缓存的 text2vec-base-chinese"""
    global _emb_model_cache
    if _emb_model_cache is not None:
        return _emb_model_cache
    try:
        from sentence_transformers import SentenceTransformer
        local_cache = Path.home() / ".cache" / "sentence_transformers" / "text2vec-base-chinese"
        model_path = str(local_cache) if local_cache.exists() else "shibing624/text2vec-base-chinese"
        device = _resolve_embedding_device()
        _emb_model_cache = SentenceTransformer(model_path, device=device)
        print(f"[eval] embedding 模型已加载: {model_path} (device={device})")
    except Exception as e:
        print(f"[eval] embedding 模型加载失败，回退到字符匹配: {e}")
        _emb_model_cache = None
    return _emb_model_cache


def calculate_semantic_similarity(a: str, b: str) -> float:
    """用 embedding 余弦相似度衡量两段文本的语义接近程度"""
    import numpy as np
    model = _get_embedding_model()
    if model is None:
        # 模型加载失败，回退到字符匹配
        return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()
    try:
        vecs = model.encode([a or " ", b or " "], normalize_embeddings=True)
        return float(np.dot(vecs[0], vecs[1]))
    except Exception:
        return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


# 保留旧函数名作为别名，兼容其他模块调用
calculate_lexical_similarity = calculate_semantic_similarity


def generate_human_report(results: List[EvalResult], ragas_report: Dict[str, Any], output_dir: Path) -> None:
    total = len(results)
    sims = [calculate_lexical_similarity(r.ground_truth, r.system_answer) for r in results]
    avg_sim = sum(sims) / max(total, 1)
    exact = sum(1 for s in sims if s >= 0.75)
    top_wrong = sorted(zip(results, sims), key=lambda x: x[1])[:5]
    top_correct = sorted(zip(results, sims), key=lambda x: x[1], reverse=True)[:5]

    ragas_scores = ragas_report.get("scores", {}) if isinstance(ragas_report, dict) else {}
    business_rows = []
    failure_rows = []
    for item in results:
        topic = infer_topic(item.question)
        ctx_titles = extract_book_titles_from_contexts(item.retrieved_contexts)
        ctx_text = "\n".join(item.retrieved_contexts or [])
        key_terms = extract_key_terms(item.question + " " + item.ground_truth)
        missing_terms = _extract_missing_terms(item.question, item.retrieved_contexts, item.ground_truth)
        signals = _extract_retrieval_signals(item.retrieved_contexts)
        topic_hit = 1 if (
            (topic == "cat" and any("猫" in t for t in ctx_titles)) or
            (topic == "dog" and any(("狗" in t or "犬" in t) for t in ctx_titles)) or
            topic == "other"
        ) else 0
        keyword_hit_count = sum(1 for term in key_terms if term in ctx_text)
        evidence_coverage = 0.0
        if key_terms:
            evidence_coverage = min(keyword_hit_count / max(len(key_terms), 1), 1.0)
        failure_type = "ok"
        if not ctx_titles:
            failure_type = "no_title"
        elif not signals["has_definition"] and any(k in item.question for k in ["是什么", "定义", "什么意思", "指的"]):
            failure_type = "missing_definition"
        elif not signals["has_training"] and any(k in item.question for k in ["训练", "口令", "体罚", "奖励", "惩罚", "过来", "身体屏障法"]):
            failure_type = "missing_training"
        elif not signals["has_origin"] and any(k in item.question for k in ["祖先", "驯化", "起源", "演化"]):
            failure_type = "missing_origin"
        elif not signals["has_social"] and any(k in item.question for k in ["尾巴", "目光", "气味", "社交", "聊天", "竖尾"]):
            failure_type = "missing_social"
        business_rows.append({
            "id": item.id,
            "question": item.question,
            "topic": topic,
            "topic_hit": topic_hit,
            "keyword_hit_count": keyword_hit_count,
            "keyword_total": len(key_terms),
            "evidence_coverage": round(evidence_coverage, 4),
            "retrieved_titles": ctx_titles,
            "signals": signals,
            "missing_terms": missing_terms,
            "failure_type": failure_type,
        })
        if failure_type != "ok":
            failure_rows.append({
                "id": item.id,
                "question": item.question,
                "failure_type": failure_type,
                "missing_terms": missing_terms,
                "retrieved_titles": ctx_titles,
            })

    avg_topic_hit = sum(r["topic_hit"] for r in business_rows) / max(total, 1)
    avg_keyword_hit = sum((r["keyword_hit_count"] / max(r["keyword_total"], 1)) for r in business_rows) / max(total, 1)
    avg_coverage = sum(r["evidence_coverage"] for r in business_rows) / max(total, 1)

    raw_ragas_score = 0.0
    if ragas_scores:
        raw_ragas_score = sum(float(ragas_scores.get(k, 0.0) or 0.0) for k in ["faithfulness", "answer_relevancy", "context_precision", "llm_claim_coverage"]) / 4.0
    business_score = (0.4 * avg_topic_hit) + (0.3 * avg_keyword_hit) + (0.3 * avg_coverage)
    final_score = (0.7 * raw_ragas_score) + (0.3 * business_score)

    lines = []
    lines.append("# RAG 评测报告")
    lines.append("")
    lines.append("## 基本信息")
    lines.append(f"- 样本总数：{total}")
    lines.append(f"- 平均语义相似度：{avg_sim:.4f}")
    lines.append(f"- 高相似命中率(>=0.75)：{exact / max(total, 1):.4f}")
    lines.append(f"- 评测提供方：{ragas_report.get('provider', '')}")
    lines.append(f"- 评测模型：{ragas_report.get('model', '')}")
    lines.append(f"- embedding 模型：{ragas_report.get('embedding_model', '')}")
    lines.append("")
    lines.append("## RAGAS / 评分结果")
    lines.append("```json")
    lines.append(json.dumps(ragas_report, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 业务指标")
    lines.append(f"- 主题命中率：{avg_topic_hit:.4f}")
    lines.append(f"- 关键词命中率：{avg_keyword_hit:.4f}")
    lines.append(f"- 证据覆盖率：{avg_coverage:.4f}")
    lines.append(f"- 综合分：{final_score:.4f}")
    lines.append("")
    lines.append("## 失败模式统计")
    failure_counter: Dict[str, int] = {}
    for row in failure_rows:
        failure_counter[row["failure_type"]] = failure_counter.get(row["failure_type"], 0) + 1
    lines.append("```json")
    lines.append(json.dumps({"failure_counter": failure_counter, "failures": failure_rows[:20]}, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 业务指标明细")
    lines.append("```json")
    lines.append(json.dumps(business_rows, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 最容易出错的样本")
    for item, sim in top_wrong:
        lines.append(f"- [{item.id}] 相似度={sim:.4f} | 问题：{item.question}")
        lines.append(f"  - 标准答案：{item.ground_truth}")
        lines.append(f"  - 系统答案：{item.system_answer}")
    lines.append("")
    lines.append("## 最接近标准答案的样本")
    for item, sim in top_correct:
        lines.append(f"- [{item.id}] 相似度={sim:.4f} | 问题：{item.question}")
        lines.append(f"  - 标准答案：{item.ground_truth}")
        lines.append(f"  - 系统答案：{item.system_answer}")
    lines.append("")
    lines.append("## 每题结构化证据块")
    for item in results:
        raw_events = (item.raw_response or {}).get("events", []) if isinstance(item.raw_response, dict) else []
        done_event = next((e for e in raw_events if isinstance(e, dict) and e.get("type") == "done"), {})
        structured = done_event.get("retrieved_contexts_structured", []) if isinstance(done_event, dict) else []
        lines.append(f"### {item.id} | {item.question}")
        if structured:
            lines.append("```json")
            lines.append(json.dumps(structured, ensure_ascii=False, indent=2))
            lines.append("```")
        else:
            lines.append("- 无结构化证据块")
        lines.append("")
    lines.append("## 结论")
    if ragas_report.get("enabled"):
        lines.append("- RAGAS 评分流程已成功完成。")
    else:
        lines.append("- RAGAS 未完全启用，当前使用兜底评分结果。")
    lines.append("- 本报告同时输出了 RAGAS 原始报告和项目综合报告。")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _build_ragas_llm():
    """创建 RAGAS 评测用的 LLM 实例（DeepSeek / OpenAI 兼容）"""
    from openai import OpenAI

    api_key = resolve_eval_api_key()
    base_url = resolve_eval_base_url()
    model = resolve_eval_model()

    return OpenAI(api_key=api_key, base_url=base_url), model


def _call_llm(client, model: str, system_prompt: str, user_prompt: str) -> str:
    """调用 LLM 并返回文本内容"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=resolve_ragas_llm_max_tokens(),
            timeout=resolve_ragas_timeout(),
        )
        content = resp.choices[0].message.content
        return content or ""
    except Exception as e:
        print(f"[ragas] LLM 调用失败: {e}")
        return ""


def _parse_json_from_response(text: str) -> Any:
    """从 LLM 响应中提取 JSON"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        m2 = re.search(r"\{.*\}", text, re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group(0))
            except Exception:
                pass
    return None


def _compute_faithfulness(client, model: str, question: str, answer: str, contexts: List[str]) -> float:
    """faithfulness: 答案中的声明是否都有上下文支持"""
    system = "你是 RAG 评测专家。请从答案中提取所有事实声明，并判断每个声明是否能被检索上下文支持。返回 JSON。"
    ctx_text = "\n---\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    user = f"""问题：{question}

答案：
{answer}

检索上下文：
{ctx_text}

请返回 JSON：
{{
  "claims": [
    {{"statement": "声明内容", "supported": true/false}}
  ]
}}

只返回 JSON，不要其他内容。"""

    raw = _call_llm(client, model, system, user)
    data = _parse_json_from_response(raw)
    if not data or "claims" not in data:
        return 0.0
    claims = data["claims"]
    if not claims:
        return 0.0
    supported = sum(1 for c in claims if c.get("supported"))
    return supported / len(claims)


def _compute_answer_relevancy(client, model: str, question: str, answer: str) -> float:
    """answer_relevancy: 从答案反推生成的问题是否与原问题相关"""
    system = "你是 RAG 评测专家。请从答案中反推 3 个问题，并判断每个问题是否与原问题语义相关。返回 JSON。"
    user = f"""原问题：{question}

答案：
{answer}

请从答案中反推 3 个问题，并判断每个问题是否与原问题语义相关。
返回 JSON：
{{
  "generated_questions": [
    {{"question": "反推的问题", "relevant": true/false}}
  ]
}}

只返回 JSON，不要其他内容。"""

    raw = _call_llm(client, model, system, user)
    data = _parse_json_from_response(raw)
    if not data or "generated_questions" not in data:
        return 0.0
    questions = data["generated_questions"]
    if not questions:
        return 0.0
    relevant = sum(1 for q in questions if q.get("relevant"))
    return relevant / len(questions)


def _compute_context_precision(client, model: str, question: str, contexts: List[str]) -> float:
    """context_precision: 相关上下文是否排在前面"""
    system = "你是 RAG 评测专家。请判断每个检索上下文是否与问题相关。返回 JSON。"
    ctx_items = "\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts))
    user = f"""问题：{question}

检索上下文（按排序顺序）：
{ctx_items}

请判断每个上下文是否与问题相关。返回 JSON：
{{
  "relevance": [true/false, true/false, ...]
}}

数组长度必须与上下文数量一致。只返回 JSON，不要其他内容。"""

    raw = _call_llm(client, model, system, user)
    data = _parse_json_from_response(raw)
    if not data or "relevance" not in data:
        return 0.0
    relevance = data["relevance"]
    if not relevance:
        return 0.0
    relevant_count = sum(1 for r in relevance if r)
    if relevant_count == 0:
        return 0.0
    first_relevant_idx = next((i for i, r in enumerate(relevance) if r), len(relevance))
    return 1.0 / (first_relevant_idx + 1)


def _compute_context_recall(client, model: str, question: str, ground_truth: str, contexts: List[str]) -> float:
    """context_recall: 标准答案中的信息是否都能从上下文中找到"""
    system = "你是 RAG 评测专家。请从标准答案中提取关键信息点，并判断每个信息点是否能从检索上下文中找到。返回 JSON。"
    ctx_text = "\n---\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    user = f"""问题：{question}

标准答案：
{ground_truth}

检索上下文：
{ctx_text}

请从标准答案中提取关键信息点，并判断每个信息点是否能从检索上下文中找到。
返回 JSON：
{{
  "ground_truth_claims": [
    {{"claim": "信息点", "found": true/false}}
  ]
}}

只返回 JSON，不要其他内容。"""

    raw = _call_llm(client, model, system, user)
    data = _parse_json_from_response(raw)
    if not data or "ground_truth_claims" not in data:
        return 0.0
    claims = data["ground_truth_claims"]
    if not claims:
        return 0.0
    found = sum(1 for c in claims if c.get("found"))
    return found / len(claims)


def _normalized_book_title(title: str) -> str:
    title = re.sub(r"[（(].*?[）)]", "", title or "")
    return normalize_text(title).replace("zlibrary", "").replace("zlib", "")


def _anchor_keywords(anchor: str) -> List[str]:
    normalized = normalize_text(anchor)
    clauses = [part for part in re.split(r"[，。；：、]", normalized) if len(part) >= 3]
    keywords = [token for clause in clauses for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", clause)]
    # 中文连续文本会被正则视为单个 token；补充重叠 4-gram，才能容忍 OCR 插字、漏字和“常/常常”等轻微差异。
    for clause in clauses or [normalized]:
        if len(clause) >= 8:
            keywords.extend(clause[index:index + 4] for index in range(0, len(clause) - 3, 2))
    ignored = {"的是", "以及", "之间", "基本上", "一个", "这个", "我们"}
    return list(dict.fromkeys(token for token in keywords if token not in ignored))


def _gold_evidence_matches_context(evidence: Dict[str, Any], context: str) -> bool:
    normalized_context = normalize_text(context)
    title = _normalized_book_title(str(evidence.get("book_title", "")))
    context_title_match = not title or title in normalized_context
    if not context_title_match:
        context_title_match = any(token in normalized_context for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", title))
    anchor = normalize_text(str(evidence.get("anchor_text", "")))
    if not context_title_match or not anchor:
        return False
    if anchor in normalized_context:
        return True
    keywords = _anchor_keywords(str(evidence.get("anchor_text", "")))
    if not keywords:
        return False
    hit_count = sum(normalize_text(keyword) in normalized_context for keyword in keywords)
    required_hits = 1 if len(keywords) <= 2 else max(2, int(np.ceil(len(keywords) * 0.45)))
    return hit_count >= required_hits


def _gold_evidence_metrics(result: EvalResult) -> Dict[str, Optional[float]]:
    raw_groups = result.gold_evidence_groups or []
    gold_groups = [group.get("alternatives", []) for group in raw_groups if group.get("alternatives")]
    gold = result.gold_evidence or []
    # 旧格式每条证据是一个必须命中的事实组；新格式允许同一 claim 有多条等价来源。
    groups = gold_groups or [[evidence] for evidence in gold]
    if not groups:
        return {"evidence_recall_at_k": None, "mrr_at_k": None, "ndcg_at_k": None, "average_precision_at_k": None}
    contexts = result.retrieved_contexts or []
    matched_gold_ids: set[int] = set()
    context_relevance: List[int] = []
    first_rank: Optional[int] = None
    precision_sum = 0.0
    dcg = 0.0

    for context_index, context in enumerate(contexts):
        neighborhood = "\n".join(contexts[max(0, context_index - 1):context_index + 2])
        newly_matched = {
            gold_index for gold_index, alternatives in enumerate(groups)
            if gold_index not in matched_gold_ids and any(_gold_evidence_matches_context(evidence, neighborhood) for evidence in alternatives)
        }
        if newly_matched:
            matched_gold_ids.update(newly_matched)
            if first_rank is None:
                first_rank = context_index + 1
            precision_sum += len(matched_gold_ids) / (context_index + 1)
        context_relevance.append(len(newly_matched))
        dcg += len(newly_matched) / np.log2(context_index + 2)

    matched_count = len(matched_gold_ids)
    recall = matched_count / len(groups)
    mrr = 1.0 / first_rank if first_rank else 0.0
    average_precision = precision_sum / len(groups)
    ideal_dcg = sum(1.0 / np.log2(index + 1) for index in range(1, min(len(groups), len(contexts)) + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    recall, mrr, average_precision, ndcg = (min(1.0, max(0.0, value)) for value in (recall, mrr, average_precision, ndcg))
    return {"evidence_recall_at_k": round(recall, 4), "mrr_at_k": round(mrr, 4), "ndcg_at_k": round(ndcg, 4), "average_precision_at_k": round(average_precision, 4)}


def compute_retrieval_metrics(results: List[EvalResult]) -> Dict[str, Any]:
    per_item = [{"id": result.id, "question": result.question, **_gold_evidence_metrics(result)} for result in results]
    metric_names = ["evidence_recall_at_k", "mrr_at_k", "ndcg_at_k", "average_precision_at_k"]
    averages = {name: round(sum(row[name] for row in per_item if row[name] is not None) / max(sum(1 for row in per_item if row[name] is not None), 1), 4) for name in metric_names}
    annotated_count = sum(1 for result in results if result.gold_evidence or result.gold_evidence_groups)
    return {"enabled": bool(annotated_count), "annotated_samples": annotated_count, "total_samples": len(results), "scores": averages, "per_item": per_item}


def _compute_ragas_for_item(client, model: str, r: EvalResult) -> Dict[str, float]:
    """对单个样本计算四个 RAGAS 指标"""
    contexts = r.retrieved_contexts or [""]
    scores = {}
    try:
        scores["faithfulness"] = _compute_faithfulness(client, model, r.question, r.system_answer, contexts)
    except Exception as e:
        print(f"[ragas] {r.id} faithfulness 失败: {e}")
        scores["faithfulness"] = 0.0
    try:
        scores["answer_relevancy"] = _compute_answer_relevancy(client, model, r.question, r.system_answer)
    except Exception as e:
        print(f"[ragas] {r.id} answer_relevancy 失败: {e}")
        scores["answer_relevancy"] = 0.0
    try:
        scores["context_precision"] = _compute_context_precision(client, model, r.question, contexts)
    except Exception as e:
        print(f"[ragas] {r.id} context_precision 失败: {e}")
        scores["context_precision"] = 0.0
    try:
        scores["llm_claim_coverage"] = _compute_context_recall(client, model, r.question, r.ground_truth, contexts)
    except Exception as e:
        print(f"[ragas] {r.id} llm_claim_coverage 失败: {e}")
        scores["llm_claim_coverage"] = 0.0
    return scores


def compute_ragas_metrics(results: List[EvalResult], output_dir: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "enabled": False,
        "reason": "",
        "provider": resolve_eval_provider(),
        "model": resolve_eval_model(),
        "embedding_model": resolve_embedding_model(),
    }
    ragas_scores: Dict[str, Any] = {}

    try:
        client, model = _build_ragas_llm()
        print(f"[ragas] 使用 LLM: {model}, 样本数: {len(results)}")

        max_workers = resolve_ragas_max_workers()
        per_item_scores: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_compute_ragas_for_item, client, model, r): r for r in results}
            for future in as_completed(futures):
                r = futures[future]
                try:
                    scores = future.result()
                    per_item_scores.append({"id": r.id, "question": r.question, **scores})
                    print(f"[ragas] {r.id} 完成: {scores}")
                except Exception as e:
                    print(f"[ragas] {r.id} 异常: {e}")
                    per_item_scores.append({"id": r.id, "question": r.question, "faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "llm_claim_coverage": 0.0})

        if per_item_scores:
            avg_scores = {}
            for key in ["faithfulness", "answer_relevancy", "context_precision", "llm_claim_coverage"]:
                vals = [s[key] for s in per_item_scores if key in s]
                avg_scores[key] = round(sum(vals) / max(len(vals), 1), 4)
            ragas_scores = avg_scores
            report = {
                "enabled": True,
                "mode": "ragas_direct",
                "provider": resolve_eval_provider(),
                "model": resolve_eval_model(),
                "embedding_model": resolve_embedding_model(),
                "scores": ragas_scores,
                "per_item": per_item_scores,
            }
            print(f"[ragas] 平均分: {ragas_scores}")
    except Exception as exc:
        report = {"enabled": False, "reason": f"ragas 评分失败: {exc}"}
        import traceback
        traceback.print_exc()

    # fallback: 语义近似准确率
    per_item = []
    total = 0.0
    exact_hit = 0
    for r in results:
        sim = calculate_lexical_similarity(r.ground_truth, r.system_answer)
        total += sim
        if sim >= 0.75:
            exact_hit += 1
        per_item.append(
            {
                "id": r.id,
                "question": r.question,
                "ground_truth": r.ground_truth,
                "system_answer": r.system_answer,
                "similarity": round(sim, 4),
                "agent_type": r.agent_type,
            }
        )
    avg = round(total / max(len(results), 1), 4)
    hit_rate = round(exact_hit / max(len(results), 1), 4)
    fallback_report = {
        "enabled": True,
        "mode": "fallback_similarity",
        "provider": resolve_eval_provider(),
        "model": resolve_eval_model(),
        "embedding_model": resolve_embedding_model(),
        "average_similarity": avg,
        "exact_hit_rate": hit_rate,
        "threshold": 0.75,
        "per_item": per_item,
    }
    final_report = {**report, "fallback": fallback_report}
    (output_dir / "ragas_scores.json").write_text(json.dumps({"ragas": report, "fallback": fallback_report}, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_report


async def run_eval(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"评测输入文件不存在: {input_path}")
    items = parse_jsonl_file(input_path) if input_path.suffix.lower() == ".jsonl" else parse_txt_file(input_path)
    output_jsonl = Path(args.output_jsonl)
    if input_path.resolve() != output_jsonl.resolve():
        write_jsonl(output_jsonl, items)

    base_url = args.base_url
    token = args.token or os.getenv("EVAL_BEARER_TOKEN", "")
    if not token:
        login_username = args.login_username or os.getenv("EVAL_LOGIN_USERNAME", "")
        login_password = args.login_password or os.getenv("EVAL_LOGIN_PASSWORD", "")
        if not login_username or not login_password:
            raise ValueError("请提供 --token，或者提供 --login-username / --login-password，或设置 EVAL_BEARER_TOKEN")
        async with httpx.AsyncClient() as client:
            token = await login_and_get_token(client, base_url, login_username, login_password)
            print(f"已自动登录，用户名={login_username}")

    results: List[EvalResult] = []
    async with httpx.AsyncClient() as client:
        for item in items:
            session_id = f"eval_{item.id}_{uuid.uuid4().hex[:8]}"
            backend = await ask_backend(client, base_url, token, item.question, session_id)
            results.append(
                EvalResult(
                    id=item.id,
                    question=item.question,
                    ground_truth=item.answer,
                    system_answer=backend["system_answer"],
                    agent_type=backend["agent_type"],
                    session_id=backend["session_id"],
                    retrieved_contexts=backend.get("retrieved_contexts", []),
                    raw_response=backend["raw_response"],
                    claims=item.claims,
                    gold_evidence=item.gold_evidence,
                    gold_evidence_groups=item.gold_evidence_groups,
                )
            )
            print(f"[{item.id}] {item.question[:40]}... -> {backend['agent_type']}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "agent_type", "ground_truth", "system_answer"])
        for result in results:
            writer.writerow([result.id, result.question, result.agent_type, result.ground_truth, result.system_answer])

    ragas_report = compute_ragas_metrics(results, out_dir)
    retrieval_report = compute_retrieval_metrics(results)
    ragas_report["retrieval_metrics"] = retrieval_report
    (out_dir / "ragas_report.json").write_text(json.dumps(ragas_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "retrieval_metrics.json").write_text(json.dumps(retrieval_report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"total": len(results), "ragas_report": ragas_report, "retrieval_metrics": retrieval_report}, ensure_ascii=False, indent=2))
    generate_human_report(results, ragas_report, out_dir)

    print(f"已完成 {len(results)} 条样本评测")
    print(f"JSONL: {out_dir / 'results.jsonl'}")
    print(f"CSV:   {out_dir / 'results.csv'}")
    print(f"MD:    {out_dir / 'report.md'}")
    print(f"Ragas: {out_dir / 'ragas_report.json'}")


def main() -> None:
    # 所有评测文件统一放在项目根目录（动物百科多agent项目/）下
    project_root = Path(__file__).resolve().parent.parent.parent
    default_eval_data = str(project_root / "eval_data" / "qa.jsonl")
    default_output_dir = str(project_root / "eval_results" / "run_001")

    parser = argparse.ArgumentParser(description="猫狗百科 RAG 评测脚本")
    parser.add_argument("--input", default=default_eval_data, help="评测集路径，支持 JSONL 或题目/答案 TXT")
    parser.add_argument("--output-jsonl", default=default_eval_data, help="TXT 输入转换后的 JSONL 输出路径；JSONL 输入时不覆盖源文件")
    parser.add_argument("--output-dir", default=default_output_dir, help="评测结果输出目录")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="后端地址")
    parser.add_argument("--token", default="", help="登录后 Bearer token")
    parser.add_argument("--login-username", default="user", help="自动登录用户名")
    parser.add_argument("--login-password", default="user123", help="自动登录密码")
    args = parser.parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
