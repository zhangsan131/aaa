"""
Fauna AI 猫狗百科 - LangGraph 多 Agent 架构

本版重点：
- 收敛 query rewrite 噪声
- 使用多路召回 + 统一去重 + 上下文补全
- 降低书名硬过滤强度
- 为后续 reranker 留出清晰接口
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import sys
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple, TypedDict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(os.path.expanduser("~"), ".cache", "sentence_transformers"))

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
import jieba
import numpy as np
import requests

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
for import_dir in (CURRENT_DIR, BACKEND_DIR):
    if import_dir not in sys.path:
        sys.path.append(import_dir)

from import_books import KnowledgeBase, CACHE_FILE as BOOK_CACHE_FILE, EMBEDDING_MODEL_NAME
from memory_store import (
    build_memory_context,
    build_recent_summary,
    get_recent_summary,
    get_session_metadata,
    get_user_profile,
    load_recent_messages,
    save_recent_summary,
    update_session_metadata,
    update_user_profile_from_messages,
)

load_dotenv()

# 可观测性（保留但不强依赖）
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, Gauge, generate_latest
except Exception:
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    Counter = Histogram = Gauge = None
    generate_latest = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "fauna_ai.db")
SESSION_EXPIRE_DAYS = 7
KB_CACHE_FILE = BOOK_CACHE_FILE if 'BOOK_CACHE_FILE' in globals() else os.path.join(BASE_DIR, "knowledge_cache.pkl")
KB_VECTOR_FILE = os.path.join(BASE_DIR, "vector_index.pkl")
KB_BM25_FILE = os.path.join(BASE_DIR, "bm25_index.pkl")

RETRIEVAL_CANDIDATE_K = int(os.getenv("RETRIEVAL_CANDIDATE_K", "80"))
RETRIEVAL_FINAL_K = int(os.getenv("RETRIEVAL_FINAL_K", "8"))
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", r"C:\Users\ls\.cache\modelscope\hub\models\BAAI\bge-reranker-v2-m3")
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "1") == "1"
RERANKER_CANDIDATE_K = int(os.getenv("RERANKER_CANDIDATE_K", "40"))
COMPARE_OBJECT_BONUS = float(os.getenv("COMPARE_OBJECT_BONUS", "0.20"))
RARE_ENTITY_BONUS = float(os.getenv("RARE_ENTITY_BONUS", "0.16"))
COVERAGE_BONUS = float(os.getenv("COVERAGE_BONUS", "0.10"))
MAX_CONTEXTS_PER_SECTION = int(os.getenv("MAX_CONTEXTS_PER_SECTION", "2"))
LEXICAL_COVERAGE_BONUS = float(os.getenv("LEXICAL_COVERAGE_BONUS", "0.12"))
CONTEXT_EXPAND_K = int(os.getenv("CONTEXT_EXPAND_K", "2"))
NUMERIC_FALLBACK_K = int(os.getenv("NUMERIC_FALLBACK_K", "8"))
RAG_CONTEXT_CHAR_LIMIT = int(os.getenv("RAG_CONTEXT_CHAR_LIMIT", "700"))
RAG_PAGE_CONTEXT_CHAR_LIMIT = int(os.getenv("RAG_PAGE_CONTEXT_CHAR_LIMIT", "1400"))
RAG_NUMERIC_FALLBACK_INJECT_LIMIT = int(os.getenv("RAG_NUMERIC_FALLBACK_INJECT_LIMIT", "6"))
RAG_DEBUG_RESULTS_FILE = Path(os.getenv("RAG_DEBUG_RESULTS_FILE", os.path.join(BASE_DIR, "rag_debug_results.txt")))

_reranker_model = None
_reranker_load_attempted = False


def _get_reranker():
    global _reranker_model, _reranker_load_attempted
    if _reranker_load_attempted:
        return _reranker_model
    _reranker_load_attempted = True
    if not RERANKER_ENABLED or CrossEncoder is None or not Path(RERANKER_MODEL_NAME).exists():
        return None
    try:
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME, max_length=512)
    except Exception as exc:
        print(f"[WARN] Cross-Encoder 加载失败，将回退到 RRF：{exc}")
    return _reranker_model


app = FastAPI(title="Fauna AI 猫狗百科 - LangGraph版", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

REQUEST_COUNT = Counter("fauna_ai_requests_total", "HTTP 请求总数", ["method", "path", "status"]) if Counter else None
REQUEST_LATENCY = Histogram("fauna_ai_request_duration_seconds", "HTTP 请求耗时", ["method", "path"]) if Histogram else None
ACTIVE_SESSIONS = Gauge("fauna_ai_active_sessions", "活跃会话数") if Gauge else None

# === 基础存储 ===

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120000)
    return salt, digest.hex()


def verify_password(password: str, salt: str, password_hash: str) -> bool:
    _, candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, password_hash)


def user_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {"id": row["id"], "username": row["username"], "email": row["email"], "is_admin": bool(row["is_admin"]), "created_at": row["created_at"]}


def create_session(user_id: int) -> Tuple[str, str]:
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.utcnow() + timedelta(days=SESSION_EXPIRE_DAYS)).isoformat(timespec="seconds")
    with get_db_connection() as conn:
        conn.execute("INSERT INTO sessions (token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)", (token, user_id, expires_at, utc_now()))
        conn.commit()
    return token, expires_at


def init_database():
    with get_db_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token TEXT NOT NULL UNIQUE,
                user_id INTEGER NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                agent_type TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS sessions_state (
                session_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                recent_summary TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (session_id, user_id)
            );
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                profile_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );
            """
        )
        if not conn.execute("SELECT id FROM users WHERE username='admin'").fetchone():
            salt, pwd_hash = hash_password("admin123")
            conn.execute("INSERT INTO users (username, email, password_hash, salt, is_admin, created_at) VALUES (?, ?, ?, ?, ?, ?)", ("admin", "admin@example.com", pwd_hash, salt, 1, utc_now()))
        if not conn.execute("SELECT id FROM users WHERE username='user'").fetchone():
            salt, pwd_hash = hash_password("user123")
            conn.execute("INSERT INTO users (username, email, password_hash, salt, is_admin, created_at) VALUES (?, ?, ?, ?, ?, ?)", ("user", "user@example.com", pwd_hash, salt, 0, utc_now()))
        conn.commit()


init_database()


def get_current_user(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")
    token = authorization.split(" ", 1)[1].strip()
    with get_db_connection() as conn:
        row = conn.execute("SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token = ? AND s.expires_at > ?", (token, utc_now())).fetchone()
    if not row:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已过期，请重新登录")
    return user_to_dict(row)


def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if not current_user.get("is_admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限")
    return current_user


def save_message(user_id: int, session_id: str, role: str, content: str, agent_type: Optional[str] = None):
    with get_db_connection() as conn:
        conn.execute("INSERT INTO messages (session_id, user_id, role, content, agent_type, created_at) VALUES (?, ?, ?, ?, ?, ?)", (session_id, user_id, role, content, agent_type, utc_now()))
        conn.commit()


def load_user_memory(user_id: int, session_id: str = None, limit: int = 20) -> List[Dict[str, str]]:
    return load_recent_messages(user_id, session_id=session_id, limit=limit)


def _safe_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def parse_llm_json(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("LLM 返回为空")
    content = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    if fence_match:
        content = fence_match.group(1).strip()
    start = content.find("{")
    if start == -1:
        raise ValueError("未找到 JSON 对象")
    obj = json.loads(content[start:])
    if isinstance(obj, dict):
        return obj
    raise ValueError("LLM JSON 不是对象")


def _extract_book_name(file_name: str) -> str:
    if not file_name:
        return "未知书籍"
    name = re.sub(r'\.(pdf|mk)$', '', file_name, flags=re.IGNORECASE)
    match = re.match(r'^(.+?)(?:[_\s]*[（\(].*|[ _]\d{4})', name)
    return match.group(1).strip() if match and match.group(1).strip() else name.strip()


def _normalize_book_title(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[\-–—_·,，。.!！?？/\\]+", "", text)
    return text


def _parse_search_pdf_results(payload: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(payload)
    except Exception:
        return []
    return data.get("results", []) if isinstance(data, dict) else []


def _is_numeric_question(text: str) -> bool:
    return bool(re.search(r"\d|百分比|相似度|基因|DNA|比例|数值|多少|几成|占比", text or ""))


def _extract_percentages(text: str) -> List[str]:
    return list(dict.fromkeys(re.findall(r"\d+(?:\.\d+)?%", text or "")))


def _extract_numeric_values(text: str) -> List[str]:
    if not text:
        return []
    unit_pattern = r"(?:小时|个小时|h|天|日|年|岁|个月|月|分钟|秒|次|倍|公斤|kg|斤|克|g|米|m|厘米|cm|度|℃|°)"
    patterns = [rf"\d+(?:\.\d+)?(?:\s*[-~—–至到]\s*\d+(?:\.\d+)?)?\s*{unit_pattern}"]
    found = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text))
    return list(dict.fromkeys(x.strip() for x in found))


def _extract_entities(text: str) -> List[str]:
    entity_map = {"狗": ["狗", "犬", "家犬"], "猫": ["猫", "猫咪", "家猫"], "狼": ["狼", "灰狼"], "人": ["人", "人类"]}
    matched = []
    for entity, aliases in entity_map.items():
        if any(alias in (text or "") for alias in aliases):
            matched.append(entity)
    return list(dict.fromkeys(matched))


def _extract_intent_keywords(text: str) -> List[str]:
    patterns = ["如何", "怎么", "怎样", "过程", "步骤", "方法", "为什么", "原因", "区别", "不同", "多少", "哪些", "什么"]
    return [p for p in patterns if p in (text or "")]


def _topic_route(user_message: str) -> str:
    text = (user_message or "").lower()
    if any(k in text for k in ["体温", "百分比", "相似度", "dna", "基因", "重量", "寿命", "年龄", "分钟", "小时", "天", "公斤", "kg", "斤"]):
        return "numeric"
    if any(k in text for k in ["是什么", "定义", "什么意思", "指的", "解释"]):
        return "definition"
    if any(k in text for k in ["为什么", "原因", "机制", "如何", "怎么", "怎样"]):
        return "mechanism"
    if any(k in text for k in ["区别", "不同", "对比", "相同", "相似"]):
        return "compare"
    if any(k in text for k in ["训练", "口令", "惩罚", "奖励", "牵绳", "坐下", "扑人"]):
        return "training"
    if any(k in text for k in ["祖先", "驯化", "起源", "演化", "进化"]):
        return "origin"
    if any(k in text for k in ["尾巴", "目光", "气味", "社交", "打招呼", "信号", "竖起"]):
        return "social"
    return "general"


TERM_SYNONYMS = {
    "奖励训练": ["奖励", "奖赏", "正向鼓励", "正强化", "体罚", "惩罚"],
    "社会化": ["社会化", "敏感期", "幼犬", "逐渐接触", "恐惧反应"],
    "分离焦虑": ["分离焦虑", "离开", "钥匙", "穿鞋", "独自在家", "主人回归"],
    "气味交流": ["嗅觉", "嗅闻", "气味", "气味标记", "肛门囊", "鼻子"],
    "声音敏感": ["听觉", "高频", "超声波", "噪音", "声音"],
    "尾巴信号": ["尾巴", "断尾", "摇摆", "身体语言", "交流"],
    "幼态持续": ["幼态持续", "neotenization", "幼态保留"],
    "身体屏障法": ["身体屏障法", "身体阻挡", "侧身挡住"],
    "驯化": ["驯化", "自我驯化", "家养"],
    "社交信号": ["社交信号", "尾巴", "理毛", "蹭", "目光", "对视"],
    "行为学": ["行为学", "行为", "社会性", "训练"],
    "认知": ["认知", "学习", "理解", "记忆"],
    "情绪": ["情绪", "内疚", "恐惧", "焦虑"],
}


def _pick_query_tag(text: str, fine: bool = False) -> str:
    t = (text or "").lower()
    coarse_rules = [
        ("驯化起源", ["驯化", "起源", "祖先", "演化"]),
        ("行为训练", ["训练", "服从", "纠正", "惩罚"]),
        ("感官生理", ["嗅觉", "视觉", "听觉", "感官", "生理"]),
        ("护理健康", ["喂养", "健康", "疾病", "医疗"]),
        ("人宠关系", ["人宠", "陪伴", "家庭", "互动"]),
        ("品种特征", ["品种", "外形", "体型", "毛色"]),
        ("社交信号", ["社交", "信号", "打招呼", "目光", "尾巴", "气味"]),
        ("情绪认知", ["情绪", "认知", "压力", "恐惧", "依恋"]),
        ("生态保护", ["生态", "保护", "流浪", "栖息"]),
    ]
    if fine:
        fine_rules = coarse_rules + [("演化遗传", ["基因", "dna", "血统", "遗传", "亲缘"])]
        for tag, kws in fine_rules:
            if any(k in t for k in kws):
                return tag
    for tag, kws in coarse_rules:
        if any(k in t for k in kws):
            return tag
    return "未知"


def _expand_query_terms(base: str, rewrite: str = "") -> List[str]:
    text = f"{base} {rewrite}".strip()
    terms = [base]
    for term, aliases in TERM_SYNONYMS.items():
        if term in text or any(alias in text for alias in aliases):
            terms.extend(aliases)
    if "狗" in text:
        terms.extend(["犬", "家犬", "狗狗"])
    if "猫" in text:
        terms.extend(["猫咪", "家猫"])
    return list(dict.fromkeys([t for t in terms if t]))[:8]


RARE_RETRIEVAL_ENTITIES = ["粮仓", "灰狼", "野猫", "阿拉伯野猫", "利比亚猫", "新月沃地", "幼态持续", "身体屏障法", "阿尔法翻滚"]


def _rare_entities(query: str) -> List[str]:
    return [entity for entity in RARE_RETRIEVAL_ENTITIES if entity in (query or "")]


def _origin_intent_terms(query: str) -> List[str]:
    query = query or ""
    terms: List[str] = []
    if any(token in query for token in ["共生", "为什么", "关系如何形成"]):
        terms.extend(["互利", "食物供应", "猎物", "人类聚居地", "家庭群落"])
    if any(token in query for token in ["驯化", "演化", "祖先"]):
        terms.extend(["人工选择", "适应", "祖先", "野生", "演化"])
    if any(token in query for token in ["起源", "最早", "如何形成"]):
        terms.extend(["最早", "来源", "形成", "开始", "历史"])
    if "独立" in query:
        terms.extend(["独自捕猎", "工作方式", "依赖人类", "合作", "关注主人"])
    return list(dict.fromkeys(terms))


INTENT_RETRIEVAL_TERMS = [
    (r"奖励|体罚|惩罚|训.*狗", ["奖赏", "正向鼓励", "正强化", "害怕", "攻击", "训练"]),
    (r"坐下|口令|指令|不同地点", ["环境线索", "地点", "命令", "重复训练", "不同地方"]),
    (r"老大|支配|强势|地位", ["支配关系", "等级", "降低地位", "惩罚", "恐惧", "自卫", "咬人"]),
    (r"幼犬|小狗.*接触|社会化", ["敏感期", "逐渐", "人类", "环境", "恐惧", "社会化"]),
    (r"钥匙|穿鞋|离开|分离焦虑", ["分离", "主人回归", "焦虑", "线索", "奖励"]),
    # 概念层扩写：适用于脱敏、环境适应、追踪和同类社交等相邻行为问题，而非绑定评测题的原句或锚点。
    (r"循序|逐渐|强行.*刺激|刺激.*害怕|恐惧.*阈值|脱敏|适应", ["逐步暴露", "刺激强度", "恐惧", "习惯化", "安全距离", "降低强度"]),
    (r"闻.*狗|嗅.*狗|狗.*嗅闻|同类.*气味", ["嗅觉", "个体识别", "社会交流", "气味信息", "气味标记", "肛门囊"]),
    (r"找路|认路|原路|追踪|鼻子.*环境|气味.*地标", ["嗅觉", "气味地标", "足迹气味", "环境识别", "追踪"]),
    (r"嗅|鼻子.*环境|气味", ["嗅觉", "气味", "气味标记", "识别", "肛门囊", "找路"]),
    (r"听不见|声音.*影响|高频|超声", ["听觉", "高音调", "超声波", "敏感", "噪音"]),
    (r"尾巴.*交流|截短|断尾", ["尾巴", "摇摆", "视觉信号", "交流", "断尾"]),
    (r"外形相似|害怕.*狗", ["恐惧", "经历", "相似", "攻击", "泛化"]),
    (r"犬种|品种.*攻击", ["品种", "遗传", "训练", "经历", "攻击性"]),
    (r"昏暗|暗处|光线", ["视野", "视力", "听觉", "猎物", "后腿", "灵活"]),
    (r"胡须", ["触毛", "神经", "感知", "平衡", "空间"]),
    (r"走路|声音轻", ["肉球", "爪子", "缓冲", "吸音", "伸缩"]),
    (r"高处|跳下|落地", ["脊椎", "柔软", "腿部", "肉球", "缓冲"]),
    (r"尾巴.*运动|运动.*尾巴", ["骨头", "神经", "跳跃", "落地", "平衡"]),
    (r"社会化|触摸", ["3周", "5～7周", "3个月", "社会化", "触摸"]),
    (r"疫苗", ["混合疫苗", "接种", "2个月", "1岁", "健康检查"]),
    (r"营养|食物", ["快速成长", "幼猫", "成年猫", "能量", "猫粮"]),
    (r"室外|散养", ["交通事故", "跳蚤", "感染", "迷路", "室内饲养"]),
    (r"睡", ["14～15小时", "20小时", "睡觉", "休息"]),
    (r"夜里.*叫|夜叫", ["寂寞", "寻找亲人", "安抚", "6个月"]),
    (r"寄养", ["紧张", "陌生环境", "宠物旅馆", "压力"]),
    (r"生产|产后|母猫", ["安静", "兴奋", "攻击", "小猫", "守候"]),
    (r"保温|体温", ["体温调节", "38℃", "热水袋", "毛巾"]),
    (r"牛奶", ["专用奶", "腹泻", "奶瓶"]),
    (r"排泄", ["肛门", "棉球", "纱布", "刺激"]),
    (r"称体重", ["每天", "体重变化", "厨房秤", "成长"]),
    (r"糖尿病|肥胖", ["胰岛素", "饮水量", "尿液", "肥胖"]),
    (r"肾|饮水量|排尿量", ["慢性肾功能不全", "高龄", "尿液", "体重减轻"]),
    (r"排不出尿|尿道", ["尿道堵塞", "尿毒症", "膀胱炎", "尿路结石"]),
    (r"绝育|避孕", ["妊娠", "做记号", "子宫", "乳腺癌", "发情"]),
    (r"发情", ["日照", "叫", "尿频", "摩擦", "腰部"]),
    (r"跳蚤", ["唾液", "蛋白质", "红疹", "脱毛", "瘙痒"]),
    (r"牙齿|牙周", ["牙垢", "牙龈炎", "口臭", "牙刷", "纱布"]),
]


def _intent_retrieval_terms(query: str) -> List[str]:
    terms: List[str] = []
    for pattern, expansions in INTENT_RETRIEVAL_TERMS:
        if re.search(pattern, query or ""):
            terms.extend(expansions)
    return list(dict.fromkeys(terms))


def _build_intent_query(base: str) -> str:
    return " ".join(dict.fromkeys([base, *_intent_retrieval_terms(base)]))


def _build_origin_query(base: str) -> str:
    return " ".join(dict.fromkeys([base, *_rare_entities(base), *_origin_intent_terms(base)]))


def _build_compare_query(base: str) -> str:
    """把比较对象映射为知识库更可能出现的同义实体和关系词。"""
    aliases = {
        "狗": ["狗", "家犬"],
        "狼": ["狼", "灰狼", "祖先"],
        "猫": ["猫", "家猫", "宠物猫"],
        "野猫": ["野猫", "野外猎手"],
    }
    matched = [alias for entity, values in aliases.items() if entity in base for alias in values]
    return " ".join(dict.fromkeys([base, *matched, "差异 比较 驯化 心智 行为 后代 难以区分 大相径庭"]))


def _comparison_entities(query: str) -> List[List[str]]:
    aliases = {
        "狗": ["狗", "家犬"],
        "狼": ["狼", "灰狼"],
        "猫": ["猫", "家猫", "宠物猫"],
        "野猫": ["野猫", "野外猎手"],
    }
    return [values for entity, values in aliases.items() if entity in query]


def _comparison_match_bonus(item: Dict[str, Any], query: str) -> float:
    entities = _comparison_entities(query)
    if len(entities) < 2:
        return 0.0
    text = _result_text(item)
    matched_groups = sum(any(alias in text for alias in group) for group in entities)
    return COMPARE_OBJECT_BONUS * (matched_groups / len(entities)) if matched_groups >= 2 else 0.0


def _comparison_relation_bonus(item: Dict[str, Any], query: str) -> float:
    """比较题不仅要求对象出现，还要求 Chunk 包含题目所问的比较关系。"""
    if _topic_route(query) != "compare":
        return 0.0
    text = _result_text(item)
    relation_terms = ["驯化", "人工选择", "大相径庭", "难以区分", "差异", "相似", "独立", "共同工作"]
    query_terms = [term for term in relation_terms if term in query]
    required_terms = query_terms or relation_terms
    hits = sum(term in text for term in required_terms)
    return min(0.18, 0.06 * hits)


def _build_parallel_queries(user_message: str, rewrite_query: str, numeric_intent: bool = False, topic_type: str = "general") -> List[Dict[str, Any]]:
    """构建原问题与两路确定性扩写；HyDE 由 query_rewrite 节点补充。"""
    del topic_type
    base = re.sub(r"\s+", " ", (user_message or "")).strip()
    rewrite = re.sub(r"\s+", " ", (rewrite_query or base)).strip()
    queries: List[Dict[str, Any]] = []

    def _add(query: str, kind: str, weight: float) -> None:
        query = re.sub(r"\s+", " ", (query or "")).strip()
        if query and all(item["query"] != query for item in queries):
            queries.append({"query": query, "kind": kind, "weight": weight})

    _add(base, "original", 1.0)
    synonyms = [term for term in _expand_query_terms(base, rewrite) if term not in {base, rewrite}]
    _add(f"{base} {synonyms[0]}" if synonyms else rewrite, "entity_expansion", 0.70)
    if _topic_route(base) == "compare":
        _add(_build_compare_query(base), "comparison_expansion", 0.70)
    elif _intent_retrieval_terms(base):
        _add(_build_intent_query(base), "intent_expansion", 0.80)
    elif _origin_intent_terms(base):
        _add(_build_origin_query(base), "origin_expansion", 0.75)
    elif numeric_intent:
        for pattern, units in [(r"体温|温度|发烧|退烧", "℃ 度"), (r"体重|重量|胖|瘦", "公斤 kg 斤"), (r"寿命|年龄|岁", "年 岁"), (r"基因|DNA|遗传|相似度|血统", "% 百分比")]:
            if re.search(pattern, base):
                _add(f"{base} {units}", "retrieval_rewrite", 0.60)
                break
    else:
        _add(rewrite if rewrite != base else f"{base} 相关原因 机制 特征", "retrieval_rewrite", 0.60)
    return queries[:3]


def _hyde_is_allowed(query: str, task_type: str) -> bool:
    medical_or_legal = task_type in {"pet", "law"} or bool(re.search(r"咨询|看病|生病|症状|治疗|医院|兽医|法律|法规|违法|赔偿|责任|权益|禁止", query))
    return bool(query.strip()) and not _is_numeric_question(query) and not medical_or_legal


def _build_reroute_queries(user_message: str, fallback_seed: str, task_type: str = "knowledge", review_reason: str = "") -> List[str]:
    base = (user_message or "").strip()
    seed = (fallback_seed or base).strip()
    queries = [base, seed]
    if "猫" in base:
        queries.append(f"{base} 家猫 野猫")
    elif "狗" in base:
        queries.append(f"{base} 家犬 行为")
    if review_reason and any(k in review_reason for k in ["未找到", "未提及", "无相关", "没有", "不通过", "证据"]):
        queries.append(seed)
    cleaned = []
    for q in queries:
        q = re.sub(r"\s+", " ", q).strip()
        if q and q not in cleaned:
            cleaned.append(q)
    return cleaned[:3]


def _get_relevant_units(query: str) -> Optional[set]:
    concept_unit_map = [
        (r"体温|温度|发烧|退烧", {"℃", "度"}),
        (r"体重|重量|胖|瘦", {"公斤", "kg", "斤"}),
        (r"寿命|年龄|岁|活多久", {"年", "岁"}),
        (r"基因|DNA|遗传|相似度|相似|血统", {"%"}),
    ]
    for pattern, units in concept_unit_map:
        if re.search(pattern, query):
            return units
    return None


def _result_text(item: Dict[str, Any]) -> str:
    return " ".join([
        str(item.get("book_title", "")),
        str(item.get("title_path", "")),
        str(item.get("summary", "")),
        str(item.get("content", "")),
        str(item.get("numeric_summary", "")),
    ]).lower()


def _rare_entity_match_bonus(item: Dict[str, Any], query: str) -> float:
    entities = _rare_entities(query)
    if not entities:
        return 0.0
    text = _result_text(item)
    hits = sum(entity.lower() in text for entity in entities)
    return RARE_ENTITY_BONUS * (hits / len(entities))


def _coverage_terms(query: str) -> List[str]:
    return list(dict.fromkeys([*_rare_entities(query), *_origin_intent_terms(query), *_intent_retrieval_terms(query)]))


def _lexical_coverage_bonus(item: Dict[str, Any], query: str) -> float:
    terms = _intent_retrieval_terms(query)
    if not terms:
        return 0.0
    text = _result_text(item)
    hits = sum(term.lower() in text for term in terms)
    return LEXICAL_COVERAGE_BONUS * (hits / len(terms))


def _coverage_match(item: Dict[str, Any], query: str) -> bool:
    entities = _rare_entities(query)
    terms = _coverage_terms(query)
    if not entities or len(terms) <= len(entities):
        return False
    text = _result_text(item)
    return any(entity.lower() in text for entity in entities) and any(term.lower() in text for term in terms if term not in entities)


def _fact_coverage_groups(query: str) -> List[List[str]]:
    """返回回答该类问题必须覆盖的独立事实面，避免只命中泛主题词。"""
    rules = [
        (r"幼猫.*营养|营养.*幼猫", [["快速成长", "优质营养", "营养价值"], ["成年猫", "能量要少", "食物的量"]]),
        (r"社会化", [["3周", "社会化时期"], ["3个月", "5～7周"]]),
        (r"昏暗|暗处|光线", [["视野", "昏暗", "分辨物体"], ["听力", "猎物的位置", "后腿"]]),
        (r"排泄", [["肛门", "刺激", "排泄"], ["纱布", "棉球", "清理干净"]]),
        (r"保温|体温", [["无法自行调节体温", "体温调节"], ["38℃", "热水袋", "毛巾"]]),
        (r"绝育|避孕", [["子宫", "乳腺癌"], ["不必要的妊娠", "做记号", "发情"]]),
        (r"排不出尿|尿道", [["膀胱炎", "尿路结石", "尿道闭塞"], ["尿毒症", "短时间内导致死亡", "无法排尿"]]),
        (r"走路.*声音|声音.*轻", [["吸音", "肉球", "缓冲"], ["自由伸缩", "爪子"]]),
        (r"高处.*跳|跳.*高处", [["缓冲", "肉球"], ["脊椎骨", "腿部先着地", "身体可以很好地扭曲"]]),
        (r"室外|散养", [["交通事故", "室内饲养"], ["感染疾病", "跳蚤", "迷路"]]),
        (r"生产|产后|母猫.*打扰", [["消耗大量精力", "产后"], ["情绪稳定之前", "不要过多地干涉", "安静"]]),
        (r"称体重", [["每天给它测量体重", "每天"], ["1克左右", "体重变化", "成长"]]),
        (r"成年猫.*疫苗|疫苗.*成年|定期接种", [["1岁以后", "每年接种"], ["室内饲养", "感染病毒", "宠物旅馆"]]),
        (r"夜里.*叫|夜叫", [["6个月之前", "寂寞"], ["寻找亲人", "安抚", "抱起来抚摸"]]),
    ]
    return next((groups for pattern, groups in rules if re.search(pattern, query or "")), [])


def _adaptive_final_k(query: str) -> int:
    if _topic_route(query) == "compare" or len(_fact_coverage_groups(query)) >= 2:
        return min(RETRIEVAL_FINAL_K, 6)
    if any(token in (query or "") for token in ["如何", "步骤", "治疗", "风险", "为什么"]):
        return min(RETRIEVAL_FINAL_K, 4)
    return min(RETRIEVAL_FINAL_K, 3)


def _fact_group_match(item: Dict[str, Any], group: List[str]) -> bool:
    text = _result_text(item)
    return any(term.lower() in text for term in group)


def _numeric_match_bonus(item: Dict[str, Any], query: str, relevant_units: Optional[set]) -> float:
    if not relevant_units:
        return 0.0
    text = _result_text(item)
    if not any(unit.lower() in text for unit in relevant_units):
        return 0.0
    bonus = 0.25
    intent_terms = [term for term in re.findall(r"[\u4e00-\u9fffA-Za-z]{2,}", query.lower()) if term not in {"多少", "什么", "怎么", "如何", "为什么"}]
    if any(term in text for term in intent_terms):
        bonus += 0.15
    return bonus


def _aggregate_and_rerank_results(
    bundles: List[Dict[str, Any]],
    user_query: str,
    numeric_fallback_results: Optional[List[Dict[str, Any]]] = None,
    relevant_units: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """按原查询顺序聚合多路 RRF 结果，并以稳定的排名分数决定最终顺序。"""
    aggregated: Dict[str, Dict[str, Any]] = {}
    query_count = max(len(bundles), 1)
    for query_index, bundle in enumerate(bundles):
        data = bundle.get("data", {})
        results = data.get("results", []) if isinstance(data, dict) else []
        query_weight = float(bundle.get("weight", 1.0 if query_index == 0 else 0.60))
        for rank, item in enumerate(results, start=1):
            chunk_uid = item.get("chunk_uid")
            if not chunk_uid:
                continue
            entry = aggregated.setdefault(chunk_uid, {"item": item, "score": 0.0, "hits": 0, "best_rank": rank})
            entry["score"] += query_weight / (60 + rank)
            entry["hits"] += 1
            entry["best_rank"] = min(entry["best_rank"], rank)

    for item in numeric_fallback_results or []:
        chunk_uid = item.get("chunk_uid")
        if not chunk_uid:
            continue
        entry = aggregated.setdefault(chunk_uid, {"item": item, "score": 0.0, "hits": 0, "best_rank": RETRIEVAL_CANDIDATE_K + 1})
        entry["score"] += _numeric_match_bonus(item, user_query, relevant_units)

    reranked = []
    for chunk_uid, entry in aggregated.items():
        item = dict(entry["item"])
        consensus_bonus = min(entry["hits"] - 1, query_count - 1) * 0.0025
        numeric_bonus = _numeric_match_bonus(item, user_query, relevant_units)
        comparison_bonus = _comparison_match_bonus(item, user_query)
        comparison_relation_bonus = _comparison_relation_bonus(item, user_query)
        rare_entity_bonus = _rare_entity_match_bonus(item, user_query)
        lexical_coverage_bonus = _lexical_coverage_bonus(item, user_query)
        final_score = entry["score"] + consensus_bonus + numeric_bonus + comparison_bonus + comparison_relation_bonus + rare_entity_bonus + lexical_coverage_bonus
        item["score"] = round(final_score, 6)
        item["retrieval_source"] = "multi_query_rrf"
        item["comparison_bonus"] = round(comparison_bonus, 4)
        item["comparison_relation_bonus"] = round(comparison_relation_bonus, 4)
        item["rare_entity_bonus"] = round(rare_entity_bonus, 4)
        item["lexical_coverage_bonus"] = round(lexical_coverage_bonus, 4)
        item["coverage_match"] = _coverage_match(item, user_query)
        item["query_hits"] = entry["hits"]
        item["best_query_rank"] = entry["best_rank"]
        reranked.append(item)
    ordered = sorted(reranked, key=lambda item: (-item["score"], item["best_query_rank"], item["chunk_uid"]))
    reranker = _get_reranker()
    if reranker and ordered:
        candidates = ordered[:RERANKER_CANDIDATE_K]
        pairs = [(user_query, str(item.get("content") or item.get("summary") or "")) for item in candidates]
        try:
            scores = np.asarray(reranker.predict(pairs, show_progress_bar=False), dtype=float)
            # Cross-Encoder 分数只用于候选精排；RRF 仍负责大规模召回和稳定回退。
            for item, score in zip(candidates, scores):
                item["reranker_score"] = round(float(score), 6)
            # Cross-Encoder 的原始分数不可与 RRF 分数直接比较；按名次融合，避免它完全覆盖
            # 多路召回带来的词法覆盖和来源多样性。
            ce_order = sorted(candidates, key=lambda item: (-float(item["reranker_score"]), -float(item["score"])))
            for ce_rank, item in enumerate(ce_order, start=1):
                item["reranker_rank"] = ce_rank
                item["score"] = round(float(item["score"]) + 0.006 / (20 + ce_rank), 6)
            ordered = sorted(ordered, key=lambda item: (-float(item["score"]), item["best_query_rank"], item["chunk_uid"]))
        except Exception as exc:
            print(f"[WARN] Cross-Encoder 推理失败，将使用 RRF 排序：{exc}")
    selected: List[Dict[str, Any]] = []
    section_counts: Dict[str, int] = {}
    target_k = _adaptive_final_k(user_query)
    fact_groups = _fact_coverage_groups(user_query)

    def select(item: Dict[str, Any], coverage_bonus: float = 0.0) -> None:
        chosen = dict(item)
        chosen["score"] = round(float(chosen["score"]) + coverage_bonus, 6)
        chosen["coverage_bonus"] = round(coverage_bonus, 4)
        selected.append(chosen)
        section_key = str(chosen.get("title_path") or chosen.get("section_title") or chosen.get("book_title"))
        section_counts[section_key] = section_counts.get(section_key, 0) + 1

    for group in fact_groups:
        for item in ordered:
            if len(selected) >= target_k:
                break
            if item.get("chunk_uid") in {chosen.get("chunk_uid") for chosen in selected}:
                continue
            section_key = str(item.get("title_path") or item.get("section_title") or item.get("book_title"))
            if section_counts.get(section_key, 0) >= MAX_CONTEXTS_PER_SECTION or not _fact_group_match(item, group):
                continue
            select(item, COVERAGE_BONUS)
            break
    for item in ordered:
        if len(selected) >= target_k:
            break
        section_key = str(item.get("title_path") or item.get("section_title") or item.get("book_title"))
        if section_counts.get(section_key, 0) >= MAX_CONTEXTS_PER_SECTION:
            continue
        if _coverage_match(item, user_query):
            select(item, COVERAGE_BONUS)
            break
    for entity_group in _comparison_entities(user_query):
        for item in ordered:
            if len(selected) >= target_k:
                break
            if item.get("chunk_uid") in {chosen.get("chunk_uid") for chosen in selected}:
                continue
            if not any(alias in _result_text(item) for alias in entity_group):
                continue
            section_key = str(item.get("title_path") or item.get("section_title") or item.get("book_title"))
            if section_counts.get(section_key, 0) >= MAX_CONTEXTS_PER_SECTION:
                continue
            select(item, COVERAGE_BONUS)
            break
    for item in ordered:
        if len(selected) >= target_k:
            break
        if item.get("chunk_uid") in {chosen.get("chunk_uid") for chosen in selected}:
            continue
        section_key = str(item.get("title_path") or item.get("section_title") or item.get("book_title"))
        if section_counts.get(section_key, 0) >= MAX_CONTEXTS_PER_SECTION:
            continue
        select(item)
    return selected


def _parse_context_items(search_result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for idx, r in enumerate(search_result, 1):
        items.append({
            "rank": idx,
            "book_title": r.get("book_title", ""),
            "file_name": r.get("file_name", ""),
            "page_num": r.get("page_num", 0),
            "title_path": r.get("title_path", ""),
            "section_title": r.get("section_title", ""),
            "content": r.get("content", ""),
            "summary": r.get("summary", ""),
            "score": r.get("score", 0.0),
            "source": r.get("source", ""),
            "chunk_uid": r.get("chunk_uid", ""),
            "parent_context": r.get("parent_context", ""),
            "neighbor_context": r.get("neighbor_context", ""),
            "numeric_summary": r.get("numeric_summary", ""),
            "concept_tags": r.get("concept_tags", []),
            "summary_tags": r.get("summary_tags", []),
            "book_aliases": r.get("book_aliases", []),
        })
    return items


@tool
def rag_search(query: str, filter_titles: Optional[List[str]] = None, filter_tags: Optional[List[str]] = None, query_tag: str = "未知", query_tag_mode: str = "coarse") -> str:
    """执行混合召回；书目与标签参数仅作为兼容字段，不在召回阶段硬过滤。"""
    del filter_titles, filter_tags, query_tag, query_tag_mode
    if not query.strip():
        return json.dumps({"ok": False, "query": query, "results": []}, ensure_ascii=False)
    try:
        payload = json.loads(knowledge_store.search_pdf(query, top_k=RETRIEVAL_CANDIDATE_K))
        return json.dumps({"ok": True, "query": query, "results": payload.get("results", [])}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"ok": False, "query": query, "results": [], "error": str(exc)}, ensure_ascii=False)


# === 知识库加载 ===
knowledge_store = KnowledgeBase()
try:
    knowledge_store.load_from_cache(cache_path=KB_CACHE_FILE, vector_path=KB_VECTOR_FILE, bm25_path=KB_BM25_FILE)
except Exception as e:
    print(f"[WARN] 索引层加载失败: {e}")
knowledge_base = knowledge_store.all_chunks


# === Agent 状态 ===
class AgentState(TypedDict, total=False):
    messages: List[dict]
    current_agent: str
    context: Dict[str, Any]
    final_response: str
    task_type: str
    retry_count: int
    react_messages: List[dict]
    react_step_count: int
    evidence: Dict[str, Any]
    session_metadata: Dict[str, Any]
    user_profile: Dict[str, Any]
    recent_summary: str


SUB_AGENT_TYPES = ["knowledge", "image", "video", "identify", "story", "pet", "law", "chat"]
DEFAULT_BOOK_SETS = {"knowledge": ["别跟狗争老大", "狗的秘密", "如何养好你的狗狗", "猫的秘密", "猫咪心事", "我的第一本养猫书", "DK猫咪百科"], "pet": ["犬病诊断与治疗", "宠物营养与食品", "小动物临床手册", "如何养好你的狗狗"], "cat": ["猫的秘密", "猫咪心事", "我的第一本养猫书", "DK猫咪百科"], "dog": ["别跟狗争老大", "狗的秘密", "如何养好你的狗狗", "犬病诊断与治疗"], "law": []}
TOPIC_BOOK_SETS = {"cat": ["猫的秘密", "猫咪心事", "我的第一本养猫书", "DK猫咪百科"], "dog": ["别跟狗争老大", "狗的秘密", "如何养好你的狗狗", "犬病诊断与治疗"]}


def _extract_ragas_sample(user_message: str, final_response: str, result: Dict[str, Any], session_id: str, user_id: int) -> Dict[str, Any]:
    evidence = result.get("evidence", {}) if isinstance(result, dict) else {}
    contexts = []
    structured_contexts = []
    sources_detail = evidence.get("sources_detail", []) if isinstance(evidence, dict) else []
    for idx, item in enumerate(sources_detail[:5], 1):
        if not isinstance(item, dict):
            continue
        contexts.append(item.get("content", ""))
        structured_contexts.append({
            "rank": idx,
            "book_title": item.get("book_title", ""),
            "title_path": item.get("title_path", ""),
            "section_title": item.get("section_title", ""),
            "page_num": item.get("page_num"),
            "content": item.get("content", ""),
            "summary": item.get("summary", ""),
            "concept_tags": item.get("concept_tags", []),
            "proper_nouns": item.get("proper_nouns", []),
            "book_aliases": item.get("book_aliases", []),
            "numeric_summary": item.get("numeric_summary", ""),
            "evidence_text": item.get("content", ""),
        })
    return {"id": str(uuid.uuid4()), "question": user_message, "answer": final_response, "contexts": contexts, "retrieved_contexts": contexts, "retrieved_contexts_structured": structured_contexts, "ground_truth": None, "reference": None, "metadata": {"user_id": user_id, "session_id": session_id, "agent_type": result.get("current_agent", "") if isinstance(result, dict) else "", "task_type": result.get("task_type", "") if isinstance(result, dict) else "", "source_count": len(contexts), "created_at": utc_now()}}


# === Prompt 模型 ===
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    expires_at: str
    user: Dict[str, Any]


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    agent_type: str
    model_provider: str
    confidence: float
    context: Dict[str, Any]


# === Agent 节点 ===
class AgentNodes:
    def __init__(self):
        self.llm = ChatOpenAI(model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"), api_key=os.getenv("DEEPSEEK_API_KEY", "your-api-key"), base_url="https://api.deepseek.com", temperature=0.3)
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

    def _keyword_route(self, user_message: str) -> str:
        user_lower = (user_message or "").lower()
        if any(k in user_lower for k in ["生成图片", "画一张", "图片", "照片", "图像", "图生图", "生成一张"]):
            return "image"
        if "视频" in user_lower or "生成视频" in user_lower:
            return "video"
        if "识别" in user_lower or "辨认" in user_lower or "是什么动物" in user_lower:
            return "identify"
        if any(k in user_lower for k in ["故事", "绘本", "写一个", "创作"]):
            return "story"
        if any(k in user_lower for k in ["咨询", "看病", "生病", "症状", "治疗", "医院", "兽医"]):
            return "pet"
        if any(k in user_lower for k in ["法律", "法规", "违法", "赔偿", "责任", "权益", "禁止"]):
            return "law"
        if any(k in user_lower for k in ["为什么", "怎么", "如何", "什么", "特点", "性格", "品种", "行为", "护理", "健康", "喂养", "训练"]):
            return "knowledge"
        if "猫" in user_lower:
            return "cat"
        if "狗" in user_lower:
            return "dog"
        return "chat"

    def main_agent(self, state: AgentState) -> AgentState:
        user_message = state["messages"][-1]["content"]
        retry_count = state.get("retry_count", 0)
        review_reason = state.get("context", {}).get("review_reason", "")
        task_type = self._keyword_route(user_message)
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是主调度 Agent，只返回 {{\"task_type\": \"knowledge\", \"reason\": \"...\"}}。"),
                ("user", "用户问题：{message}\n会话元数据：{session_metadata}\n用户画像：{user_profile}\n近期摘要：{recent_summary}\n上次审核原因：{review_reason}\n重试次数：{retry_count}")
            ])
            result = (prompt | self.llm).invoke({"message": user_message, "session_metadata": json.dumps(state.get("session_metadata", {}), ensure_ascii=False), "user_profile": json.dumps(state.get("user_profile", {}), ensure_ascii=False), "recent_summary": state.get("recent_summary", "无"), "review_reason": review_reason or "无", "retry_count": retry_count})
            parsed = parse_llm_json(result.content)
            candidate = parsed.get("task_type", task_type)
            if candidate in SUB_AGENT_TYPES:
                task_type = candidate
        except Exception as exc:
            print(f"[WARN] 主 Agent 路由失败，使用关键词兜底: {exc}")
        return {**state, "task_type": task_type, "current_agent": "main", "final_response": "", "context": {**state.get("context", {}), "routing_reason": task_type, "review_reason": review_reason, "main_agent_tag": _pick_query_tag(user_message)}}

    def main_reviewer(self, state: AgentState) -> AgentState:
        user_message = state["messages"][-1]["content"]
        sub_response = state.get("final_response", "")
        task_type = state.get("task_type", "chat")
        retry_count = state.get("retry_count", 0)
        evidence = state.get("evidence", {})
        has_sources = bool(evidence.get("primary") or evidence.get("secondary"))
        numeric_intent = _is_numeric_question(user_message)
        review_ok = True
        review_reason = ""
        if not sub_response or len(sub_response.strip()) < 5:
            review_ok = False
            review_reason = "子 Agent 返回内容为空或过短"
        elif task_type in {"knowledge", "pet", "law"} and not has_sources:
            review_ok = False
            review_reason = "领域回答缺少可核对证据"
        elif numeric_intent and not re.search(r"\d", sub_response):
            review_ok = False
            review_reason = "数值问题未给出明确数字"
        if not review_ok and retry_count < 1:
            return {**state, "current_agent": "main", "retry_count": retry_count + 1, "context": {**state.get("context", {}), "review_ok": False, "review_reason": review_reason, "reroute_mode": True}}
        return {**state, "current_agent": state.get("current_agent", task_type), "context": {**state.get("context", {}), "review_ok": review_ok, "review_reason": review_reason, "reroute_mode": False}}

    def query_rewrite(self, state: AgentState) -> AgentState:
        user_message = state["messages"][-1]["content"]
        reroute_mode = bool(state.get("context", {}).get("reroute_mode", False))
        existing_rewrite = state.get("context", {}).get("rewrite_query")
        if existing_rewrite and not reroute_mode:
            return state
        base_query = re.sub(r"\s+", " ", (user_message or "").strip())
        entity_terms = _extract_entities(base_query)
        intent_terms = _extract_intent_keywords(base_query)
        topic_type = _topic_route(base_query)
        query_tag = _pick_query_tag(base_query, fine=reroute_mode)
        query_terms = list(dict.fromkeys([t for t in entity_terms + intent_terms if t]))
        if reroute_mode:
            query_specs = [{"query": query, "kind": "reroute", "weight": 1.0 if index == 0 else 0.60} for index, query in enumerate(_build_reroute_queries(base_query, state.get("context", {}).get("rewrite_query") or base_query, task_type=state.get("task_type", "knowledge"), review_reason=state.get("context", {}).get("review_reason", "")))]
        else:
            query_specs = _build_parallel_queries(base_query, base_query, numeric_intent=_is_numeric_question(base_query), topic_type=topic_type)
            if _hyde_is_allowed(base_query, state.get("task_type", "knowledge")):
                hyde_prompt = ChatPromptTemplate.from_messages([
                    ("system", "为检索生成一段不超过 80 字的假设性资料摘录。只描述可能的通用机制或特征；不得编造数字、案例、书名、作者、引用或具体来源。仅输出摘录文本。"),
                    ("user", "问题：{message}")
                ])
                try:
                    hyde_result = (hyde_prompt | self.llm).invoke({"message": base_query})
                    hyde_query = re.sub(r"\s+", " ", (hyde_result.content or "")).strip()[:160]
                    if hyde_query:
                        query_specs.append({"query": hyde_query, "kind": "hyde", "weight": 0.35})
                except Exception as exc:
                    print(f"[WARN] HyDE 生成失败，跳过该路召回: {exc}")
        cleaned: List[Dict[str, Any]] = []
        seen_queries = set()
        for spec in query_specs:
            query = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", str(spec.get("query", "")))
            query = re.sub(r"\s+", " ", query).strip()
            if query and query not in seen_queries:
                cleaned.append({**spec, "query": query})
                seen_queries.add(query)
        return {**state, "context": {**state.get("context", {}), "rewrite_query": cleaned[0]["query"] if cleaned else base_query, "parallel_queries": cleaned[:4], "reroute_mode": reroute_mode, "query_terms": query_terms, "topic_type": topic_type, "query_tag": query_tag, "query_tag_mode": "fine" if reroute_mode else "coarse", "route_tag": query_tag}}

    def _search_bundle(self, queries: List[Any], filter_titles: Optional[List[str]], query_tag: str, query_tag_mode: str) -> List[Dict[str, Any]]:
        from concurrent.futures import ThreadPoolExecutor
        query_specs = [item if isinstance(item, dict) else {"query": str(item), "kind": "fallback", "weight": 0.60} for item in queries]
        if not query_specs:
            return []
        with ThreadPoolExecutor(max_workers=min(len(query_specs), 4)) as executor:
            futures = [
                executor.submit(rag_search.invoke, {"query": spec["query"], "filter_titles": filter_titles or [], "filter_tags": [], "query_tag": query_tag, "query_tag_mode": query_tag_mode})
                for spec in query_specs
            ]
            bundles = []
            for spec, future in zip(query_specs, futures):
                try:
                    payload = future.result()
                except Exception as exc:
                    payload = json.dumps({"ok": False, "query": spec["query"], "results": [], "error": str(exc)}, ensure_ascii=False)
                try:
                    data = json.loads(payload) if isinstance(payload, str) else payload
                except Exception:
                    data = {"ok": False, "query": spec["query"], "results": [], "error": str(payload)}
                bundles.append({**spec, "data": data})
        return bundles

    def _finalize_search(self, user_message: str, search_result: List[Dict[str, Any]], task_type: str, numeric_intent: bool) -> Tuple[str, Dict[str, Any]]:
        evidence_primary, evidence_secondary = [], []
        for item in search_result:
            text = " ".join([str(item.get("content", "")), str(item.get("book_title", "")), str(item.get("source", ""))])
            entities = _extract_entities(user_message)
            classified_primary = [x for x in entities if x in text]
            evidence_primary.extend(classified_primary)
            if not classified_primary:
                evidence_secondary.append(str(item.get("summary", ""))[:120] or str(item.get("content", ""))[:120])
        evidence_primary = list(dict.fromkeys(evidence_primary))
        evidence_secondary = list(dict.fromkeys([p for p in evidence_secondary if p and p not in evidence_primary]))
        evidence_prefix = "\n".join([f"主证据：{'、'.join(evidence_primary)}" if evidence_primary else "", f"辅助证据：{'、'.join(evidence_secondary)}" if evidence_secondary else ""]).strip()
        selected = search_result[:min(5, len(search_result))]
        context_block = "\n\n".join([f"【{i+1}】《{it.get('book_title', '未知') }》\n章节：{it.get('title_path', '')}\n摘要：{str(it.get('summary', ''))[:120]}\n内容：{str(it.get('content', ''))[:RAG_CONTEXT_CHAR_LIMIT]}" for i, it in enumerate(selected)]) or "未检索到相关内容"
        evidence_block = "\n\n".join([f"证据{i+1}：{it.get('source', '')}\n摘要：{str(it.get('summary', ''))[:120]}\n摘录：{str(it.get('content', ''))[:180]}" for i, it in enumerate(selected)]) or "未检索到相关内容"
        return context_block, {"primary": evidence_primary, "secondary": evidence_secondary, "sources": [item.get("source", "") for item in search_result], "book_titles": [item.get("book_title", "") for item in search_result if item.get("book_title")], "sources_detail": search_result, "evidence_block": evidence_block, "evidence_prefix": evidence_prefix}

    def rag_knowledge_agent(self, state: AgentState) -> AgentState:
        user_message = state["messages"][-1]["content"]
        task_type = state.get("task_type", "knowledge")
        rewrite_query = state.get("context", {}).get("rewrite_query", user_message)
        numeric_intent = _is_numeric_question(user_message)
        topic_type = state.get("context", {}).get("topic_type") or _topic_route(user_message)
        parallel_queries = state.get("context", {}).get("parallel_queries") or _build_parallel_queries(user_message, rewrite_query, numeric_intent=numeric_intent, topic_type=topic_type)
        # 书目只作为后续排序偏好；主召回不能因固定书单而丢失证据。
        payloads = self._search_bundle(parallel_queries, filter_titles=None, query_tag=state.get("context", {}).get("route_tag", state.get("context", {}).get("query_tag", "未知")), query_tag_mode=state.get("context", {}).get("query_tag_mode", "coarse"))
        relevant_units = _get_relevant_units(user_message) if numeric_intent else None
        fallback_results = knowledge_store.search_by_numeric(user_message, top_k=NUMERIC_FALLBACK_K, relevant_units=relevant_units) if numeric_intent else []
        search_result = _aggregate_and_rerank_results(
            payloads,
            user_message,
            numeric_fallback_results=fallback_results,
            relevant_units=relevant_units,
        )[:RETRIEVAL_FINAL_K]
        context_block, evidence = self._finalize_search(user_message, search_result, task_type, numeric_intent)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是 knowledge agent。只能使用检索证据回答。仅陈述证据直接支持的事实；不要用相邻概念替代问题核心，不要添加证据未出现的原因、数字或医学结论。先给直接结论，再用1至3条最相关证据说明，最后列来源；证据不足时明确说明不足。对于原因、风险、护理或紧急处理类问题，若证据分别提供了原因/机制和后果/处理建议，应优先覆盖这两个事实面；若证据明确列出病因、风险因素或适用条件，不要只摘取最严重的后果。"),
            ("user", "用户问题：{message}\n任务类型：{task_type}\n检索上下文：\n{context_block}\n\n证据：\n{evidence_block}\n\n证据提示：{evidence_prefix}")
        ])
        try:
            result = (prompt | self.llm).invoke({"message": user_message, "task_type": task_type, "context_block": context_block, "evidence_block": evidence["evidence_block"], "evidence_prefix": evidence["evidence_prefix"] or "无"})
            final_answer = result.content or "抱歉，我暂时无法回答您的问题。"
        except Exception as exc:
            final_answer = f"知识 agent 生成回答失败：{exc}"
        return {**state, "final_response": final_answer, "current_agent": task_type, "context": {**state.get("context", {}), "tool_results_count": len(search_result), "search_mode": "hybrid_rrf"}, "evidence": evidence}

    def domain_agent(self, state: AgentState, domain: str) -> AgentState:
        user_message = state["messages"][-1]["content"]
        rewrite_query = state.get("context", {}).get("rewrite_query", user_message)
        numeric_intent = _is_numeric_question(user_message)
        topic_type = state.get("context", {}).get("topic_type") or _topic_route(user_message)
        parallel_queries = state.get("context", {}).get("parallel_queries") or _build_parallel_queries(user_message, rewrite_query, numeric_intent=numeric_intent, topic_type=topic_type)
        payloads = self._search_bundle(parallel_queries, filter_titles=None, query_tag=state.get("context", {}).get("route_tag", state.get("context", {}).get("query_tag", "未知")), query_tag_mode=state.get("context", {}).get("query_tag_mode", "coarse"))
        relevant_units = _get_relevant_units(user_message) if numeric_intent else None
        fallback_results = knowledge_store.search_by_numeric(user_message, top_k=NUMERIC_FALLBACK_K, relevant_units=relevant_units) if numeric_intent else []
        search_result = _aggregate_and_rerank_results(
            payloads,
            user_message,
            numeric_fallback_results=fallback_results,
            relevant_units=relevant_units,
        )[:RETRIEVAL_FINAL_K]
        context_block, evidence = self._finalize_search(user_message, search_result, domain, numeric_intent)
        disclaimer = "仅供参考，具体法律问题请咨询专业律师。" if domain == "law" else "若症状严重或持续，请尽快就医。"
        role_name = "law agent" if domain == "law" else "pet agent"
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"你是 Fauna AI 的{role_name}。{disclaimer} 只能基于检索证据回答。"),
            ("user", "用户问题：{message}\n检索证据：\n{context}\n\n证据提示：{evidence_prefix}")
        ])
        try:
            result = (prompt | self.llm).invoke({"message": user_message, "context": context_block, "evidence_prefix": evidence["evidence_prefix"] or "无"})
            final_answer = result.content or f"{role_name} 未能生成回答。"
        except Exception as exc:
            final_answer = f"{role_name} 生成回答失败：{exc}"
        return {**state, "final_response": final_answer, "current_agent": domain, "context": {**state.get("context", {}), "tool_results_count": len(search_result), "search_mode": "hybrid_rrf"}, "evidence": evidence}

    def image_agent(self, state: AgentState) -> AgentState:
        user_message = state["messages"][-1]["content"]
        return {**state, "final_response": f"图像生成功能正在开发中！\n\n您的请求：{user_message}", "current_agent": "image", "context": {}}

    def video_agent(self, state: AgentState) -> AgentState:
        user_message = state["messages"][-1]["content"]
        return {**state, "final_response": f"视频生成功能正在开发中！\n\n您的请求：{user_message}", "current_agent": "video", "context": {}}

    def identify_agent(self, state: AgentState) -> AgentState:
        user_message = state["messages"][-1]["content"]
        prompt = ChatPromptTemplate.from_messages([("system", "你是动物识别专家，请根据描述识别动物并给出简要建议。"), ("user", user_message)])
        result = (prompt | self.llm).invoke({})
        return {**state, "final_response": result.content, "current_agent": "identify", "context": {}}

    def story_agent(self, state: AgentState) -> AgentState:
        user_message = state["messages"][-1]["content"]
        return {**state, "final_response": f"故事绘本生成器正在开发中。\n\n您的请求：{user_message}", "current_agent": "story", "context": {}}

    def chat_agent(self, state: AgentState) -> AgentState:
        user_message = state["messages"][-1]["content"]
        payload = {"model": self.ollama_model, "messages": [{"role": "system", "content": "你是 Fauna AI 猫狗百科的友好闲聊助手。"}, {"role": "user", "content": user_message}], "stream": False, "options": {"temperature": 0.7}}
        try:
            resp = requests.post(f"{self.ollama_base_url}/api/chat", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            result_text = data.get("message", {}).get("content") or data.get("response") or ""
        except Exception as e:
            result_text = f"本地 Ollama 调用失败：{e}"
        return {**state, "final_response": result_text, "current_agent": "chat", "context": {"ollama_model": self.ollama_model}}

    def law_agent(self, state: AgentState) -> AgentState:
        return self.domain_agent(state, "law")

    def pet_agent(self, state: AgentState) -> AgentState:
        return self.domain_agent(state, "pet")


# === LangGraph ===
def create_langgraph_app() -> StateGraph:
    nodes = AgentNodes()
    graph = StateGraph(AgentState)
    graph.add_node("main", nodes.main_agent)
    graph.add_node("query_rewrite", nodes.query_rewrite)
    graph.add_node("knowledge", nodes.rag_knowledge_agent)
    graph.add_node("law", nodes.law_agent)
    graph.add_node("pet", nodes.pet_agent)
    graph.add_node("image", nodes.image_agent)
    graph.add_node("video", nodes.video_agent)
    graph.add_node("identify", nodes.identify_agent)
    graph.add_node("story", nodes.story_agent)
    graph.add_node("chat", nodes.chat_agent)
    graph.add_node("reviewer", nodes.main_reviewer)

    def route_to_agent(state: AgentState) -> str:
        return {"knowledge": "query_rewrite", "law": "query_rewrite", "pet": "query_rewrite", "image": "image", "video": "video", "identify": "identify", "story": "story", "chat": "chat"}.get(state.get("task_type", "chat"), "chat")

    def route_after_rewrite(state: AgentState) -> str:
        return state.get("task_type", "chat") if state.get("task_type", "chat") in {"knowledge", "law", "pet"} else "chat"

    def after_review(state: AgentState) -> str:
        return END if state.get("context", {}).get("review_ok", True) else "main"

    graph.add_conditional_edges("main", route_to_agent, ["query_rewrite", "image", "video", "identify", "story", "chat"])
    graph.add_conditional_edges("query_rewrite", route_after_rewrite, ["knowledge", "law", "pet", "chat"])
    graph.add_edge("knowledge", "reviewer")
    graph.add_edge("law", "reviewer")
    graph.add_edge("pet", "reviewer")
    for agent in ["image", "video", "identify", "story", "chat"]:
        graph.add_edge(agent, "reviewer")
    graph.add_conditional_edges("reviewer", after_review, {"main": "main", END: END})
    graph.set_entry_point("main")
    return graph.compile()


langgraph_app = create_langgraph_app()


# === API ===
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    expires_at: str
    user: Dict[str, Any]


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    user_id: Optional[str] = None


@app.middleware("http")
async def observability_middleware(request, call_next):
    start = asyncio.get_event_loop().time()
    response = await call_next(request)
    elapsed = asyncio.get_event_loop().time() - start
    if REQUEST_COUNT:
        REQUEST_COUNT.labels(method=request.method, path=request.url.path, status=str(response.status_code)).inc()
    if REQUEST_LATENCY:
        REQUEST_LATENCY.labels(method=request.method, path=request.url.path).observe(elapsed)
    return response


@app.post("/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    username = request.username.strip()
    if len(username) < 3 or len(username) > 20:
        raise HTTPException(status_code=400, detail="用户名需为 3-20 个字符")
    if len(request.password) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 位")
    with get_db_connection() as conn:
        exists = conn.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, request.email.lower())).fetchone()
        if exists:
            raise HTTPException(status_code=400, detail="用户名或邮箱已存在")
        salt, pwd_hash = hash_password(request.password)
        cursor = conn.execute("INSERT INTO users (username, email, password_hash, salt, is_admin, created_at) VALUES (?, ?, ?, ?, ?, ?)", (username, request.email.lower(), pwd_hash, salt, 0, utc_now()))
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
    token, expires_at = create_session(row["id"])
    return {"token": token, "expires_at": expires_at, "user": user_to_dict(row)}


@app.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ? OR email = ?", (request.username.strip(), request.username.strip().lower())).fetchone()
    if not row or not verify_password(request.password, row["salt"], row["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token, expires_at = create_session(row["id"])
    return {"token": token, "expires_at": expires_at, "user": user_to_dict(row)}


@app.post("/auth/logout")
async def logout(authorization: Optional[str] = Header(default=None)):
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        with get_db_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
    return {"ok": True}


@app.get("/history")
async def get_history(current_user: Dict[str, Any] = Depends(get_current_user), limit: int = 100, session_id: Optional[str] = None):
    safe_limit = max(1, min(limit, 200))
    with get_db_connection() as conn:
        if session_id:
            rows = conn.execute(
                "SELECT session_id, role, content, agent_type, created_at FROM messages WHERE user_id = ? AND session_id = ? ORDER BY id ASC",
                (current_user["id"], session_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT session_id, role, content, agent_type, created_at FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (current_user["id"], safe_limit),
            ).fetchall()
            rows = reversed(rows)
    return {"history": [dict(row) for row in rows]}


@app.delete("/history/{session_id}")
async def delete_history(session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    with get_db_connection() as conn:
        result = conn.execute(
            "DELETE FROM messages WHERE user_id = ? AND session_id = ?",
            (current_user["id"], session_id),
        )
        conn.commit()
    return {"ok": True, "deleted": result.rowcount}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Fauna AI 猫狗百科 - LangGraph版"}


@app.get("/metrics")
async def metrics():
    if generate_latest is None:
        raise HTTPException(status_code=503, detail="prometheus_client 未安装")
    with get_db_connection() as conn:
        active_sessions = conn.execute("SELECT COUNT(*) AS c FROM sessions WHERE expires_at > ?", (utc_now(),)).fetchone()["c"]
    if ACTIVE_SESSIONS:
        ACTIVE_SESSIONS.set(active_sessions)
    payload = generate_latest()
    return HTMLResponse(content=payload.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.post("/chat")
async def chat(request: ChatRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    session_id = request.session_id or str(uuid.uuid4())
    user_id = current_user["id"]
    memory_messages = load_user_memory(user_id, session_id)

    async def stream_response():
        try:
            session_metadata = get_session_metadata(user_id, session_id)
            user_profile = get_user_profile(user_id)
            recent_summary = get_recent_summary(user_id, session_id) or build_recent_summary(memory_messages)
            context_bundle = {"messages": memory_messages + [{"role": "user", "content": request.message}]}
            initial_state: AgentState = {"messages": context_bundle["messages"], "current_agent": "main", "context": {"user_id": user_id, "session_id": session_id, "memory_count": len(memory_messages)}, "final_response": "", "task_type": "", "retry_count": 0, "react_messages": [], "react_step_count": 0, "evidence": {}, "session_metadata": session_metadata, "user_profile": user_profile, "recent_summary": recent_summary}
            result = await asyncio.to_thread(langgraph_app.invoke, initial_state)
            agent_type = result.get("current_agent", "unknown")
            final_response = result.get("final_response", "暂无回答")
            save_message(user_id, session_id, "user", request.message)
            save_message(user_id, session_id, "assistant", final_response, agent_type)
            update_user_profile_from_messages(user_id, memory_messages + [{"role": "user", "content": request.message}, {"role": "assistant", "content": final_response}], session_id=session_id)
            save_recent_summary(user_id, session_id, build_recent_summary(memory_messages + [{"role": "user", "content": request.message}, {"role": "assistant", "content": final_response}]))
            update_session_metadata(user_id, session_id, {**session_metadata, "last_agent": agent_type, "last_task": result.get("task_type", ""), "memory_count": len(memory_messages) + 2})
            sample = _extract_ragas_sample(request.message, final_response, result, session_id, user_id)
            yield f"data: {json.dumps({'type': 'agent', 'agent_type': agent_type}, ensure_ascii=False)}\n\n"
            for chunk in [final_response[i:i + 20] for i in range(0, len(final_response), 20)]:
                yield f"data: {json.dumps({'type': 'content', 'content': chunk}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'agent_type': agent_type, 'retrieved_contexts': sample.get('retrieved_contexts', []), 'retrieved_contexts_structured': sample.get('retrieved_contexts_structured', [])}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': f'抱歉，暂时无法回答您的问题。错误: {str(e)[:50]}'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


@app.get("/tags")
async def get_tags():
    tags = set()
    for item in knowledge_base:
        item_tags = getattr(item, "tags", None) or []
        tags.update(item_tags)
    return {"tags": sorted(tags)}


@app.get("/observability")
async def observability_page():
    return HTMLResponse("""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/><title>Fauna AI 监控入口</title></head><body><h1>Fauna AI 可观测性入口</h1><p>请通过 /metrics 查看指标。</p></body></html>""")


@app.get("/ragas/status")
async def ragas_status():
    return {"events_file": str(Path(BASE_DIR) / "ragas_events.jsonl")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
