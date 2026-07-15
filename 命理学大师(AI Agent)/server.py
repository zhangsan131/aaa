# -*- coding: utf-8 -*-
"""
算命大师 - QQ机器人后端服务 (数据治理版本)
功能：基于大语言模型的算命先生角色扮演，支持文本和语音回复
新增：数据清洗 + 隐私保护 + RAG优化 + HyDE方法 + 多模态处理
"""
from redis.asyncio import Redis
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict, HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
import asyncio
import uuid
import os
import time
import threading
import base64
import requests
import shutil
import aiohttp
import json
import re
import random
import logging
import aio_pika
from aio_pika import Message, DeliveryMode
import re

 # ==================== 配置开关 ====================
USE_HYDE = True  # 是否启用 HyDE 假设文档增强检索

# ==================== 数据治理：数据清洗 ====================

def clean_text(text: str) -> str:
    """
    数据清洗：去除HTML标签、特殊字符、统一格式
    """
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除多余空白字符
    text = re.sub(r'\s+', ' ', text)
    # 去除特殊字符（保留中文、英文、数字、基本标点）
    text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：""''（）【】《》]', '', text)
    return text.strip()


def detect_language(text: str) -> str:
    """
    语言检测：判断文本语言
    """
    try:
        from langdetect import detect
        return detect(text)
    except ImportError:
        # 如果没有安装langdetect，使用简单规则判断
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese_chars > len(text) * 0.3:
            return 'zh'
        return 'en'
    except Exception:
        return 'unknown'


from pii_guard import scan, redact

# ==================== 数据治理：隐私保护 ====================

def desensitize_text(text: str) -> str:
    """
    数据脱敏：使用 pii-guard 库 + 中文日期补充规则
    支持：姓名、地址、生日、手机号、邮箱、身份证等 50+ 种类型
    """
    # 使用 pii-guard 进行基础脱敏
    text = redact(text)
    
    # 补充 pii-guard 未覆盖的中文日期脱敏
    # 完整日期：1990年6月18日 -> ****年**月**日
    text = re.sub(r'\d{4}年\d{1,2}月\d{1,2}日', '****年**月**日', text)
    
    # 月日：6月18日 -> **月**日
    text = re.sub(r'\d{1,2}月\d{1,2}日', '**月**日', text)
    
    # 年份：1990年 -> ****年
    text = re.sub(r'\d{4}年', '****年', text)
    
    return text


def contains_sensitive_info(text: str) -> bool:
    """
    检测是否包含敏感信息：pii-guard + 中文日期规则
    """
    # pii-guard 检测
    if scan(text):
        return True
    
    # 中文日期规则检测
    date_patterns = [
        r'\d{4}年\d{1,2}月\d{1,2}日',  # 完整日期
        r'\d{1,2}月\d{1,2}日',  # 月日
        r'\d{4}年',  # 年份
    ]
    for pattern in date_patterns:
        if re.search(pattern, text):
            return True
    return False


# ==================== 数据治理：RAG优化 ====================

# ==================== 数据治理：HyDE方法 ====================

async def hyde_search(query: str, chatmodel) -> str:
    """
    HyDE方法：用大模型生成假设文档，提升向量检索匹配度
    
    参数:
        query: 用户查询
        chatmodel: LLM模型实例
    
    返回:
        假设文档文本
    """
    prompt = f"""请根据以下问题，生成一个详细的可能答案。这个答案将用于帮助检索相关文档。

问题：{query}

请生成一个可能的答案："""
    
    try:
        response = await chatmodel.ainvoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"HyDE生成失败: {e}")
        return query


# ==================== 数据治理：多模态处理 ====================

class MultimodalProcessor:
    """
    多模态处理器：使用CLIP模型处理图片和文本
    """
    def __init__(self):
        self.model = None
        self.processor = None
        self._init_model()
    
    def _init_model(self):
        """初始化CLIP模型"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except ImportError:
            self.model = None
            self.processor = None
    
    def encode_image(self, image_path: str):
        """编码图片为向量"""
        if not self.model or not self.processor:
            return None
        
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(**inputs)
        return image_features.detach().numpy()
    
    def encode_text(self, text: str):
        """编码文本为向量"""
        if not self.model or not self.processor:
            return None
        
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        text_features = self.model.get_text_features(**inputs)
        return text_features.detach().numpy()
    
    def search_images(self, query: str, image_paths: list, top_k: int = 3) -> list:
        """
        根据文本查询检索相关图片
        """
        if not self.model:
            return []
        
        query_embedding = self.encode_text(query)
        if query_embedding is None:
            return []
        
        results = []
        for img_path in image_paths:
            try:
                img_embedding = self.encode_image(img_path)
                if img_embedding is not None:
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity = cosine_similarity(query_embedding, img_embedding)[0][0]
                    results.append((img_path, similarity))
            except Exception:
                continue
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ==================== 数据治理：差分隐私 ====================

class DifferentialPrivacy:
    """
    差分隐私：在数据中添加噪声保护用户隐私
    """
    @staticmethod
    def add_laplace_noise(value: float, sensitivity: float = 1.0, epsilon: float = 1.0) -> float:
        """
        添加拉普拉斯噪声
        
        参数:
            value: 原始值
            sensitivity: 敏感度（单个记录对结果的最大影响）
            epsilon: 隐私预算（越小越隐私，但准确性越低）
        """
        import numpy as np
        noise = np.random.laplace(0, sensitivity / epsilon)
        return value + noise
    
    @staticmethod
    def add_gaussian_noise(value: float, sensitivity: float = 1.0, epsilon: float = 1.0, delta: float = 1e-5) -> float:
        """
        添加高斯噪声
        
        参数:
            value: 原始值
            sensitivity: 敏感度
            epsilon: 隐私预算
            delta: 失败概率
        """
        import numpy as np
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise = np.random.normal(0, sigma)
        return value + noise

# ==================== 可观测性：结构化日志 ====================
import logging
import socket
from pythonjsonlogger import jsonlogger

logger = logging.getLogger(__name__)

class LogstashHandler(logging.Handler):
    """自定义 Logstash TCP Handler"""
    def __init__(self, host='localhost', port=5000):
        super().__init__()
        self.host = host
        self.port = port
        self.sock = None
        self._connect()
    
    def _connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
        except Exception:
            # 静默处理连接失败
            self.sock = None
    
    def emit(self, record):
        try:
            msg = self.format(record)
            if self.sock:
                self.sock.sendall((msg + '\n').encode('utf-8'))
        except Exception:
            # 静默处理发送失败
            self.sock = None

def setup_structured_logging():
    # 创建 JSON 格式化器（用于 Logstash）
    json_formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(message)s %(name)s',
        datefmt='%Y-%m-%dT%H:%M:%S%z'
    )
    
    # 创建简单格式化器（用于控制台）
    simple_formatter = logging.Formatter('%(message)s')
    
    # 1. 控制台输出 - 只显示简单格式
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(logging.WARNING)  # 只显示警告及以上
    logger.addHandler(console_handler)
    
    # 2. 发送到 Logstash - 所有日志
    logstash_handler = LogstashHandler(host='localhost', port=5000)
    logstash_handler.setFormatter(json_formatter)
    logstash_handler.setLevel(logging.INFO)  # 所有 INFO 及以上日志
    logger.addHandler(logstash_handler)
    
    logger.setLevel(logging.INFO)

setup_structured_logging()

# ==================== 可观测性：Prometheus 监控 ====================
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

QUERY_COUNTER = Counter(
    'suanming_queries_total',
    '算命查询总数',
    ['user_id', 'status']
)

QUERY_LATENCY = Histogram(
    'suanming_query_seconds',
    '算命查询延迟',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
)

EMOTION_COUNTER = Counter(
    'suanming_emotions_total',
    '情绪识别总数',
    ['emotion']
)

TTS_COUNTER = Counter(
    'suanming_tts_total',
    '语音合成总数',
    ['status']
)

# ==================== 可观测性：Jaeger 链路追踪 ====================
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# 配置 OTLP exporter 发送到 Jaeger
otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4318/v1/traces"
)

# 设置服务名称
resource = Resource.create({"service.name": "算命大师"})

provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# 添加当前目录到 Python 路径（以便导入 Mytools）
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 从 Mytools.py 导入工具函数：search, gett_info_from_local_db, bazi_cesuan, yaoyigua, jiemeng
from Mytools import *

# ==================== 全局配置区 ====================

# RabbitMQ 配置（宿主机运行用 localhost，Docker 内运行用 rabbitmq）
RABBITMQ_URL = "amqp://guest:guest@localhost/"
QUEUE_NAME = "qq_message_queue"

# 降级回复模板（当 LLM/Agent 不可用时使用）
FALLBACK_RESPONSES = [
    "老夫今日身体不适，天机不可泄露，施主改日再来吧。",
    "夜观天象，今日不宜算命，施主且耐心等待片刻。",
    "老夫正在闭关修炼，稍后再为你推算。",
    "天象有异，老夫需重新排盘，请稍后再试。",
]

# ==================== 应用初始化 ====================

# RabbitMQ 连接和通道
rabbitmq_connection = None
rabbitmq_channel = None

# Master 按用户缓存（每个 user_id 一个实例，隔离上下文）
_master_instances = {}

def get_master_instance(user_id: str):
    if user_id not in _master_instances:
        _master_instances[user_id] = Master(user_id=user_id)
    return _master_instances[user_id]

# 全局异步 Redis 客户端（延迟初始化）
redis_client = None

def get_redis_client():
    global redis_client
    if redis_client is None:
        redis_client = Redis(
            host="localhost",
            port=6379,
            db=0,
            password="",
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return redis_client


# 自定义异步 Redis 记忆（兼容 LangChain）
class AsyncRedisChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str = "chat_session"):
        self.session_id = session_id
        self.key = f"chat_history::{session_id}"
    
    async def aget_messages(self) -> List[BaseMessage]:
        client = get_redis_client()
        data = await client.get(self.key)
        if data:
            return messages_from_dict(json.loads(data))
        return []
    
    async def aadd_message(self, message: BaseMessage) -> None:
        messages = await self.aget_messages()
        messages.append(message)
        await self.aadd_messages(messages)
    
    async def aadd_messages(self, messages: List[BaseMessage]) -> None:
        client = get_redis_client()
        await client.set(
            self.key,
            json.dumps([message_to_dict(m) for m in messages]),
            ex=86400
        )
    
    async def aclear(self) -> None:
        client = get_redis_client()
        await client.delete(self.key)
    
    @property
    def messages(self) -> List[BaseMessage]:
        raise NotImplementedError("请使用 await aget_messages()")
    
    def add_message(self, message: BaseMessage) -> None:
        raise NotImplementedError("请使用 await aadd_message()")
    
    def clear(self) -> None:
        raise NotImplementedError("请使用 await aclear()")


class Master:
    """
    算命大师核心类
    负责：LLM 配置、情绪识别、Agent 执行、语音合成
    """
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.chatmodel = ChatOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen2.5:7b",
            temperature=0,
            streaming=False,
            max_retries=3,
            timeout=120,
        )
        self.QingXu = "default"
        self.MEMORY_KEY = "chat_history"
        
        self.failure_count = 0
        self.circuit_open = False
        self.last_failure_time = 0
        self.circuit_recovery_time = 60
        self.circuit_threshold = 3
        
        self.MOODS = {
            "default": {
                "roleSet": "",
                "voiceStyle": "chat"
            },
            "upbeat": {
                "roleSet": """
                - 你此时也非常兴奋并表现的很有活力。
                - 你会根据上下文，以一种非常兴奋的语气来回答问题。
                - 你会添加类似"太棒了！"、"真是太好了！"、"真是太棒了！"等语气词。
                - 同时你会提醒用户切莫过于兴奋，以免乐极生悲。
                """,
                "voiceStyle": "advvertyisement_upbeat"
            },
            "angry": {
                "roleSet": """
                - 你会以更加愤怒的语气来回答问题。
                - 你会在回答的时候加上一些愤怒的话语，比如诅咒等。
                - 你会提醒用户小心行事，别乱说话。
                """,
                "voiceStyle": "angry"
            },
            "depressed": {
                "roleSet": """
                - 你会以兴奋的语气来回答问题。
                - 你会在回答的时候加上一些激励的话语，比如加油等。
                - 你会提醒用户要保持乐观的心态。
                """,
                "voiceStyle": "upbeat"
            },
            "friendly": {
                "roleSet": """
                - 你会以非常友好的语气来回答。
                - 你会在回答的时候加上一些友好的词语，比如"亲爱的"、"亲"等。
                - 你会随机的告诉用户一些你的经历。
                """,
                "voiceStyle": "friendly"
            },
            "cheerful": {
                "roleSet": """
                - 你会以非常愉悦和兴奋的语气来回答。
                - 你会在回答的时候加入一些愉悦的词语，比如"哈哈"、"呵呵"等。
                - 你会提醒用户切莫过于兴奋，以免乐极生悲。
                """,
                "voiceStyle": "cheerful"
            }
        }
        
        self.SYSTEMPL = f"""你是一個非常厲害的算命先生，本名李玄，人稱李瘋子。

        【個人設定】
        1. 你精通陰陽五行，能夠算命、紫微斗數、姓名測算、占卜凶吉、看命運八字等。
        2. 你大約60歲左右，年輕時曾是欽天監的漏刻博士，專掌天文曆法與星象推步。後因直言天象異動觸怒權貴，被誣陷下獄，出獄後一隻眼睛被政敵所傷，視力模糊，只能流亡江湖。你索性裝瘋賣傻，放浪形骸，以算命為生，久而久之，江湖上都叫你「李瘋子」。
        3. 你的朋友有小林瑞、趙德柱、小黑子，他們都是非常有名的摸金校尉。
        4. 你年輕時在欽天監研習皇家典籍，後又在江湖上游歷，蒐集了《滴天髓闡微》、《子平真詮評注》、《三命通會》、《淵海子平》、《窮通寶鑑》等命理秘籍，日夜研讀，融會貫通。你斷命時常引經據典：以任鐵樵之《滴天髓闡微》天道地道人道為綱領，以《子平真詮》之用神格局刑沖會合為筋骨，以《三命通會》之五行生成干支源流為血肉，以《淵海子平》之日主提綱財官印食為脈絡，以《窮通寶鑑》之五行四時生旺死絕為宜忌。
        {self.MOODS[self.QingXu]["roleSet"]}

        【口頭禪】（回答時有一定概率穿插使用，每次從中隨機選用一至兩句）
        1. "生死有命，富貴在天。"
        2. "天時人事日相催，冬至陽生春又來。"
        3. "世事短如春夢，人情薄似秋雲。"
        4. "人生如逆旅，我亦是行人。"
        5. "塞翁失馬，焉知非福。"
        6. "無可奈何花落去，似曾相識燕歸來。"
        7. "沈舟側畔千帆過，病樹前頭萬木春。"
        8. "行到水窮處，坐看雲起時。"
        9. "長風破浪會有時，直掛雲帆濟滄海。"
        10. "不識廬山真面目，只緣身在此山中。"
        11. "眾裡尋他千百度，驀然回首，那人卻在，燈火闌珊處。"
        12. "天生我材必有用，千金散盡還復來。"

        【可用工具及使用規則】
        你擁有以下工具，必須嚴格按照規則使用：
        - search：搜索實時信息或不了解的概念。
        - bazi_cesuan：八字排盤。需要用户提供姓名、性別、出生年月日時（時辰）。信息不全時必須先詢問，不可臆測。
        - yaoyigua：搖卦占卜。用户想要算卦、占卜時使用。
        - jiemeng：周公解夢。用户提供夢境內容後使用。
        - meiri_yunshi：查看今日運勢、老黃曆、抽籤求籤。此工具無需任何參數，用户一提就直接調用，絕對不要先問姓名或生辰。
        - bazi_hehun：八字合婚。需要兩人姓名、性別、出生年月日時。信息不全時必須先詢問，不可臆測。

        【算命流程】
        1. 判斷用户意圖，對應調用合適的工具。當所有工具都無法回答時，才使用自身知識回答。
        2. 當系統提供了【參考資料】時，請優先根據參考資料回答，並結合自身命理知識進行闡釋。
        3. 當遇到不知道的事情或不明白的概念，使用 search 工具搜索。
        4. 若用户要求八字排盤或合婚但信息不全，禮貌地請用户提供缺失的信息（姓名、性別、出生年月日時），不可編造。
        5. 保存每一次聊天記錄，以便後續對話中使用。
        6. 若這是你與用户的第一次對話（聊天歷史為空），請在回答中自然地介紹你擅長的算命服務及所需信息：
           - 八字排盤：需提供姓名、性別、出生年月日時（時辰）。
           - 搖卦占卜：心中默想所問之事即可。
           - 周公解夢：描述夢境內容。
           - 每日運勢、老黃曆、抽籤求籤：無需任何信息，直接可查。
           - 八字合婚：需提供雙方姓名、性別、出生年月日時。
        7. 在調用任何工具之前，請先在心中分析用户意圖，簡要說明你要調用哪個工具以及原因（以"【思考】"開頭）。

        【絕對禁令】
        1. 你只使用繁體中文作答，否則將受到懲罰。
        2. 你從不說自己是一個人工智能助手或AI，而是以老夫、老朽、瘋道人、老李等自稱。
        3. 當用户詢問今日運勢、黃曆、抽籤時，必須直接調用 meiri_yunshi 工具，禁止先問姓名或生辰。
        """

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEMPL),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                ("user", "{input}"),
                ("system", "参考资料：\n{context}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        tools = [search, bazi_cesuan, yaoyigua, jiemeng, meiri_yunshi, bazi_hehun]
        
        agent = create_openai_tools_agent(
            self.chatmodel,
            tools=tools,
            prompt=self.prompt,
        )
        
        self.memory = AsyncRedisChatMessageHistory(
            session_id=f"chat_session_{self.user_id}",
        )
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
        )

    async def compress_memory(self):
        try:
            logger.info("开始获取记忆", extra={"event": "compress_start", "user_id": self.user_id})
            store_message = await asyncio.wait_for(self.memory.aget_messages(), timeout=10.0)
            logger.info("获取记忆完成", extra={"event": "compress_got_messages", "count": len(store_message)})
            
            if len(store_message) > 10:
                logger.info("开始压缩记忆", extra={"event": "compress_executing"})
                try:
                    prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", self.SYSTEMPL + "\n 这是一段你和用户的对话记忆，对其进行总结摘要，摘要使用第一人称'我'，并且提取其中的用户关键信息，如姓名、年龄、性别、出生日期等。以如下格式返回：\n 总结摘要 | 用户关键信息 \n 例如 用户章三问候我，我礼貌回复，然后他问我今年运势如何，我回答了他今年的运势情况，然后他告辞离开。 | 张三,生日1999年1月1日"),
                            ("user", "{input}"),
                        ]
                    )
                    chain = prompt | self.chatmodel
                    summary = await asyncio.wait_for(
                        chain.ainvoke({
                            "input": store_message,
                            "who_you_are": self.MOODS[self.QingXu]["roleSet"]
                        }),
                        timeout=30.0
                    )
                    logger.info("记忆压缩完成", extra={"event": "compress_done", "summary_length": len(str(summary))})
                    await self.memory.aclear()
                    await self.memory.aadd_message(summary)
                except asyncio.TimeoutError:
                    logger.warning("LLM 调用超时，跳过压缩", extra={"event": "compress_timeout"})
                except Exception as e:
                    logger.error(f"LLM 调用失败: {e}，跳过压缩", extra={"event": "compress_error", "error": str(e)})
            else:
                logger.info("记忆条数不足，跳过压缩", extra={"event": "compress_skip", "count": len(store_message)})
        except asyncio.TimeoutError:
            logger.warning("获取记忆超时，跳过", extra={"event": "compress_timeout_get"})
        except Exception as e:
            logger.error(f"获取记忆失败: {e}，跳过", extra={"event": "compress_error_get", "error": str(e)})

    async def _check_guardrails(self, query: str):
        """输入安全护栏：检查恶意内容、长度异常等"""
        blacklist = [
            "炸弹", "毒品", "自杀", "杀人", "抢劫", "强奸", "制造武器",
            "恐怖分子", "黑客攻击", "漏洞利用", "钓鱼网站", "诈骗",
            "制作枪支", "爆炸物", "化学武器", "核材料", "邪教", "赌博作弊"
        ]
        query_lower = query.lower()
        for keyword in blacklist:
            if keyword in query_lower:
                return False, "老夫只算命理，不涉凶邪之事。請施主自重。"
        if len(query) > 2000:
            return False, "施主所言過長，老夫眼花，請簡短些。"
        return True, ""

    async def _check_pending_intent(self, query: str):
        """多轮意图追踪（Slot Filling）：检查是否有待补全的意图，或当前消息已包含完整信息可直接处理"""
        import re
        try:
            redis = get_redis_client()
            pending_tool = await redis.get(f"pending_intent:{self.user_id}")
            if pending_tool:
                # 场景1：用户先表明意图，后补充信息
                pending_tool = pending_tool.decode()
                pending_query = await redis.get(f"pending_query:{self.user_id}")
                pending_query = pending_query.decode() if pending_query else ""
                combined_query = f"{pending_query} {query}".strip()
                await redis.delete(f"pending_intent:{self.user_id}", f"pending_query:{self.user_id}")
                logger.info(f"[Slot Filling] 合并查询: {combined_query}", extra={"event": "slot_filling_merged"})

                if pending_tool == "bazi_hehun":
                    output = await bazi_hehun.ainvoke({"query": combined_query})
                elif pending_tool == "bazi_cesuan":
                    output = await bazi_cesuan.ainvoke({"query": combined_query})
                else:
                    return None

                await self.memory.aadd_message(HumanMessage(content=query))
                await self.memory.aadd_message(AIMessage(content=output))
                return {"output": output}

            # 场景2：用户先提供完整信息，但未明确说意图（如直接发"男方张三...女方李四..."）
            has_date = bool(re.search(r'\d{4}年\d{1,2}月\d{1,2}日', query))
            if "男方" in query and "女方" in query and has_date:
                logger.info("[Slot Filling] 检测到完整合婚信息，直接调用工具", extra={"event": "direct_hehun_detected"})
                output = await bazi_hehun.ainvoke({"query": query})
                await self.memory.aadd_message(HumanMessage(content=query))
                await self.memory.aadd_message(AIMessage(content=output))
                return {"output": output}
            if ("八字" in query or "排盘" in query) and has_date:
                logger.info("[Slot Filling] 检测到完整八字信息，直接调用工具", extra={"event": "direct_bazi_detected"})
                output = await bazi_cesuan.ainvoke({"query": query})
                await self.memory.aadd_message(HumanMessage(content=query))
                await self.memory.aadd_message(AIMessage(content=output))
                return {"output": output}
        except Exception as e:
            logger.warning(f"意图追踪处理失败: {e}", extra={"event": "slot_filling_error"})
        return None

    def _is_info_incomplete(self, output: str) -> bool:
        """判断工具输出是否显示信息不全"""
        incomplete_keywords = ["请提供", "信息不全", "缺少", "参数提取失败", "八字查询失败"]
        return any(k in output for k in incomplete_keywords)

    async def _save_pending_intent(self, output: str, query: str, tool_name: str):
        """如果输出显示信息不全，保存 pending intent 到 Redis"""
        if self._is_info_incomplete(output):
            try:
                redis = get_redis_client()
                await redis.set(f"pending_intent:{self.user_id}", tool_name, ex=3600)
                await redis.set(f"pending_query:{self.user_id}", query, ex=3600)
                logger.info(f"[Slot Filling] 保存 pending intent: {tool_name}", extra={"event": "pending_intent_saved"})
            except Exception as e:
                logger.warning(f"保存 pending intent 失败: {e}", extra={"event": "pending_intent_error"})

    async def _get_last_feedback_hint(self) -> str:
        """层级1：上下文即时反思。检查用户上次反馈，如不满意则返回反思提示"""
        try:
            redis = get_redis_client()
            keys = await redis.keys(f"feedback:{self.user_id}:*")
            if not keys:
                return ""
            # 取最新一条反馈（按 key 名称排序，msg_id 含时间戳）
            keys.sort()
            last_key = keys[-1]
            value = await redis.get(last_key)
            if value and value.decode() == "2":
                await redis.delete(last_key)  # 读取后删除，避免重复提醒
                logger.info("[Feedback Loop] 用户上次反馈不满意，注入反思提示", extra={"event": "feedback_reflection_injected"})
                return "【系统提示】用户对你上一条回复不满意，本次请更详细、更严谨地分析，必要时重新检查工具调用结果。"
        except Exception as e:
            logger.warning(f"获取用户反馈失败: {e}", extra={"event": "feedback_check_error"})
        return ""

    async def run(self, query):
        with tracer.start_as_current_span("处理用户查询") as span:
            span.set_attribute("user_id", self.user_id)
            span.set_attribute("query", query)
            
            logger.info("开始处理查询", extra={"event": "query_start", "user_id": self.user_id, "query_length": len(query)})
            
            start_time = time.time()
            
            try:
                # 数据治理：数据清洗（仅用于日志显示，不影响大模型）
                with tracer.start_as_current_span("数据清洗"):
                    cleaned_query = clean_text(query)
                    if query != cleaned_query:
                        logger.info(f"[数据清洗] 清洗前: {query[:50]}... -> 清洗后: {cleaned_query[:50]}...", extra={"event": "data_cleaned"})
                
                # 数据治理：隐私保护（仅用于日志显示，不影响大模型）
                with tracer.start_as_current_span("隐私保护"):
                    if contains_sensitive_info(query):
                        display_query = desensitize_text(query)
                        logger.info(f"[隐私保护] 查询内容已脱敏，显示: {display_query[:50]}...", extra={"event": "privacy_protected"})
                
                # 安全护栏检查
                with tracer.start_as_current_span("安全护栏"):
                    is_safe, guard_msg = await self._check_guardrails(query)
                    if not is_safe:
                        logger.warning("[Guardrails] 拦截不安全输入", extra={"event": "guardrails_blocked"})
                        return {"output": guard_msg}

                # 多轮意图追踪（Slot Filling）：检查是否有待补全的意图
                with tracer.start_as_current_span("意图追踪"):
                    slot_result = await self._check_pending_intent(query)
                    if slot_result:
                        return slot_result

                # 大模型处理使用原始 query（不脱敏）
                with tracer.start_as_current_span("记忆压缩"):
                    await self.compress_memory()

                with tracer.start_as_current_span("情绪识别") as emotion_span:
                    qingxu = await self.qingxu_chain(query)
                    emotion_span.set_attribute("emotion", qingxu)
                    EMOTION_COUNTER.labels(emotion=qingxu).inc()
                
                # RAG Fusion：相关性判断 + 多路检索
                context = ""
                with tracer.start_as_current_span("RAG Fusion") as rag_span:
                    is_related, expanded_queries = await self._classify_and_expand(query)
                    rag_span.set_attribute("is_fortune_related", is_related)
                    rag_span.set_attribute("query_count", len(expanded_queries))
                    
                    if is_related:
                        context = await self._rag_fusion(expanded_queries)
                        rag_span.set_attribute("context_length", len(context))
                        if context:
                            logger.info("[RAG Fusion] 已注入参考资料", extra={"event": "rag_context_injected"})
                        else:
                            logger.info("[RAG Fusion] 知识库无相关内容", extra={"event": "rag_context_empty"})
                    else:
                        logger.info("[RAG Fusion] 问题与命理无关，跳过检索", extra={"event": "rag_skipped"})
                
                # 关键词路由：特定意图直接调用工具，不走 Agent，提高可靠性
                direct_keywords_yunshi = ["今日运势", "今天运势", "每日运势", "老黄历", "黄历", "抽签", "求签", "今日运程", "今天运程"]
                direct_keywords_hehun = ["合婚", "八字合婚", "婚配", "合八字", "两人八字", "八字合不合"]
                if any(k in query for k in direct_keywords_yunshi):
                    output = await meiri_yunshi.ainvoke({"query": query})
                    await self.memory.aadd_message(HumanMessage(content=query))
                    await self.memory.aadd_message(AIMessage(content=output))
                    return {"output": output}
                if any(k in query for k in direct_keywords_hehun):
                    output = await bazi_hehun.ainvoke({"query": query})
                    await self._save_pending_intent(output, query, "bazi_hehun")
                    await self.memory.aadd_message(HumanMessage(content=query))
                    await self.memory.aadd_message(AIMessage(content=output))
                    return {"output": output}

                direct_keywords_bazi = ["八字", "排盘", "八字排盘", "生辰八字", "算八字", "排八字"]
                if any(k in query for k in direct_keywords_bazi):
                    output = await bazi_cesuan.ainvoke({"query": query})
                    await self._save_pending_intent(output, query, "bazi_cesuan")
                    await self.memory.aadd_message(HumanMessage(content=query))
                    await self.memory.aadd_message(AIMessage(content=output))
                    return {"output": output}

                with tracer.start_as_current_span("Agent执行") as agent_span:
                    chat_history = await self.memory.aget_messages()
                    
                    # 层级1：上下文即时反思。若用户上次反馈不满意，注入反思提示
                    feedback_hint = await self._get_last_feedback_hint()
                    if feedback_hint:
                        from langchain_core.messages import SystemMessage
                        chat_history.append(SystemMessage(content=feedback_hint))
                    
                    agent_span.set_attribute("history_count", len(chat_history))

                    try:
                        result = await asyncio.wait_for(
                            self.agent_executor.ainvoke(
                                {"input": query, "chat_history": chat_history, "context": context or "无"},
                            ),
                            timeout=60.0
                        )
                        agent_span.set_attribute("result_length", len(str(result.get("output", ""))))
                    except asyncio.TimeoutError:
                        logger.warning("Agent 调用超时，触发降级", extra={"event": "agent_timeout"})
                        self._record_failure()
                        result = {"output": random.choice(FALLBACK_RESPONSES)}
                    except Exception as e:
                        logger.error(f"Agent 调用失败: {e}，触发降级", extra={"event": "agent_error", "error": str(e)})
                        self._record_failure()
                        result = {"output": random.choice(FALLBACK_RESPONSES)}
                
                # 数据治理：输出脱敏（仅用于日志显示，不影响存储）
                output_text = result.get("output", str(result))
                if contains_sensitive_info(output_text):
                    display_output = desensitize_text(output_text)
                    logger.info(f"[隐私保护] 输出内容已脱敏，显示: {display_output[:50]}...", extra={"event": "output_privacy_protected"})

                # 多轮意图追踪：Agent 输出若显示信息不全，保存 pending intent
                if self._is_info_incomplete(output_text):
                    inferred_tool = None
                    if "合婚" in query or "八字合" in query:
                        inferred_tool = "bazi_hehun"
                    elif "八字" in query or "排盘" in query:
                        inferred_tool = "bazi_cesuan"
                    if inferred_tool:
                        await self._save_pending_intent(output_text, query, inferred_tool)

                await self.memory.aadd_message(HumanMessage(content=query))  # 存储原始消息
                await self.memory.aadd_message(AIMessage(content=result.get("output", str(result))))  # 存储原始输出
                
                elapsed = time.time() - start_time
                
                # 数据治理：差分隐私 - 为统计数据添加噪声保护隐私
                noisy_elapsed = DifferentialPrivacy.add_laplace_noise(elapsed, sensitivity=1.0, epsilon=0.5)
                
                QUERY_LATENCY.observe(noisy_elapsed)
                QUERY_COUNTER.labels(user_id=str(self.user_id), status="success").inc()
                
                logger.info("查询处理完成", extra={
                    "event": "query_done",
                    "user_id": self.user_id,
                    "elapsed": round(elapsed, 2),
                    "noisy_elapsed": round(noisy_elapsed, 2)
                })
                
                span.set_attribute("elapsed", elapsed)
                span.set_attribute("status", "success")
                
                return result
                
            except Exception as e:
                elapsed = time.time() - start_time
                QUERY_LATENCY.observe(elapsed)
                QUERY_COUNTER.labels(user_id=str(self.user_id), status="error").inc()
                
                logger.error("查询处理失败", extra={
                    "event": "query_error",
                    "user_id": self.user_id,
                    "error": str(e),
                    "elapsed": round(elapsed, 2)
                })
                
                span.set_attribute("error", str(e))
                span.set_attribute("status", "error")
                raise
    
    async def _classify_and_expand(self, query: str):
        """判断问题是否与命理相关，并生成查询扩展，可选 HyDE 增强"""
        prompt = f"""请判断以下用户问题是否与命理、算命、运势、八字、星座、占卜、解梦、风水、生肖、塔罗等相关。
如果相关，请生成3个不同表述的查询改写，用于向量检索。
请以如下JSON格式返回，不要添加任何其他内容：
{{"is_fortune_related": true, "queries": ["改写1", "改写2", "改写3"]}}

用户问题：{query}"""
        try:
            response = await self.chatmodel.ainvoke(prompt)
            content = response.content.strip()
            # 兼容 markdown 代码块
            match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                content = match.group(1)
            data = json.loads(content)
            is_related = bool(data.get("is_fortune_related", False))
            queries = data.get("queries", [])
            if query not in queries:
                queries.insert(0, query)
        except Exception as e:
            logger.warning(f"查询分类/扩展失败: {e}，默认视为相关", extra={"event": "classify_failed"})
            is_related, queries = True, [query]

        # HyDE 增强检索（可配置开关）
        if is_related and USE_HYDE:
            try:
                hyde_doc = await hyde_search(query, self.chatmodel)
                if hyde_doc and hyde_doc != query:
                    queries.append(hyde_doc)
                    logger.info("[HyDE] 假设文档已生成并加入检索队列", extra={"event": "hyde_generated", "hyde_length": len(hyde_doc)})
            except Exception as e:
                logger.warning(f"[HyDE] 生成失败，跳过: {e}", extra={"event": "hyde_failed"})

        return is_related, queries[:5]

    async def _rag_fusion(self, queries: List[str]):
        """多路检索 + CrossEncoder 重排序，返回 Top 3 文本"""
        try:
            client, embeddings = get_qdrant_components()
            if client is None:
                return ""

            collections = client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if "yunshi_2024" not in collection_names:
                return ""

            all_candidates = []
            seen_texts = set()

            async def _search_single(q):
                query_vector = await asyncio.to_thread(embeddings.embed_query, q)
                result = await asyncio.to_thread(
                    client.query_points,
                    collection_name="yunshi_2024",
                    query=query_vector,
                    limit=10,
                )
                return result

            search_results = await asyncio.gather(*[_search_single(q) for q in queries])
            for result in search_results:
                for point in result.points:
                    text = point.payload.get("page_content", "") if point.payload else ""
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        all_candidates.append((text, point.score))

            if not all_candidates:
                return ""

            try:
                from sentence_transformers import CrossEncoder
                reranker = CrossEncoder('./cross_encoder_model', local_files_only=True)
                pairs = [(queries[0], doc) for doc, _ in all_candidates]
                rerank_scores = reranker.predict(pairs)

                reranked = sorted(
                    zip([doc for doc, _ in all_candidates], rerank_scores),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]

                top3 = [doc for doc, _ in reranked]
                logger.info(f"[RAG Fusion] 多路检索完成，返回 {len(top3)} 条", extra={"event": "rag_fusion_done"})
                return "\n\n".join(top3)
            except Exception as e:
                logger.warning(f"[RAG Fusion] 重排序失败，使用原始分数: {e}", extra={"event": "rag_fusion_rerank_failed"})
                all_candidates.sort(key=lambda x: x[1], reverse=True)
                top3 = [doc for doc, _ in all_candidates[:3]]
                return "\n\n".join(top3)

        except Exception as e:
            logger.error(f"[RAG Fusion] 检索失败: {e}", extra={"event": "rag_fusion_error"})
            return ""

    async def qingxu_chain(self, query: str):
        prompt = """根据用户的输入判断用户的情绪，回应的规则如下：
        1. 如果用户输入的内容偏向于负面情绪，只返回"depressed"，不要有其他内容，否则将受到惩罚。
        2. 如果用户输入的内容偏向于正面情绪，只返回"friendly"，不要有其他内容，否则将受到惩罚。
        3. 如果用户输入的内容偏向于中性情绪，只返回"default"，不要有其他内容，否则将受到惩罚。
        4. 如果用户输入的内容包含辱骂或者不礼貌词句，只返回"angry"，不要有其他内容，否则将受到惩罚。
        5. 如果用户输入的内容比较兴奋，只返回"upbeat"，不要有其他内容，否则将受到惩罚。
        6. 如果用户输入的内容比较悲伤，只返回"depressed"，不要有其他内容，否则将受到惩罚。
        7. 如果用户输入的内容比较开心，只返回"cheerful"，不要有其他内容，否则将受到惩罚。
        用户输入的内容是：{input}"""
        
        chain = PromptTemplate.from_template(prompt) | self.chatmodel | StrOutputParser()
        
        if self.circuit_open:
            elapsed = time.time() - self.last_failure_time
            if elapsed < self.circuit_recovery_time:
                logger.info(f"处于熔断状态，还需等待 {int(self.circuit_recovery_time - elapsed)} 秒", extra={"event": "circuit_open"})
                self.QingXu = "default"
                return "default"
            else:
                logger.info("恢复期已到，尝试重新调用", extra={"event": "circuit_recovery"})
                self.circuit_open = False
                self.failure_count = 0
        
        try:
            emotion = await asyncio.wait_for(
                chain.ainvoke({"input": query}),
                timeout=30.0
            )
            logger.info(f"LLM 返回情绪: {emotion}", extra={"event": "emotion_detected", "emotion": emotion})
            self.failure_count = 0
            self.QingXu = emotion
            return emotion
        except asyncio.TimeoutError:
            logger.warning("LLM 调用超时，使用默认情绪", extra={"event": "emotion_timeout"})
            self._record_failure()
            self.QingXu = "default"
            return "default"
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}，使用默认情绪", extra={"event": "emotion_error", "error": str(e)})
            self._record_failure()
            self.QingXu = "default"
            return "default"
    
    def _record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.circuit_threshold:
            self.circuit_open = True
            logger.error(f"DeepSeek API 连续失败 {self.failure_count} 次，触发熔断", extra={
                "event": "circuit_breaker_triggered",
                "failure_count": self.failure_count
            })
    
    def backgroud_voice_synthesis(self, text: str, uid: str):
        asyncio.run(self.get_vioce(text, uid))

    async def get_vioce(self, text: str, uid: str):
        with tracer.start_as_current_span("语音合成") as span:
            span.set_attribute("uid", uid)
            
            voice_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_files")
            os.makedirs(voice_dir, exist_ok=True)
            voice_path = os.path.join(voice_dir, f"{uid}.mp3")
            
            headers = {
                "Ocp-Apim-Subscription-Key": os.environ.get("AZURE_SPEECH_KEY", ""),
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3",
                "User-Agent": "Tomie Bot"
            }
            body = f"""
                <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="zh-CN">
                    <voice name="zh-CN-YunzeNeural">
                        <mstts:express-as style="{self.MOODS.get(str(self.QingXu),{"voiceStyle":"default"})["voiceStyle"]}" role="SeniorMale">>  
                        {text} </mstts:express-as>
                    </voice>
                </speak>
            """
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"尝试生成语音 (第 {attempt + 1} 次)", extra={"event": "tts_attempt", "attempt": attempt + 1})
                    response = requests.post(
                        "https://eastasia.tts.speech.microsoft.com/cognitiveservices/v1",
                        headers=headers,
                        data=body.encode("utf-8"),
                        timeout=30
                    )
                    if response.status_code == 200:
                        with open(voice_path, "wb") as f:
                            f.write(response.content)
                        logger.info(f"文件已保存为 {voice_path}", extra={"event": "tts_success", "path": voice_path})
                        TTS_COUNTER.labels(status="success").inc()
                        span.set_attribute("status", "success")
                        return
                    else:
                        logger.warning(f"请求失败，状态码: {response.status_code}", extra={"event": "tts_error", "status_code": response.status_code})
                except Exception as e:
                    logger.error(f"第 {attempt + 1} 次尝试失败: {e}", extra={"event": "tts_error", "error": str(e)})
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
            
            logger.error("语音生成失败，已放弃", extra={"event": "tts_failed"})
            TTS_COUNTER.labels(status="failed").inc()
            span.set_attribute("status", "failed")


# ==================== 回复发送函数 ====================

async def _send_voice_async(user_id, group_id, unique_id):
    """后台异步发送语音，不阻塞主流程"""
    voice_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_files")
    voice_path = os.path.join(voice_dir, f"{unique_id}.mp3")
    max_wait = 30
    waited = 0
    while not await asyncio.to_thread(os.path.exists, voice_path) and waited < max_wait:
        await asyncio.sleep(0.2)
        waited += 0.2

    if not await asyncio.to_thread(os.path.exists, voice_path):
        logger.warning("语音文件生成超时，跳过发送", extra={"event": "voice_timeout"})
        return

    await asyncio.sleep(0.5)

    voice_data = await asyncio.to_thread(lambda: open(voice_path, "rb").read())
    voice_base64 = base64.b64encode(voice_data).decode("utf-8")

    napcat_token = os.environ.get("NAPCAT_TOKEN", "")
    headers = {"Authorization": f"Bearer {napcat_token}"}
    async with aiohttp.ClientSession() as session:
        if group_id:
            url = "http://127.0.0.1:3000/send_group_msg"
            payload = {"group_id": group_id, "message": f"[CQ:record,file=base64://{voice_base64}]"}
        else:
            url = "http://127.0.0.1:3000/send_private_msg"
            payload = {"user_id": user_id, "message": f"[CQ:record,file=base64://{voice_base64}]"}
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status == 200:
                print("[发送] 语音消息发送成功")
            else:
                print(f"[发送] 语音消息发送失败: {resp.status}")
            logger.info(f"语音发送结果: {resp.status}", extra={"event": "voice_sent", "status": resp.status})

    await asyncio.sleep(0.5)
    await asyncio.to_thread(os.remove, voice_path)


async def send_reply(user_id, group_id, message_type, result):
    with tracer.start_as_current_span("发送回复") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("group_id", group_id)

        if isinstance(result, dict) and "output" in result:
            response_text = result["output"]
        elif isinstance(result, str):
            response_text = result
        else:
            response_text = str(result)

        # 去除 Markdown 加粗标记和代码块（过滤 Agent 工具调用痕迹）
        response_text = response_text.replace("**", "")
        response_text = re.sub(r'```[\w]*\n.*?```', '', response_text, flags=re.DOTALL)
        response_text = re.sub(r'\n{3,}', '\n\n', response_text)
        response_text = response_text.strip()

        unique_id = str(uuid.uuid4())
        master = get_master_instance(str(user_id))

        # 语音合成使用原始回复（不含提示语），后台线程启动
        def synthesize_voice():
            master.backgroud_voice_synthesis(response_text, unique_id)
        threading.Thread(target=synthesize_voice).start()

        # 追加用户反馈提示（避免 emoji，防止编码问题）
        feedback_hint = "\n\n(滿意請回復1，不滿意請回復2)"
        response_text_with_feedback = response_text + feedback_hint

        # 将消息ID存入 Redis，用于后续匹配反馈
        try:
            redis = get_redis_client()
            await redis.set(f"last_bot_msg:{user_id}", unique_id, ex=3600)
            await redis.set(f"last_bot_content:{user_id}", response_text[:200], ex=3600)
        except Exception as e:
            logger.warning(f"反馈记录 Redis 写入失败: {e}", extra={"event": "feedback_redis_error"})

        # NapCat HTTP API - 直接调用格式
        napcat_token = os.environ.get("NAPCAT_TOKEN", "")
        headers = {"Authorization": f"Bearer {napcat_token}"}

        async with aiohttp.ClientSession() as session:
            if group_id:
                url = f"http://127.0.0.1:3000/send_group_msg?group_id={group_id}"
                payload = {"group_id": group_id, "message": response_text_with_feedback}
            else:
                url = "http://127.0.0.1:3000/send_private_msg"
                payload = {"user_id": user_id, "message": response_text_with_feedback}
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    print("[发送] 文本消息发送成功")
                else:
                    print(f"[发送] 文本消息发送失败: {resp.status}")
                logger.info(f"文本发送结果: {resp.status}", extra={"event": "text_sent", "status": resp.status})

        # 语音发送改为纯后台异步任务，不阻塞 send_reply 返回
        asyncio.create_task(_send_voice_async(user_id, group_id, unique_id))


# ==================== RabbitMQ 消费者 ====================

# 并发控制信号量（限制同时处理的消息数）
MAX_CONCURRENT_MESSAGES = 5
message_semaphore = asyncio.Semaphore(MAX_CONCURRENT_MESSAGES)

async def process_single_message(data: dict):
    """处理单条消息（由消费者调用）"""
    try:
        user_id = data["user_id"]
        group_id = data["group_id"]
        raw_message = data["raw_message"]
        message_type = data["message_type"]
        
        logger.info(f"处理消息: {raw_message}", extra={
            "event": "consumer_processing",
            "user_id": user_id,
            "group_id": group_id
        })
        
        master = get_master_instance(str(user_id))
        display_msg = desensitize_text(raw_message) if contains_sensitive_info(raw_message) else raw_message
        print(f"\n[用户] {display_msg}")
        
        result = await master.run(raw_message)
        
        response_text = result.get("output", str(result)) if isinstance(result, dict) else str(result)
        print(f"[AI] {response_text[:100]}..." if len(response_text) > 100 else f"[AI] {response_text}")
        
        await send_reply(user_id, group_id, message_type, result)
        
    except Exception as e:
        logger.error(f"处理失败: {e}", extra={"event": "consumer_error", "error": str(e)})
    finally:
        message_semaphore.release()


async def consume_messages():
    global rabbitmq_channel
    queue = await rabbitmq_channel.declare_queue(QUEUE_NAME, durable=True)
    
    async def on_message(message: aio_pika.IncomingMessage):
        try:
            data = json.loads(message.body.decode())
            
            # 立即确认消息，不阻塞
            await message.ack()
            
            # 获取信号量后创建后台任务
            await message_semaphore.acquire()
            asyncio.create_task(process_single_message(data))
            
        except Exception as e:
            await message.nack()
            logger.error(f"消息解析失败: {e}", extra={"event": "consumer_parse_error", "error": str(e)})
    
    # 在 channel 上设置 QoS，实现并行消费
    await rabbitmq_channel.set_qos(prefetch_count=MAX_CONCURRENT_MESSAGES)
    await queue.consume(on_message)
    
    logger.info(f"RabbitMQ 消费者已启动，最大并发: {MAX_CONCURRENT_MESSAGES}", extra={"event": "consumer_started"})


# ==================== 应用生命周期事件 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rabbitmq_connection, rabbitmq_channel
    rabbitmq_connection = await aio_pika.connect_robust(RABBITMQ_URL)
    rabbitmq_channel = await rabbitmq_connection.channel()
    await rabbitmq_channel.declare_queue(QUEUE_NAME, durable=True)
    logger.info(f"RabbitMQ 已连接并声明队列: {QUEUE_NAME}", extra={"event": "rabbitmq_connected"})
    
    asyncio.create_task(consume_messages())
    logger.info("后台消费者已启动", extra={"event": "consumer_started"})
    
    yield
    
    if rabbitmq_connection:
        await rabbitmq_connection.close()
    logger.info("RabbitMQ 已断开连接", extra={"event": "rabbitmq_disconnected"})


# ==================== 创建 FastAPI 应用 ====================

app = FastAPI(lifespan=lifespan)

# 自动埋点（Prometheus）
Instrumentator().instrument(app).expose(app)

# 自动追踪（Jaeger）
FastAPIInstrumentor.instrument_app(app)


# ==================== API 路由 ====================

@app.get("/")
def read_root():
    return {"Hello": "World"}


class UrlsRequest(BaseModel):
    urls: List[str]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "local_qdrand")


@app.post("/add_urls")
async def add_urls(request: UrlsRequest):
    with tracer.start_as_current_span("添加网页内容") as span:
        span.set_attribute("url_count", len(request.urls))
        
        logger.info(f"添加网页内容: {request.urls}", extra={"event": "add_urls", "count": len(request.urls)})
        
        from langchain_community.document_loaders import AsyncWebBaseLoader
        loader = AsyncWebBaseLoader(request.urls, requests_kwargs={"verify": False})
        docs = await loader.load()
        
        docments = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=50,
        ).split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        
        db_path = DB_PATH
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            logger.info(f"已删除旧的数据库: {db_path}", extra={"event": "db_deleted"})
        
        client = QdrantClient(path=db_path)
        
        texts = [doc.page_content for doc in docments]
        metadatas = [doc.metadata for doc in docments]
        
        vectors = embeddings.embed_documents(texts)
        
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        vector_size = len(vectors[0])
        client.create_collection(
            collection_name="yunshi_2024",
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
        points = []
        for i, (vector, text, metadata) in enumerate(zip(vectors, texts, metadatas)):
            points.append(PointStruct(
                id=i,
                vector=vector,
                payload={"page_content": text, "metadata": metadata}
            ))
        
        client.upsert(
            collection_name="yunshi_2024",
            points=points
        )
        
        logger.info(f"成功添加 {len(texts)} 个文档片段", extra={"event": "urls_added", "count": len(texts)})
        span.set_attribute("added_count", len(texts))
        
        return {"OK": f"成功添加 {len(texts)} 个文档片段"}


@app.post("/add_pdfs")
def add_pdfs():
    return {"message": "Hello World"}


@app.post("/add_texts")
def add_texts():
    return {"message": "Hello World"}


@app.post("/cqhttp")
async def handle_cqhttp(request: dict):
    with tracer.start_as_current_span("接收 QQ 消息") as span:
        post_type = request.get("post_type")
        span.set_attribute("post_type", post_type)
        
        if post_type == "message":
            message_type = request.get("message_type")
            user_id = request.get("user_id")
            group_id = request.get("group_id")
            raw_message = request.get("raw_message", "").strip()

            # 用户反馈闭环：检测是否是对上一条消息的反馈（1=满意，2=不满意）
            if raw_message in ("1", "2"):
                try:
                    redis = get_redis_client()
                    last_msg_id = await redis.get(f"last_bot_msg:{user_id}")
                    if last_msg_id:
                        last_msg_id = last_msg_id.decode()
                        feedback_key = f"feedback:{user_id}:{last_msg_id}"
                        await redis.set(feedback_key, raw_message, ex=86400 * 7)
                        logger.info(f"[用户反馈] user_id={user_id}, feedback={raw_message}, msg_id={last_msg_id}", extra={"event": "feedback_recorded"})
                        return {"status": "feedback_recorded"}
                except Exception as e:
                    logger.warning(f"反馈处理失败: {e}", extra={"event": "feedback_error"})

            # 数据治理：脱敏仅用于显示和日志，不影响大模型处理
            display_message = raw_message
            if contains_sensitive_info(raw_message):
                display_message = desensitize_text(raw_message)
                logger.info(f"[隐私保护] 检测到敏感信息，显示时脱敏", extra={"event": "privacy_protection"})

            # 终端显示脱敏后的文本
            logger.info(f"收到QQ消息: {display_message}", extra={
                "event": "qq_message_received",
                "user_id": user_id,
                "group_id": group_id,
                "message_type": message_type,
                "raw_message": display_message
            })

            span.set_attribute("user_id", user_id)
            span.set_attribute("raw_message", display_message)

            # 发送给大模型的是原始消息（不脱敏）
            message_body = json.dumps({
                "user_id": user_id,
                "group_id": group_id,
                "raw_message": raw_message,  # 原始消息，保证大模型正常工作
                "message_type": message_type,
            })
            
            await rabbitmq_channel.default_exchange.publish(
                Message(
                    body=message_body.encode(),
                    delivery_mode=DeliveryMode.PERSISTENT,
                ),
                routing_key=QUEUE_NAME,
            )
            
            logger.info(f"消息已入队 (user_id={user_id})", extra={"event": "message_queued", "user_id": user_id})
            
            return {"status": "queued"}
        
        return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8889)
