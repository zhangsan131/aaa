"""
本地知识库数据入库脚本
用法：把要添加的文档放到 data/ 目录（支持 .txt .md .json），然后运行：
    python add_knowledge.py
"""
import os
# 使用 HuggingFace 国内镜像，无需代理
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import glob
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ==================== 配置 ====================
DB_PATH = "./local_qdrand"
COLLECTION_NAME = "yunshi_2024"
DATA_DIR = "./data"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# 章节标题匹配正则（覆盖常见命理书籍格式）
CHAPTER_PATTERNS = [
    r'^【.+?】$',                    # 【渊海子平】
    r'^《.+?》$',                    # 《论日为主》
    r'^[一二三四五六七八九十百千]+[、．.\s]+.+',  # 一、天道  一．论十干十二支
    r'^论\s*.+',                     # 论木、论五行生成
    r'^.+总论$',                     # 五行总论、三春甲木总论
    r'^通神论$',
    r'^六亲论$',
    r'^原\s*.+',                     # 原造化之始
    r'^基础$',
    r'^目录$',
    r'^序$',
    r'^序\s*言$',
    r'^提\s*要$',
    r'^钦定.+',                      # 钦定四库全书
    r'^天干\s*.+',
    r'^地支\s*.+',
    r'^干支\s*.+',
    r'^五行\s*.+',
    r'^十干\s*.+',
    r'^十二支\s*.+',
    r'^书籍名称[：:].+',
]


def is_chapter_title(line: str) -> bool:
    """判断一行是否为章节标题"""
    line = line.strip()
    if not line or len(line) > 60:
        return False
    for p in CHAPTER_PATTERNS:
        if re.match(p, line):
            return True
    return False


def split_by_chapters(text: str):
    """按章节标题切分文本，返回章节列表"""
    lines = text.split('\n')
    chapters = []
    current = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if is_chapter_title(stripped):
            if current:
                chapters.append('\n'.join(current))
            current = [stripped]
        else:
            current.append(stripped)

    if current:
        chapters.append('\n'.join(current))

    return chapters


def load_documents(data_dir: str):
    """读取目录下所有文档并按章节切分"""
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )

    for ext in ["*.txt", "*.md", "*.json"]:
        for filepath in glob.glob(os.path.join(data_dir, ext)):
            print(f"读取文件: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                if filepath.endswith(".json"):
                    data = json.load(f)
                    texts = []
                    if isinstance(data, list):
                        for item in data:
                            texts.append(item if isinstance(item, str) else item.get("content", ""))
                    elif isinstance(data, dict):
                        texts.append(data.get("content", ""))
                    raw_text = "\n".join(t for t in texts if t)
                else:
                    raw_text = f.read()

            # 先按章节切分
            chapters = split_by_chapters(raw_text)
            print(f"  识别到 {len(chapters)} 个章节")

            for ch in chapters:
                ch = ch.strip()
                if len(ch) < 20:
                    continue
                # 超长章节再切分，但保留标题前缀
                if len(ch) > 1200:
                    sub_chunks = text_splitter.split_text(ch)
                    # 第一个子 chunk 自带标题，后面的补标题
                    title_line = ch.split('\n')[0] if is_chapter_title(ch.split('\n')[0]) else ""
                    for i, sub in enumerate(sub_chunks):
                        if i > 0 and title_line:
                            sub = title_line + "\n" + sub
                        all_chunks.append({"text": sub, "source": os.path.basename(filepath)})
                else:
                    all_chunks.append({"text": ch, "source": os.path.basename(filepath)})

    return all_chunks


# ==================== 主流程 ====================
if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"已创建数据目录: {DATA_DIR}")
        print("请把要入库的 .txt / .md / .json 文件放到该目录，然后重新运行脚本。")
        exit(0)

    print("正在加载 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"正在连接 Qdrant 本地库: {DB_PATH}")
    client = QdrantClient(path=DB_PATH)

    raw_chunks = load_documents(DATA_DIR)
    if not raw_chunks:
        print(f"{DATA_DIR} 目录下没有找到可入库的内容。")
        exit(0)

    print(f"共读取 {len(raw_chunks)} 个章节/片段，开始生成向量...")

    points = []
    for idx, item in enumerate(raw_chunks):
        chunk = item["text"]
        vector = embeddings.embed_query(chunk)
        points.append(
            PointStruct(
                id=idx,
                vector=vector,
                payload={
                    "page_content": chunk,
                    "source": item["source"],
                },
            )
        )
        if (idx + 1) % 10 == 0:
            print(f"  已处理 {idx + 1}/{len(raw_chunks)}...")

    # 重建集合（自动清空旧数据）
    print(f"正在重建集合 {COLLECTION_NAME}（维度: 768, 距离: Cosine）...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    print(f"正在写入 Qdrant（共 {len(points)} 条）...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    print("[OK] 数据入库完成！")
    print(f"   集合: {COLLECTION_NAME}")
    print(f"   文档数: {len(points)}")
