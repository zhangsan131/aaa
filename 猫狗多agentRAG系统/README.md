# Fauna AI - 猫狗知识百科（多 Agent RAG 系统）

基于 LangGraph 多 Agent 架构的猫狗知识问答系统，采用 RAG（检索增强生成）技术，结合多路召回与重排序，提供精准的宠物知识问答服务。

---

## 项目结构

```
猫狗多agentRAG系统/
├── 前端/                        # Web 前端
│   ├── index.html              # 主聊天页面
│   ├── login.html              # 登录/注册页面
│   ├── admin.html              # 管理员后台面板
│   ├── css/
│   │   └── style.css           # 深色主题样式
│   └── js/
│       ├── app.js              # 前端核心逻辑（登录/注册/聊天/历史管理）
│       └── admin.js            # 管理员面板逻辑（用户管理/统计）
├── 后端/                        # Python 后端
│   ├── server_langgraph.py     # 主服务：FastAPI + LangGraph 多 Agent 架构
│   ├── import_books.py         # 知识库构建：JSON->向量+BM25 索引
│   ├── organized_book_pipeline.py  # 书籍预处理：PDF/EPUB/TXT -> 结构化 JSON
│   ├── ragas_evaluator.py      # RAG 评测脚本（RAGAS 指标 + 语义相似度）
│   └── knowledge_cache.pkl/vector_index.pkl/bm25_index.pkl  # 知识库缓存
└── README.md
```

## 核心功能

- **多 Agent 架构**：基于 LangGraph 的状态图，支持 9 个子 Agent 自动路由
- **混合检索**：多路召回（原查询 + 查询改写 + HyDE）+ RRF 融合 + CrossEncoder 重排序
- **知识问答**：基于宠物书籍（驯养、行为、健康、品种等）的专业知识回答
- **用户系统**：注册/登录、会话管理、历史记录、管理员面板
- **流式输出**：SSE（Server-Sent Events）实时流式返回回答
- **可观测性**：Prometheus 指标收集 + 健康检查端点

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI（Python） |
| AI 框架 | LangChain + LangGraph |
| 大语言模型 | DeepSeek API / Ollama（Qwen2.5 7B） |
| 向量检索 | FAISS + SentenceTransformer（text2vec-base-chinese） |
| 词法检索 | BM25（rank_bm25） |
| 重排序 | BGE-Reranker-v2-m3（CrossEncoder） |
| 数据库 | SQLite（用户/会话/消息） |
| 前端 | 原生 HTML + CSS + JavaScript |
| 监控 | Prometheus Client |

## 快速开始

### 环境要求

- Python 3.10+
- Node.js（可选，前端为静态页面）

### 安装依赖

```bash
cd 后端
pip install -r requirements.txt  # 见下方依赖清单
```

### 配置环境变量

创建 `.env` 文件：

```
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_MODEL=deepseek-v4-flash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

### 构建知识库

将书籍 JSON 文件放入 `解析文档/` 目录，然后运行：

```bash
cd 后端
python import_books.py
```

### 启动服务

```bash
cd 后端
python server_langgraph.py
# 服务启动在 http://0.0.0.0:8000
```

### 访问前端

直接用浏览器打开 `前端/index.html` 即可。

## 多 Agent 架构

系统通过 LangGraph 状态图编排 Agent 工作流：

```
用户消息 -> 主 Agent（路由） -> 查询改写 -> 知识检索 -> 子 Agent 回答 -> 审核 -> 输出
```

| Agent | 功能 |
|-------|------|
| **main** | 主调度，根据意图路由到子 Agent |
| **knowledge** | 知识问答（猫狗百科知识） |
| **pet** | 宠物医疗咨询 |
| **law** | 宠物相关法律咨询 |
| **image** | 图片生成（开发中） |
| **video** | 视频生成（开发中） |
| **identify** | 动物识别 |
| **story** | 故事绘本生成（开发中） |
| **chat** | 闲聊对话（本地 Ollama） |

## RAG 检索流程

1. **查询改写**：多路并行扩写（原查询、实体扩写、意图扩写、HyDE 假设文档）
2. **多路召回**：向量检索（FAISS）+ 词法检索（BM25）
3. **RRF 融合**：基于排名分数的加权融合
4. **CrossEncoder 重排序**：BGE-Reranker 精排
5. **事实面覆盖**：按知识点分组保证多样性
6. **上下文构建**：邻居窗口 + 父子层级补全

## API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/auth/login` | POST | 用户登录 |
| `/auth/register` | POST | 用户注册 |
| `/auth/logout` | POST | 退出登录 |
| `/chat` | POST | 聊天（SSE 流式） |
| `/history` | GET | 获取历史记录 |
| `/history/{session_id}` | DELETE | 删除会话历史 |
| `/health` | GET | 健康检查 |
| `/metrics` | GET | Prometheus 指标 |
| `/tags` | GET | 知识标签列表 |

## 评测

使用 RAGAS 框架进行评测：

```bash
cd 后端
python ragas_evaluator.py --input eval_data/qa.jsonl --output-dir eval_results/run_001
```

评测指标包括：Faithfulness、Answer Relevancy、Context Precision、Context Recall。
