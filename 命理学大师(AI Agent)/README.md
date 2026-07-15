# 命理学大师（AI Agent）- QQ 机器人

基于大语言模型的算命先生角色扮演 QQ 机器人，支持八字排盘、摇卦占卜、周公解梦、每日运势、八字合婚等命理服务，集成 RAG 知识检索、多模态处理、隐私保护和可观测性。

---

## 项目结构

```
命理学大师(AI Agent)/
├── server.py              # 主服务：FastAPI + LangChain Agent + RabbitMQ 消费者
├── Mytools.py             # 工具函数集（八字排盘/占卜/解梦/运势/合婚/RAG检索）
├── add_knowledge.py       # 命理知识库入库脚本（Qdrant 向量数据库）
├── data/                  # 命理知识文档存放目录
├── local_qdrand/          # Qdrant 本地向量数据库
├── voice_files/           # TTS 语音文件缓存
└── README.md
```

## 核心功能

- **角色扮演**：扮演"李疯子"——一位精通阴阳五行的 60 岁算命先生，使用繁体中文对话
- **八字排盘**：根据姓名、性别、出生年月日时进行八字排盘（调用外部 API + LLM 结构化分析）
- **摇卦占卜**：在线摇卦抽签
- **周公解梦**：根据梦境内容提取关键词进行解梦
- **每日运势**：基于真实农历黄历数据（cnlunar），包含干支、纳音、生肖冲煞、宜忌、运势签文
- **八字合婚**：双人八字排盘 + 生肖/纳音/天干地支综合评分
- **多轮意图追踪**：Slot Filling 机制，自动合并用户分次提供的信息
- **情绪感知**：LLM 识别用户情绪，自动调整回复语气风格
- **语音合成**：Azure TTS 语音输出 + NapCat QQ 机器人语音发送
- **用户反馈闭环**：满意/不满意反馈机制，自动反思改进

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI（Python） |
| AI 框架 | LangChain（AgentExecutor） |
| 大语言模型 | Qwen2.5 7B（Ollama 本地部署） |
| 向量数据库 | Qdrant（本地模式） |
| 嵌入模型 | paraphrase-multilingual-mpnet-base-v2 |
| 重排序 | CrossEncoder（ms-marco-MiniLM-L-6-v2） |
| 消息队列 | RabbitMQ（aio-pika） |
| 缓存 | Redis（聊天历史/意图追踪） |
| QQ 机器人 | NapCat（HTTP API） |
| 文本转语音 | Azure Cognitive Services TTS |
| 可观测性 | Prometheus + Jaeger + 结构化日志（Logstash） |
| 数据治理 | pii-guard（隐私脱敏）+ 差分隐私 + HyDE |

## 快速开始

### 环境要求

- Python 3.10+
- Ollama（运行 Qwen2.5 7B 模型）
- RabbitMQ
- Redis
- NapCat QQ 机器人框架

### 安装依赖

```bash
pip install -r requirements.txt  # 见下方依赖清单
```

### 启动依赖服务

```bash
# 启动 Ollama
ollama run qwen2.5:7b

# 启动 RabbitMQ
rabbitmq-server

# 启动 Redis
redis-server
```

### 构建命理知识库

将命理文档（.txt/.md/.json）放入 `data/` 目录，然后运行：

```bash
python add_knowledge.py
```

### 启动服务

```bash
python server.py
# 服务启动在 http://0.0.0.0:8889
```

## 命理工具说明

| 工具 | 触发条件 | 所需信息 |
|------|----------|----------|
| **bazi_cesuan** | 用户说"八字"、"排盘"、"生辰八字" | 姓名、性别、出生年月日时 |
| **yaoyigua** | 用户说"占卜"、"摇卦"、"抽签" | 无需参数 |
| **jiemeng** | 用户说"解梦"、"做梦" | 梦境内容描述 |
| **meiri_yunshi** | 用户说"今日运势"、"黄历"、"抽签"、"求签" | 无需参数 |
| **bazi_hehun** | 用户说"合婚"、"八字合婚"、"婚配" | 双方姓名、性别、出生年月日时 |
| **search** | 需要实时信息或未知概念 | 搜索关键词 |
| **gett_info_from_local_db** | 2024年运势或龙年运势相关问题 | 自然语言查询 |

## 系统架构

```
QQ 消息
  │
  ▼
NapCat HTTP API
  │
  ▼
FastAPI /cqhttp 端点
  │
  ▼
RabbitMQ 消息队列（异步解耦）
  │
  ▼
消费者进程（并发控制）
  │
  ├─ 安全护栏（黑名单检测）
  ├─ 意图追踪（Slot Filling）
  ├─ 记忆压缩（Redis 历史摘要）
  ├─ 情绪识别（LLM 情绪分类）
  ├─ RAG Fusion（HyDE + 多路检索 + CrossEncoder 重排序）
  ├─ 关键词直路由（运势/合婚/八字）
  ├─ LangChain Agent 执行（工具调用 + LLM 推理）
  ├─ 输出脱敏
  └─ 语音合成（Azure TTS，后台线程）
       │
       ▼
    发送回复（文本 + 语音）
```

## 数据治理特性

- **数据清洗**：去除 HTML 标签、特殊字符，统一格式
- **隐私保护**：使用 pii-guard 库检测并脱敏姓名、地址、手机号、身份证等敏感信息
- **差分隐私**：为统计数据添加拉普拉斯/高斯噪声，保护用户隐私
- **RAG 优化**：HyDE（假设文档增强检索）+ CrossEncoder 重排序
- **多模态处理**：CLIP 模型支持图文跨模态检索

## 可观测性

- **Prometheus 指标**：查询计数、延迟直方图、情绪分布、TTS 状态
- **Jaeger 链路追踪**：OpenTelemetry 集成，完整追踪用户请求链路
- **结构化日志**：JSON 格式日志，支持 Logstash 采集

## API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 健康检查 |
| `/cqhttp` | POST | QQ 消息接收端点（NapCat WebHook） |
| `/add_urls` | POST | 添加网页内容到知识库 |
| `/add_pdfs` | POST | 添加 PDF 到知识库（预留） |
| `/add_texts` | POST | 添加文本到知识库（预留） |
| `/metrics` | GET | Prometheus 指标端点 |
