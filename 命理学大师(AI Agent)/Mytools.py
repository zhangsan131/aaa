from langchain.agents import tool
from langchain_community.utilities import SerpAPIWrapper
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
import aiohttp
import asyncio
import json
import os
import glob

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["SERPAPI_API_KEY"] = os.environ.get("SERPAPI_API_KEY", "")

def _get_local_model_path(model_name: str) -> str:
    """自动查找 HuggingFace 本地缓存路径，避免联网"""
    safe_name = model_name.replace("/", "--")
    base = os.path.expanduser("~/.cache/huggingface/hub")
    pattern = os.path.join(base, f"models--{safe_name}", "snapshots", "*")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return model_name

@tool
async def search(query:str):
    """只有需要了解实时信息或不知道的事情时才需要使用该工具"""
    serp = SerpAPIWrapper()
    results = serp.run(query)
    print("实时搜索结果:",results)
    return results

# 全局变量，只创建一次 Qdrant 客户端和嵌入模型
_qdrant_client = None
_embeddings_model = None

def get_qdrant_components():
    """获取或创建 Qdrant 客户端（单例模式）"""
    global _qdrant_client, _embeddings_model
    if _qdrant_client is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, "local_qdrand")
        
        # 创建嵌入模型（使用 768 维模型，与存储的数据一致）
        model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        local_path = _get_local_model_path(model_name)
        _embeddings_model = HuggingFaceEmbeddings(
            model_name=local_path,
            model_kwargs={"local_files_only": True}
        )
        
        # 创建 Qdrant 客户端
        try:
            _qdrant_client = QdrantClient(path=db_path)
        except Exception as e:
            print(f"创建 Qdrant 客户端失败: {e}")
            _qdrant_client = None
    
    return _qdrant_client, _embeddings_model

@tool
async def gett_info_from_local_db(query:str):
    """只有回答与2024年运势或者龙年运势相关的问题的时候，会使用这个工具"""
    try:
        client, embeddings = get_qdrant_components()
        
        if client is None:
            return "本地知识库为空，请先添加文档。"
        
        # 检查集合是否存在
        try:
            collections = client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if "yunshi_2024" not in collection_names:
                return "本地知识库为空，请先添加文档。"
        except Exception as e:
            print(f"检查集合失败: {e}")
            return "本地知识库为空，请先添加文档。"
        
        # 第一步：向量检索获取候选文档（扩大召回范围）
        query_vector = embeddings.embed_query(query)
        
        search_result = client.query_points(
            collection_name="yunshi_2024",
            query=query_vector,
            limit=10,  # 先召回更多候选
        )
        
        # 提取文本内容和分数
        candidates = []
        for point in search_result.points:
            payload = point.payload
            if payload and "page_content" in payload:
                candidates.append((payload["page_content"], point.score))
        
        if not candidates:
            return "未找到相关信息。"
        
        # 第二步：CrossEncoder 重排序（精细排序）
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            pairs = [(query, doc) for doc, _ in candidates]
            rerank_scores = reranker.predict(pairs)
            
            # 按重排序分数排序
            reranked = sorted(
                zip([doc for doc, _ in candidates], rerank_scores),
                key=lambda x: x[1],
                reverse=True
            )[:3]  # 取Top 3
            
            results = [doc for doc, _ in reranked]
            print(f"[混合检索] 重排序完成，返回 {len(results)} 条结果")
        except Exception as e:
            # 重排序失败时降级为原始向量检索结果
            print(f"[混合检索] 重排序失败，使用原始结果: {e}")
            results = [doc for doc, _ in candidates[:3]]
        
        if not results:
            return "未找到相关信息。"
        
        return "\n\n".join(results)
    
    except Exception as e:
        error_msg = str(e)
        print(f"查询知识库失败: {error_msg}")
        if "Collection" in error_msg or "not found" in error_msg:
            return "本地知识库为空，请先添加文档。"
        raise e

@tool
async def bazi_cesuan(query:str):
    """只有做八字排盘的时候才会使用这个工具，需要输入姓名和出生年月日，如果用户没有输入姓名和出生年月日时不可用"""
    url = f"https://api.yuanfenju.com/index.php/v1/Bazi/cesuan"
    prompt = ChatPromptTemplate.from_template(
        """你是一个参数查询助手，根据用户输入内容找出相关的参数并按json格式返回。
        JSON字段如下：
        - "api_key":"{api_key}"
        - "name":"姓名"
        - "sex":"性别，0表示男，1表示女，根据姓名判断"
        - "type":"日历类型，0农历，1公历，默认1"
        - "year":"出生年份 例：1998"
        - "month":"出生月份 例 8"
        - "day":"出生日期，例：8"
        - "hours":"出生小时 例 14"
        - "minute":"0"
        如果没有找到相关参数，则需要提醒用户告诉你这些内容，只返回数据结构，不要有其他的评论，用户输入:{query}"""
    )
    parser = JsonOutputParser()
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen2.5:7b",
        temperature=0,
    ) | parser
    result = await chain.ainvoke({"query": query, "api_key": os.environ.get("YUANFENJU_API_KEY", "")})
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=result) as response:
            if response.status == 200:
                try:
                    json_data = await response.json()
                    bazi_str = json_data['data']['bazi_info']['bazi']

                    # ===== Structured Output：使用 JSON Mode 生成结构化分析 =====
                    analysis_prompt = ChatPromptTemplate.from_template(
                        """你是一位命理分析专家。根据以下八字信息，输出 JSON 格式的分析结果。
八字：{bazi}
请严格输出以下字段的 JSON，不要包含其他内容：
{{
  "year_pillar": "年柱，如甲子",
  "month_pillar": "月柱",
  "day_pillar": "日柱",
  "hour_pillar": "时柱",
  "day_master": "日主五行，如丙火",
  "five_elements": "五行分布描述，如木2火1土1金1水1",
  "brief_analysis": "简要命理分析，60字以内"
}}
只输出合法 JSON。"""
                    )
                    analysis_parser = JsonOutputParser()
                    analysis_chain = analysis_prompt | ChatOpenAI(
                        base_url="http://localhost:11434/v1",
                        api_key="ollama",
                        model="qwen2.5:7b",
                        temperature=0,
                    ) | analysis_parser

                    try:
                        analysis = await analysis_chain.ainvoke({"bazi": bazi_str})
                        returnstring = (
                            f"【八字排盤結果】\n"
                            f"年柱：{analysis['year_pillar']}\n"
                            f"月柱：{analysis['month_pillar']}\n"
                            f"日柱：{analysis['day_pillar']}\n"
                            f"時柱：{analysis['hour_pillar']}\n"
                            f"日主：{analysis['day_master']}\n"
                            f"五行：{analysis['five_elements']}\n"
                            f"簡批：{analysis['brief_analysis']}"
                        )
                        return returnstring
                    except Exception:
                        # 降級：JSON 解析失敗時返回原始八字字符串
                        return f"八字為{bazi_str}"
                except Exception as e:
                    return "八字查询失败，可能是你忘记询问用户姓名和出生年月日时了"
            else:
                return "技术错误,请告诉用户稍后重试"

@tool
async def yaoyigua():
    """只有用户想要占卜抽签的时候才会使用这个工具。"""
    api_key = os.environ.get("YUANFENJU_API_KEY", "")
    url = f"https://api.yuanfenju.com/index.php/v1/Zhanbu/yaogua"
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data={"api_key": api_key}) as response:
            if response.status == 200:
                returnstring = await response.json()
                # 检查数据结构
                if isinstance(returnstring, list) and len(returnstring) > 0:
                    # 如果是列表，取第一个元素
                    data = returnstring[0]
                else:
                    data = returnstring
                
                # 尝试获取 image
                if isinstance(data, dict) and 'data' in data and isinstance(data['data'], dict):
                    image = data['data'].get('image', '无法获取图片')
                else:
                    image = str(data)
                return image
            else:
                return "技术错误,请告诉用户稍后重试"

@tool
async def jiemeng(query:str):
    """只有用户想要解梦的时候才会使用这个工具,需要输入用户梦境的内容，如果缺少用户梦境的内容则不可用。"""
    api_key = os.environ.get("YUANFENJU_API_KEY", "")
    url = f"https://api.yuanfenju.com/index.php/v1/Gongju/zhougong"
    
    # 使用 LLM 提取关键词
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen2.5:7b",
        temperature=0,
    )
    
    prompt = ChatPromptTemplate.from_template("根据内容提取1个关键词，只返回关键词（必须是中文，不要有任何标点符号），内容为:{query}")
    chain = prompt | llm
    keyword = await chain.ainvoke({"query": query})
    keyword_text = keyword.content.strip()
    print("提取的关键词:", keyword_text)
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data={"api_key": api_key, "title_zhougong": keyword_text}) as response:
            if response.status == 200:
                print("====返回数据====")
                returnstring = await response.json()
                print(returnstring)
                return returnstring
            else:
                return "技术错误，请告诉用户稍后再试。"


@tool
async def meiri_yunshi(query: str):
    """当用户想要查看今日运势、每日运势、抽签、求签、查黄历、看老黄历时使用该工具。
    无需用户提供额外参数，自动根据当前日期生成运势。
    """
    import datetime
    import random
    import cnlunar

    now = datetime.datetime.now()
    date_str = now.strftime("%Y年%m月%d日")

    # 获取真实农历黄历数据
    lunar = cnlunar.Lunar(now)
    lunar_date = f"{lunar.lunarYearCn}{lunar.lunarMonthCn}{lunar.lunarDayCn}"

    # 运势签文数据（30签）
    QIANWEN = [
        {"qian": "第一签", "qianyu": "上上签", "jie": "诸事顺遂，心想事成。今日吉星高照，宜积极进取，必有佳音。"},
        {"qian": "第二签", "qianyu": "上吉签", "jie": "贵人相助，逢凶化吉。遇事莫急，自有转机，静待时机。"},
        {"qian": "第三签", "qianyu": "中吉签", "jie": "功到自然成，不必强求。循序渐进，终有所获。"},
        {"qian": "第四签", "qianyu": "中平签", "jie": "平淡是真，无大喜亦无大忧。守成为上，静待花开。"},
        {"qian": "第五签", "qianyu": "上上签", "jie": "天时地利人和，万事亨通。求财得财，求名得名。"},
        {"qian": "第六签", "qianyu": "上吉签", "jie": "云开见月明，困境将过。保持信心，好运将至。"},
        {"qian": "第七签", "qianyu": "中吉签", "jie": "小有波折，终能如愿。谨慎行事，事半功倍。"},
        {"qian": "第八签", "qianyu": "中平签", "jie": "凡事三思而后行，莫贪小利。稳扎稳打，方为上策。"},
        {"qian": "第九签", "qianyu": "上吉签", "jie": "柳暗花明又一村，绝处逢生。坚持信念，必有回响。"},
        {"qian": "第十签", "qianyu": "上上签", "jie": "鸿运当头，百事顺遂。今日所求，皆能如意。"},
        {"qian": "第十一签", "qianyu": "中吉签", "jie": "种瓜得瓜，种豆得豆。付出终有回报，耐心等待。"},
        {"qian": "第十二签", "qianyu": "中平签", "jie": "顺其自然，随遇而安。不必过于执着，随遇而安即是福。"},
        {"qian": "第十三签", "qianyu": "上吉签", "jie": "山重水复疑无路，柳暗花明又一村。困境即将过去。"},
        {"qian": "第十四签", "qianyu": "中吉签", "jie": "好事多磨，虽有小阻，终能达成。保持耐心。"},
        {"qian": "第十五签", "qianyu": "上上签", "jie": "春风得意马蹄疾，一日看尽长安花。今日大吉。"},
        {"qian": "第十六签", "qianyu": "中平签", "jie": "静观其变，不宜妄动。时机未到，耐心等待。"},
        {"qian": "第十七签", "qianyu": "上吉签", "jie": "天道酬勤，努力终有回报。今日宜积极行动。"},
        {"qian": "第十八签", "qianyu": "中吉签", "jie": "知足常乐，适可而止。贪心不足蛇吞象，见好就收。"},
        {"qian": "第十九签", "qianyu": "上上签", "jie": "紫气东来，福星高照。今日诸事皆宜，大胆前行。"},
        {"qian": "第二十签", "qianyu": "中平签", "jie": "平平淡淡才是真。无需刻意追求，一切随缘。"},
        {"qian": "第二十一签", "qianyu": "上吉签", "jie": "否极泰来，好运将至。之前的努力即将开花结果。"},
        {"qian": "第二十二签", "qianyu": "中吉签", "jie": "稳中求进，步步为营。不宜冒进，稳扎稳打为上。"},
        {"qian": "第二十三签", "qianyu": "上上签", "jie": "天时地利人和，三者兼备。今日所为，事半功倍。"},
        {"qian": "第二十四签", "qianyu": "中平签", "jie": "静水流深，大智若愚。低调行事，自有收获。"},
        {"qian": "第二十五签", "qianyu": "上吉签", "jie": "苦尽甘来，时来运转。之前的苦难即将结束。"},
        {"qian": "第二十六签", "qianyu": "中吉签", "jie": "知足常乐，随遇而安。珍惜眼前，即是幸福。"},
        {"qian": "第二十七签", "qianyu": "上上签", "jie": "大鹏一日同风起，扶摇直上九万里。今日大吉大利。"},
        {"qian": "第二十八签", "qianyu": "中平签", "jie": "欲速则不达，见小利则大事不成。耐心为上。"},
        {"qian": "第二十九签", "qianyu": "上吉签", "jie": "功夫不负有心人，铁杵磨成针。坚持就是胜利。"},
        {"qian": "第三十签", "qianyu": "上上签", "jie": "万事如意，百无禁忌。今日所想，皆能成真。"},
    ]

    # 用日期做种子，保证同一天签文一致
    random.seed(now.strftime("%Y%m%d"))
    qian = random.choice(QIANWEN)

    # 宜忌取前几条
    yi = lunar.goodThing[:5] if lunar.goodThing else ["诸事不宜"]
    ji = lunar.badThing[:5] if lunar.badThing else ["诸事无忌"]

    # 吉神方位从胎神描述中提取
    fetal = lunar.get_fetalGod()
    direction = "东南"
    for d in ["正东", "正南", "正西", "正北", "东南", "东北", "西南", "西北"]:
        if d in fetal:
            direction = d
            break

    result = f"""【{date_str} 每日运势】
农历：{lunar_date}
干支：{lunar.year8Char}年 {lunar.month8Char}月 {lunar.day8Char}日
纳音：{lunar.get_nayin()}
生肖：{lunar.chineseYearZodiac}
冲煞：{lunar.chineseZodiacClash}
十二神：{lunar.today12DayGod}
二十八宿：{lunar.today28Star}
彭祖百忌：{lunar.get_pengTaboo()}

🎯 运势签：{qian['qian']} {qian['qianyu']}
📜 解签：{qian['jie']}

📅 今日黄历
✅ 宜：{'、'.join(yi)}
❌ 忌：{'、'.join(ji)}
🧭 吉神方位：{direction}

祝君今日顺遂！"""

    return result


@tool
async def bazi_hehun(query: str):
    """只有用户想要八字合婚、合八字、看两人婚配、看两人八字合不合时才会使用这个工具。
    参数 query：用户输入的完整自然语言字符串，需包含两个人的姓名、性别和出生年月日时。
    即使信息不全，也直接传入原始 query，本工具内部会自动提取参数并处理缺失信息。
    """
    # 使用 LLM 提取两个人的信息
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_template(
        """你是一个参数提取助手，从用户输入中提取两个人的八字合婚参数，按JSON格式返回。
JSON字段如下：
- "name1":"男方姓名"
- "sex1":"男方性别，0表示男，1表示女。若用户未明确说明，根据姓名推断：常见男名（如伟、强、军、涛、磊、超、勇、杰、浩、宇等）填0，常见女名（如芳、娜、丽、敏、静、秀、玲、燕、慧、娟等）填1"
- "year1":"男方出生年（4位数字）"
- "month1":"男方出生月"
- "day1":"男方出生日"
- "hours1":"男方出生时辰（0-23点）"
- "name2":"女方姓名"
- "sex2":"女方性别，0表示男，1表示女。推断规则同上"
- "year2":"女方出生年（4位数字）"
- "month2":"女方出生月"
- "day2":"女方出生日"
- "hours2":"女方出生时辰（0-23点）"
如果信息不全，返回 {{"error": "缺少xx信息"}}。
    用户输入：{query}
    {format_instructions}"""
    )
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen2.5:7b",
        temperature=0,
    )

    chain = prompt | llm | parser
    try:
        params = await chain.ainvoke({"query": query})
    except Exception:
        return "参数提取失败，请提供双方的姓名、性别和出生年月日时。"

    # 解析结果
    if "error" in params:
        return params["error"]

    # 调用八字排盘获取两人八字（并行执行，节省耗时）
    query1 = f"{params['name1']}，性别{params['sex1']}，{params['year1']}年{params['month1']}月{params['day1']}日{params['hours1']}时"
    query2 = f"{params['name2']}，性别{params['sex2']}，{params['year2']}年{params['month2']}月{params['day2']}日{params['hours2']}时"

    bazi1, bazi2 = await asyncio.gather(
        bazi_cesuan.ainvoke({"query": query1}),
        bazi_cesuan.ainvoke({"query": query2})
    )

    # 解析八字中的年柱和日柱
    def parse_bazi(bazi_str):
        parts = bazi_str.replace("八字为", "").strip().split()
        if len(parts) >= 4:
            return {
                "year": parts[0],
                "month": parts[1],
                "day": parts[2],
                "hour": parts[3],
            }
        return None

    info1 = parse_bazi(bazi1)
    info2 = parse_bazi(bazi2)

    if not info1 or not info2:
        return "八字排盘失败，请检查出生信息是否正确。"

    # 合婚规则分析
    def get_shengxiao(ganzhi):
        """从年柱获取生肖"""
        zhi = ganzhi[1]
        shengxiao_map = {
            "子": "鼠", "丑": "牛", "寅": "虎", "卯": "兔",
            "辰": "龙", "巳": "蛇", "午": "马", "未": "羊",
            "申": "猴", "酉": "鸡", "戌": "狗", "亥": "猪"
        }
        return shengxiao_map.get(zhi, "未知")

    def get_nayin(ganzhi):
        """获取纳音"""
        nayin_map = {
            "甲子": "海中金", "乙丑": "海中金", "丙寅": "炉中火", "丁卯": "炉中火",
            "戊辰": "大林木", "己巳": "大林木", "庚午": "路旁土", "辛未": "路旁土",
            "壬申": "剑锋金", "癸酉": "剑锋金", "甲戌": "山头火", "乙亥": "山头火",
            "丙子": "涧下水", "丁丑": "涧下水", "戊寅": "城头土", "己卯": "城头土",
            "庚辰": "白蜡金", "辛巳": "白蜡金", "壬午": "杨柳木", "癸未": "杨柳木",
            "甲申": "泉中水", "乙酉": "泉中水", "丙戌": "屋上土", "丁亥": "屋上土",
            "戊子": "霹雳火", "己丑": "霹雳火", "庚寅": "松柏木", "辛卯": "松柏木",
            "壬辰": "长流水", "癸巳": "长流水", "甲午": "砂中金", "乙未": "砂中金",
            "丙申": "山下火", "丁酉": "山下火", "戊戌": "平地木", "己亥": "平地木",
            "庚子": "壁上土", "辛丑": "壁上土", "壬寅": "金箔金", "癸卯": "金箔金",
            "甲辰": "覆灯火", "乙巳": "覆灯火", "丙午": "天河水", "丁未": "天河水",
            "戊申": "大驿土", "己酉": "大驿土", "庚戌": "钗钏金", "辛亥": "钗钏金",
            "壬子": "桑柘木", "癸丑": "桑柘木", "甲寅": "大溪水", "乙卯": "大溪水",
            "丙辰": "沙中土", "丁巳": "沙中土", "戊午": "天上火", "己未": "天上火",
            "庚申": "石榴木", "辛酉": "石榴木", "壬戌": "大海水", "癸亥": "大海水"
        }
        return nayin_map.get(ganzhi, "未知")

    def analyze_hehun(info1, info2):
        """合婚分析"""
        sx1 = get_shengxiao(info1["year"])
        sx2 = get_shengxiao(info2["year"])

        # 生肖配对
        liuhe = [("鼠", "牛"), ("虎", "猪"), ("兔", "狗"), ("龙", "鸡"), ("蛇", "猴"), ("马", "羊")]
        sanhe = [("猴", "鼠", "龙"), ("虎", "马", "狗"), ("蛇", "鸡", "牛"), ("猪", "兔", "羊")]
        xiangchong = [("鼠", "马"), ("牛", "羊"), ("虎", "猴"), ("兔", "鸡"), ("龙", "狗"), ("蛇", "猪")]

        sx_score = 60
        sx_comment = "平"

        for p in liuhe:
            if sx1 in p and sx2 in p and sx1 != sx2:
                sx_score = 95
                sx_comment = "六合，上上婚配"
                break

        if sx_score < 95:
            for p in sanhe:
                if sx1 in p and sx2 in p:
                    sx_score = 90
                    sx_comment = "三合，上等婚配"
                    break

        if sx_score < 90:
            for p in xiangchong:
                if sx1 in p and sx2 in p:
                    sx_score = 40
                    sx_comment = "相冲，需谨慎"
                    break

        # 日柱分析
        day1_gan = info1["day"][0]
        day1_zhi = info1["day"][1]
        day2_gan = info2["day"][0]
        day2_zhi = info2["day"][1]

        # 天干五合
        wuhe = [("甲", "己"), ("乙", "庚"), ("丙", "辛"), ("丁", "壬"), ("戊", "癸")]
        tg_score = 60
        tg_comment = "平"
        for p in wuhe:
            if day1_gan in p and day2_gan in p and day1_gan != day2_gan:
                tg_score = 90
                tg_comment = "天干五合，夫妻同心"
                break

        # 地支六合
        dz_liuhe = [("子", "丑"), ("寅", "亥"), ("卯", "戌"), ("辰", "酉"), ("巳", "申"), ("午", "未")]
        dz_score = 60
        dz_comment = "平"
        for p in dz_liuhe:
            if day1_zhi in p and day2_zhi in p:
                dz_score = 90
                dz_comment = "地支六合，恩爱和睦"
                break

        # 纳音
        ny1 = get_nayin(info1["year"])
        ny2 = get_nayin(info2["year"])
        ny_wuxing1 = ny1[-1] if ny1 else "未知"
        ny_wuxing2 = ny2[-1] if ny2 else "未知"

        xiangsheng = [("金", "水"), ("水", "木"), ("木", "火"), ("火", "土"), ("土", "金")]
        xiangke = [("金", "木"), ("木", "土"), ("土", "水"), ("水", "火"), ("火", "金")]

        ny_score = 60
        ny_comment = "平"
        for p in xiangsheng:
            if (ny_wuxing1 == p[0] and ny_wuxing2 == p[1]) or (ny_wuxing1 == p[1] and ny_wuxing2 == p[0]):
                ny_score = 85
                ny_comment = "纳音相生，互助互补"
                break
        for p in xiangke:
            if (ny_wuxing1 == p[0] and ny_wuxing2 == p[1]) or (ny_wuxing1 == p[1] and ny_wuxing2 == p[0]):
                ny_score = 45
                ny_comment = "纳音相克，需多包容"
                break

        total_score = (sx_score + tg_score + dz_score + ny_score) // 4

        if total_score >= 90:
            level = "上上等婚"
        elif total_score >= 80:
            level = "上等婚"
        elif total_score >= 70:
            level = "中等婚"
        elif total_score >= 60:
            level = "中下等婚"
        else:
            level = "下等婚"

        return {
            "total_score": total_score,
            "level": level,
            "sx": f"{sx1} vs {sx2}：{sx_comment}",
            "tg": f"日柱天干：{tg_comment}",
            "dz": f"日柱地支：{dz_comment}",
            "ny": f"年柱纳音：{ny1} vs {ny2}，{ny_comment}"
        }

    result = analyze_hehun(info1, info2)

    advice = "两人缘分深厚，宜珍惜。"
    if result["total_score"] < 80:
        advice = "有缘分但需经营，互相包容。"
    if result["total_score"] < 60:
        advice = "缘分较浅，需多加努力。"

    return f"""【八字合婚分析】

男方：{params['name1']} ({info1['year']} {info1['month']} {info1['day']} {info1['hour']})
女方：{params['name2']} ({info2['year']} {info2['month']} {info2['day']} {info2['hour']})

📊 综合评分：{result['total_score']}分
💑 婚配等级：{result['level']}

🔹 {result['sx']}
🔹 {result['tg']}
🔹 {result['dz']}
🔹 {result['ny']}

💡 建议：{advice}
"""
