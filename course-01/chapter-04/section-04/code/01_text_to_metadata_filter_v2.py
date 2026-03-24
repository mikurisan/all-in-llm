import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.document_loaders import BiliBiliLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

# -----------------------------------------------------------------------------
# 日志与配置
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


@dataclass
class Config:
    video_urls: tuple[str, ...] = (
        "https://www.bilibili.com/video/BV1Bo4y1A7FU",
        "https://www.bilibili.com/video/BV1ug4y157xA",
        "https://www.bilibili.com/video/BV1yh411V7ge",
    )
    embed_model_name: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = os.getenv("MODEL_NAME", "")
    api_base: str = os.getenv("BASE_URL", "")
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    vector_dir: Path = Path("./chroma_store")
    queries: tuple[str, ...] = ("时间最短的视频", "播放量最高的视频")


# -----------------------------------------------------------------------------
# 1. 初始化视频数据
# -----------------------------------------------------------------------------
def load_bilibili_docs(cfg: Config):
    loader = BiliBiliLoader(video_urls=list(cfg.video_urls))
    docs = []

    for doc in loader.load():
        raw = doc.metadata
        doc.metadata = {
            "title": raw.get("title", "未知标题"),
            "author": raw.get("owner", {}).get("name", "未知作者"),
            "source": raw.get("bvid", "未知ID"),
            "view_count": raw.get("stat", {}).get("view", 0),
            "length": raw.get("duration", 0),
        }
        docs.append(doc)

    if not docs:
        raise RuntimeError("未加载到任何 B 站视频")
    logging.info("成功加载 %d 条视频数据", len(docs))
    return docs


# -----------------------------------------------------------------------------
# 2. 创建向量存储
# -----------------------------------------------------------------------------
def build_vector_store(cfg: Config, docs):
    cfg.vector_dir.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=cfg.embed_model_name)
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(cfg.vector_dir),
    )
    logging.info("向量库已构建并持久化至 %s", cfg.vector_dir)
    return store


# -----------------------------------------------------------------------------
# 3. 构造元信息描述
# -----------------------------------------------------------------------------
def build_metadata_schema() -> List[AttributeInfo]:
    return [
        AttributeInfo(name="title", description="视频标题（字符串）", type="string"),
        AttributeInfo(name="author", description="视频作者（字符串）", type="string"),
        AttributeInfo(name="view_count", description="视频观看次数（整数）", type="integer"),
        AttributeInfo(name="length", description="视频长度（整数）", type="integer"),
    ]


# -----------------------------------------------------------------------------
# 4. 初始化 LLM 客户端
# -----------------------------------------------------------------------------
def init_llm(cfg: Config) -> OpenAI:
    if not cfg.api_key:
        raise EnvironmentError("请设置环境变量 API_KEY")
    return OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)


# -----------------------------------------------------------------------------
# 5. 构造排序提示词
# -----------------------------------------------------------------------------
def build_instruction_prompt(query: str) -> str:
    return f"""
        你是一个智能助手，请将用户的问题转换成一个用于排序视频的JSON指令。

        你需要识别用户想要排序的字段和排序方向。
        - 排序字段必须是 'view_count' (观看次数) 或 'length' (时长) 之一。
        - 排序方向必须是 'asc' (升序) 或 'desc' (降序) 之一。

        例如:
        - '时间最短的视频' 或 '哪个视频时间最短' 应转换为 {{"sort_by": "length", "order": "asc"}}
        - '播放量最高的视频' 或 '哪个视频最火' 应转换为 {{"sort_by": "view_count", "order": "desc"}}

        请根据以下问题生成JSON指令:
        原始问题: "{query}"

        JSON指令:"""

# -----------------------------------------------------------------------------
# 6. 调用大模型生成排序指令
# -----------------------------------------------------------------------------
def generate_sort_instruction(client: OpenAI, cfg: Config, query: str) -> Dict[str, str]:
    prompt = build_instruction_prompt(query)
    response = client.chat.completions.create(
        model=cfg.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


# -----------------------------------------------------------------------------
# 7. 执行排序并输出结果
# -----------------------------------------------------------------------------
def display_top_video(docs, instruction: Dict[str, str]):
    sort_by = instruction.get("sort_by")
    order = instruction.get("order")
    if sort_by not in {"length", "view_count"} or order not in {"asc", "desc"}:
        logging.warning("生成的排序指令无效: %s", instruction)
        return

    reverse = order == "desc"
    sorted_docs = sorted(docs, key=lambda d: d.metadata.get(sort_by, 0), reverse=reverse)
    top = sorted_docs[0]

    print(f"标题: {top.metadata.get('title', '未知标题')}")
    print(f"作者: {top.metadata.get('author', '未知作者')}")
    print(f"观看次数: {top.metadata.get('view_count', '未知')}")
    print(f"时长: {top.metadata.get('length', '未知')}秒")
    print("=" * 50)


# -----------------------------------------------------------------------------
# 8. 任务入口
# -----------------------------------------------------------------------------
def main():
    cfg = Config()
    docs = load_bilibili_docs(cfg)
    build_metadata_schema()  # 演示保留：实际查询构造可使用
    build_vector_store(cfg, docs)  # 教学示例：展示如何落地向量库
    client = init_llm(cfg)

    for query in cfg.queries:
        print(f"\n--- 原始查询: '{query}' ---")
        try:
            instruction = generate_sort_instruction(client, cfg, query)
            print(f"--- 生成的排序指令: {instruction} ---")
            display_top_video(docs, instruction)
        except json.JSONDecodeError as err:
            logging.error("解析排序指令失败: %s", err)


if __name__ == "__main__":
    main()
