import logging
import os
from dataclasses import dataclass
from typing import List

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.document_loaders import BiliBiliLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


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
    model_name: str = os.getenv("MODEL_NAME", "")
    api_key: str = os.getenv("API_KEY", "")
    api_base: str = os.getenv("BASE_URL", "")
    embed_model: str = "BAAI/bge-small-zh-v1.5"
    video_urls: List[str] = (
        "https://www.bilibili.com/video/BV1Bo4y1A7FU",
        "https://www.bilibili.com/video/BV1ug4y157xA",
        "https://www.bilibili.com/video/BV1yh411V7ge",
    )
    queries: List[str] = ("时间最短的视频", "时长大于600秒的视频")


# -----------------------------------------------------------------------------
# 1. 初始化视频数据
# -----------------------------------------------------------------------------
def load_bilibili_documents(video_urls: List[str]):
    loader = BiliBiliLoader(video_urls=video_urls)
    docs = loader.load()

    processed_docs = []
    for doc in docs:
        original = doc.metadata
        metadata = {
            "title": original.get("title", "未知标题"),
            "author": original.get("owner", {}).get("name", "未知作者"),
            "source": original.get("bvid", "未知ID"),
            "view_count": original.get("stat", {}).get("view", 0),
            "length": original.get("duration", 0),
        }
        doc.metadata = metadata
        processed_docs.append(doc)

    return processed_docs


# -----------------------------------------------------------------------------
# 2. 创建向量存储
# -----------------------------------------------------------------------------
def build_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name=Config.embed_model)
    return Chroma.from_documents(documents, embedding=embeddings)


# -----------------------------------------------------------------------------
# 3. 配置元数据字段信息
# -----------------------------------------------------------------------------
def get_metadata_schema():
    return [
        AttributeInfo(name="title", description="视频标题（字符串）", type="string"),
        AttributeInfo(name="author", description="视频作者（字符串）", type="string"),
        AttributeInfo(name="view_count", description="视频观看次数（整数）", type="integer"),
        AttributeInfo(name="length", description="视频长度（整数）", type="integer"),
    ]


# -----------------------------------------------------------------------------
# 4. 创建自查询检索器
# -----------------------------------------------------------------------------
def build_retriever(cfg: Config, vectorstore):
    llm = ChatOpenAI(
        model_name=cfg.model_name,
        openai_api_key=cfg.api_key,
        openai_api_base=cfg.api_base,
        temperature=0,
    )
    return SelfQueryRetriever.from_llm(
        llm=llm,
        document_contents="记录视频标题、作者、观看次数等信息的视频元数据",
        metadata_field_info=get_metadata_schema(),
        vectorstore=vectorstore,
        enable_limit=True,
        verbose=True,
    )


# -----------------------------------------------------------------------------
# 5. 执行查询示例
# -----------------------------------------------------------------------------
def run_queries(retriever, queries: List[str]):
    for query in queries:
        print(f"\n--- 查询: '{query}' ---")
        results = retriever.invoke(query)
        if not results:
            print("未找到匹配的视频")
            continue

        for doc in results:
            metadata = doc.metadata
            print(f"标题: {metadata.get('title', '未知标题')}")
            print(f"作者: {metadata.get('author', '未知作者')}")
            print(f"观看次数: {metadata.get('view_count', '未知')}")
            print(f"时长: {metadata.get('length', '未知')}秒")
            print("=" * 50)


def main():
    cfg = Config()

    try:
        docs = load_bilibili_documents(cfg.video_urls)
    except Exception as exc:
        logging.error("加载BiliBili视频失败: %s", exc)
        raise SystemExit(1)

    if not docs:
        logging.error("没有成功加载任何视频，程序退出")
        raise SystemExit(1)

    vectorstore = build_vectorstore(docs)
    retriever = build_retriever(cfg, vectorstore)
    run_queries(retriever, cfg.queries)


if __name__ == "__main__":
    main()
