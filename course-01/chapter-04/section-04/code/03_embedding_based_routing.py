import logging
import os
from dataclasses import dataclass

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utils.math import cosine_similarity
from langchain_deepseek import ChatDeepSeek


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
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    base_url: str = os.getenv("BASE_URL", "")
    model_name: str = os.getenv("MODEL_NAME", "")
    api_key: str = os.getenv("API_KEY", "")
    temperature: float = 0.0


# -----------------------------------------------------------------------------
# 1. 定义路由描述
# -----------------------------------------------------------------------------
ROUTE_PROMPTS = {
    "川菜": "你是一位处理川菜的专家。用户的问题是关于麻辣、辛香、重口味的菜肴，例如水煮鱼、麻婆豆腐、鱼香肉丝、宫保鸡丁、花椒、海椒等。",
    "粤菜": "你是一位处理粤菜的专家。用户的问题是关于清淡、鲜美、原汁原味的菜肴，例如白切鸡、老火靓汤、虾饺、云吞面等。",
}


# -----------------------------------------------------------------------------
# 2. 构建向量与模型
# -----------------------------------------------------------------------------
def prepare_components(cfg: Config):
    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
    route_vectors = embeddings.embed_documents(list(ROUTE_PROMPTS.values()))
    llm = ChatDeepSeek(
        api_base=cfg.base_url,
        model=cfg.model_name,
        api_key=cfg.api_key,
        temperature=cfg.temperature,
    )
    return embeddings, np.array(route_vectors), llm


# -----------------------------------------------------------------------------
# 3. 构建处理链
# -----------------------------------------------------------------------------
def build_route_chains(llm):
    sichuan_chain = (
        PromptTemplate.from_template(
            "你是一位川菜大厨。请用正宗的川菜做法，回答关于「{query}」的问题。"
        )
        | llm
        | StrOutputParser()
    )
    cantonese_chain = (
        PromptTemplate.from_template(
            "你是一位粤菜大厨。请用经典的粤菜做法，回答关于「{query}」的问题。"
        )
        | llm
        | StrOutputParser()
    )
    return {"川菜": sichuan_chain, "粤菜": cantonese_chain}


# -----------------------------------------------------------------------------
# 4. 创建路由函数
# -----------------------------------------------------------------------------
def build_router(embeddings, route_vectors, chains):
    route_names = list(ROUTE_PROMPTS.keys())

    def route(info):
        query_vec = embeddings.embed_query(info["query"])
        scores = cosine_similarity([query_vec], route_vectors)[0]
        route_idx = int(np.argmax(scores))
        chosen_route = route_names[route_idx]
        logging.info("路由决策 -> %s", chosen_route)
        return chains[chosen_route].invoke(info)

    return RunnableLambda(route)


# -----------------------------------------------------------------------------
# 5. 演示查询
# -----------------------------------------------------------------------------
def run_demo(router):
    demo_queries = [
        "水煮鱼怎么做才嫩？",
        "如何做一碗清淡的云吞面？",
        "麻婆豆腐的核心调料是什么？",
    ]
    for idx, query in enumerate(demo_queries, 1):
        logging.info("--- 问题 %d: %s ---", idx, query)
        try:
            answer = router.invoke({"query": query})
            print(f"回答: {answer}")
        except Exception as exc:
            logging.error("执行错误: %s", exc)


def main():
    cfg = Config()
    embeddings, route_vectors, llm = prepare_components(cfg)
    chains = build_route_chains(llm)
    router = build_router(embeddings, route_vectors, chains)
    run_demo(router)


if __name__ == "__main__":
    main()
