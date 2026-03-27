import logging
import os
from dataclasses import dataclass
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI


# -----------------------------------------------------------------------------
# 日志与配置
# -----------------------------------------------------------------------------
@dataclass
class Config:
    base_url: str = os.getenv("BASE_URL", "")
    model_name: str = os.getenv("MODEL_NAME", "")
    api_key: str = os.getenv("API_KEY", "")
    temperature: float = 0.0
    demo_questions: tuple = (
        "麻婆豆腐怎么做？",
        "白切鸡的正宗做法是什么？",
        "番茄炒蛋需要放糖吗？",
    )


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def build_llm(cfg: Config) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=cfg.base_url,
        model=cfg.model_name,
        temperature=cfg.temperature,
        api_key=cfg.api_key,
    )


# -----------------------------------------------------------------------------
# 1. 设置不同菜系的处理链
# -----------------------------------------------------------------------------
def build_router(llm: ChatDeepSeek):
    sichuan_prompt = ChatPromptTemplate.from_template(
        "你是一位川菜大厨。请用正宗的川菜做法，回答关于「{question}」的问题。"
    )
    sichuan_chain = sichuan_prompt | llm | StrOutputParser()

    cantonese_prompt = ChatPromptTemplate.from_template(
        "你是一位粤菜大厨。请用经典的粤菜做法，回答关于「{question}」的问题。"
    )
    cantonese_chain = cantonese_prompt | llm | StrOutputParser()

    # 定义备用通用链
    general_prompt = ChatPromptTemplate.from_template(
        "你是一个美食助手。请回答关于「{question}」的问题。"
    )
    general_chain = general_prompt | llm | StrOutputParser()

    # -----------------------------------------------------------------------------
    # 2. 创建路由链
    # -----------------------------------------------------------------------------
    classifier_prompt = ChatPromptTemplate.from_template(
        """根据用户问题中提到的菜品，将其分类为：['川菜', '粤菜', 或 '其他']。
    不要解释你的理由，只返回一个单词的分类结果。
    问题: {question}"""
    )
    classifier_chain = classifier_prompt | llm | StrOutputParser()

    # 定义路由分支
    router_branch = RunnableBranch(
        (lambda x: "川菜" in x["topic"], sichuan_chain),
        (lambda x: "粤菜" in x["topic"], cantonese_chain),
        general_chain,
    )

    # 组合成完整路由链
    full_router_chain = (
        {"topic": classifier_chain, "question": itemgetter("question")}
        | RunnableLambda(
            lambda x: {
                "topic": x["topic"],
                "answer": router_branch.invoke(x),
            }
        )
    )
    logging.info("完整的路由链创建成功。")
    return full_router_chain


# -----------------------------------------------------------------------------
# 3. 运行演示查询
# -----------------------------------------------------------------------------
def run_demo(router_chain, questions: tuple):
    for idx, question in enumerate(questions, start=1):
        logging.info("--- 问题 %s: %s ---", idx, question)
        result = router_chain.invoke({"question": question})
        logging.info("路由决策: %s", result["topic"])
        print(f"回答: {result['answer']}\n")


def main():
    cfg = Config()
    llm = build_llm(cfg)
    router_chain = build_router(llm)
    run_demo(router_chain, cfg.demo_questions)


if __name__ == "__main__":
    main()
