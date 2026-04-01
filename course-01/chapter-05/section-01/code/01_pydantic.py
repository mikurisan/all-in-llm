from typing import List
import os
import logging
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

llm = ChatOpenAI(
    base_url=os.getenv("BASE_URL", ""),
    model=os.getenv("MODEL_NAME", ""), 
    temperature=0.1, 
    api_key=os.getenv("API_KEY")
)

class PersonInfo(BaseModel):
    name: str = Field(description="人物姓名")
    age: int = Field(description="人物年龄")
    skills: List[str] = Field(description="技能列表")

parser = PydanticOutputParser(pydantic_object=PersonInfo)

prompt = PromptTemplate(
    template="请根据以下文本提取信息。\n{format_instructions}\n{text}\n",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

logging.info("Format Instructions:\n")
print(parser.get_format_instructions())

chain = prompt | llm | parser

text = "张三今年30岁，他擅长Python和Go语言。"
result = chain.invoke({"text": text})

logging.info(f"Result type:\n{type(result)}")
logging.info(f"Result:\n{result}")