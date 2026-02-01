import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 加载环境变量
load_dotenv()

# 配置大语言模型

# 使用 AIHubmix
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    api_base="https://aihubmix.com/v1"
)

#  中文嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 加载本地 markdown 文件
docs = SimpleDirectoryReader(input_files=["./data/easy-rl-chapter1.md"]).load_data()

# 构建向量存储
index = VectorStoreIndex.from_documents(docs)

# 创建查询
query_engine = index.as_query_engine()

print(query_engine.get_prompts())

print(query_engine.query("文中举了哪些例子?"))