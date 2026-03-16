import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# 配置 LLM
Settings.llm = OpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("BASE_URL")
)

# Chinese embedding model
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 加载本地 markdown file
docs = SimpleDirectoryReader(input_files=["./data/easy-rl-chapter1.md"]).load_data()

# 构建 vector index
index = VectorStoreIndex.from_documents(docs)

# 创建 query
query_engine = index.as_query_engine()

print(query_engine.get_prompts())

print(query_engine.query("文中举了哪些强化学习的例子?"))