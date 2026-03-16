import os

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

markdown_path = "./data/easy-rl-chapter1.md"

# 加载本地 markdown document
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 将 document 分割为 chunks
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)

# Chinese embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 构建 vector index
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# prompt template
prompt = ChatPromptTemplate.from_template("""
请根据下面提供的上下文信息来回答问题.

请确保你的回答完全基于这些上下文.
如果上下文中没有足够的信息来回答问题, 请直接告知: “抱歉, 我无法根据提供的上下文找到相关信息来回答此问题.”

上下文:
{context}

问题: {question}

回答:"""
)

# 配置 LLM
llm = ChatOpenAI(
    model_name=os.getenv("MODEL_NAME"),
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base=os.getenv("BASE_URL"),
    temperature=0.7,
    max_tokens=4096
)

# 用户 query
question = "文中举了哪些强化学习的例子?"

# 在 vector db 中 retrieve 相关 chunks
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 获取 reponse
response = llm.invoke(prompt.format(question=question, context=docs_content))
print(response.content)