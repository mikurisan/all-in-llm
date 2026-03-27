import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    LLMChainExtractor,
)
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
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
    data_path: Path = Path("./data/ai.txt")
    query: str = "AI还有哪些缺陷需要克服？"
    chunk_size: int = 300
    chunk_overlap: int = 20
    retriever_k: int = 20
    rerank_top_k: int = 5
    colbert_model: str = "bert-base-uncased"
    embed_model: str = "BAAI/bge-large-zh-v1.5"
    llm_model: str = field(default_factory=lambda: os.getenv("MODEL_NAME", ""))
    llm_base_url: str = field(default_factory=lambda: os.getenv("BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))


# -----------------------------------------------------------------------------
# 1. ColBERT 重排器
# -----------------------------------------------------------------------------

class ColBERTReranker(BaseDocumentCompressor):
    def __init__(self, model_name: str, top_k: int = 5, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "tokenizer", AutoTokenizer.from_pretrained(model_name))
        object.__setattr__(self, "model", AutoModel.from_pretrained(model_name))
        self.model.eval()
        logging.info("ColBERT模型加载完成")

    def encode(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        with torch.inference_mode():
            outputs = self.model(**inputs)
        embeddings = F.normalize(outputs.last_hidden_state, p=2, dim=-1)
        return embeddings, inputs["attention_mask"]

    def maxsim(self, query_emb, doc_embs, query_mask, doc_masks):
        scores = []
        for i, doc_emb in enumerate(doc_embs):
            doc_mask = doc_masks[i : i + 1]
            sim = torch.matmul(query_emb, doc_emb.unsqueeze(0).transpose(-2, -1))
            sim = sim.masked_fill(~doc_mask.unsqueeze(1).bool(), -1e9)
            max_sim = sim.max(dim=-1)[0]
            max_sim = max_sim.masked_fill(~query_mask.unsqueeze(0).bool(), 0)
            scores.append(max_sim.sum(dim=-1).item())
        return scores

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        if not documents:
            return documents

        query_emb, query_mask = self.encode([query])
        doc_texts = [doc.page_content for doc in documents]
        doc_emb, doc_masks = self.encode(doc_texts)

        scores = self.maxsim(query_emb, doc_emb, query_mask, doc_masks)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]


# -----------------------------------------------------------------------------
# 2. 数据加载与切块
# -----------------------------------------------------------------------------

def load_and_split(cfg: Config):
    loader = TextLoader(str(cfg.data_path), encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    return splitter.split_documents(documents)


# -----------------------------------------------------------------------------
# 3. 底座组件初始化
# -----------------------------------------------------------------------------

def build_components(cfg: Config):
    embeddings = HuggingFaceBgeEmbeddings(model_name=cfg.embed_model)
    llm = ChatOpenAI(
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
        temperature=0.1,
        api_key=cfg.llm_api_key,
    )
    reranker = ColBERTReranker(model_name=cfg.colbert_model, top_k=cfg.rerank_top_k)
    compressor = LLMChainExtractor.from_llm(llm)
    return embeddings, reranker, compressor


# -----------------------------------------------------------------------------
# 4. 检索链路构建
# -----------------------------------------------------------------------------

def build_retrievers(docs, embeddings, reranker, compressor, cfg: Config):
    vectorstore = FAISS.from_documents(docs, embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": cfg.retriever_k})

    pipeline = DocumentCompressorPipeline(transformers=[reranker, compressor])

    final_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline,
        base_retriever=base_retriever,
    )
    return base_retriever, final_retriever


# -----------------------------------------------------------------------------
# 5. 查询执行与展示
# -----------------------------------------------------------------------------

def run_query(base_retriever, final_retriever, query: str):
    print(f"\n{'='*20} 开始执行查询 {'='*20}")
    print(f"查询: {query}\n")

    print("--- (1) 基础检索结果 (Top 20) ---")
    base_results = base_retriever.get_relevant_documents(query)
    for idx, doc in enumerate(base_results, 1):
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  [{idx}] {preview}...\n")

    print("\n--- (2) 管道压缩后结果 (ColBERT重排 + LLM压缩) ---")
    final_results = final_retriever.get_relevant_documents(query)
    for idx, doc in enumerate(final_results, 1):
        print(f"  [{idx}] {doc.page_content}\n")


# -----------------------------------------------------------------------------
# 入口函数
# -----------------------------------------------------------------------------

def main():
    cfg = Config()
    docs = load_and_split(cfg)
    embeddings, reranker, compressor = build_components(cfg)
    base_retriever, final_retriever = build_retrievers(
        docs, embeddings, reranker, compressor, cfg
    )
    run_query(base_retriever, final_retriever, cfg.query)


if __name__ == "__main__":
    main()
