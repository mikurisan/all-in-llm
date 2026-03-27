import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    LLMChainExtractor,
)
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


@dataclass
class Config:
    base_url: str = os.getenv("BASE_URL", "")
    model_name: str = os.getenv("MODEL_NAME", "")
    api_key: str = os.getenv("API_KEY", "")
    data_path: Path = Path("./data/ai.txt")
    bge_name: str = "BAAI/bge-large-zh-v1.5"
    colbert_name: str = "bert-base-uncased"
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 20
    rerank_k: int = 5
    query: str = "AI还有哪些缺陷需要克服？"


# -----------------------------------------------------------------------------
# 1. ColBERT 重排器
# -----------------------------------------------------------------------------
class ColBERTReranker(BaseDocumentCompressor):
    def __init__(self, model_name: str = "bert-base-uncased", top_k: int = 5):
        super().__init__()
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "tokenizer", AutoTokenizer.from_pretrained(model_name))
        object.__setattr__(self, "model", AutoModel.from_pretrained(model_name))
        object.__setattr__(self, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        logging.info("ColBERT 模型加载完成")

    # -----------------------------------------------------------------------------
    # 1.1 文本编码
    # -----------------------------------------------------------------------------
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return F.normalize(outputs.last_hidden_state, p=2, dim=-1), inputs["attention_mask"]

    # -----------------------------------------------------------------------------
    # 1.2 MaxSim 计算
    # -----------------------------------------------------------------------------
    def colbert_score(
        self,
        query_emb: torch.Tensor,
        doc_embs: torch.Tensor,
        query_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> Sequence[float]:
        scores = []
        for idx, doc_emb in enumerate(doc_embs):
            sim = torch.matmul(query_emb, doc_emb.unsqueeze(0).transpose(-2, -1))
            doc_mask_expanded = doc_mask[idx : idx + 1].unsqueeze(1).bool()
            sim = sim.masked_fill(~doc_mask_expanded, -1e9)
            max_sim = sim.max(dim=-1)[0]
            query_mask_expanded = query_mask.unsqueeze(0).bool()
            max_sim = max_sim.masked_fill(~query_mask_expanded, 0)
            scores.append(max_sim.sum(dim=-1).item())
        return scores

    # -----------------------------------------------------------------------------
    # 1.3 压缩流程
    # -----------------------------------------------------------------------------
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
        doc_emb, doc_mask = self.encode(doc_texts)

        scores = self.colbert_score(
            query_emb,
            doc_emb,
            query_mask,
            doc_mask,
        )

        reranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked[: self.top_k]]


# -----------------------------------------------------------------------------
# 2. 数据加载与切分
# -----------------------------------------------------------------------------
def load_documents(cfg: Config) -> list[Document]:
    loader = TextLoader(str(cfg.data_path), encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
        )
    return splitter.split_documents(documents)


# -----------------------------------------------------------------------------
# 3. 构建向量库与基础检索器
# -----------------------------------------------------------------------------
def build_retriever(cfg: Config, docs: list[Document]):
    embeddings = HuggingFaceBgeEmbeddings(model_name=cfg.bge_name)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": cfg.top_k})


# -----------------------------------------------------------------------------
# 4. 构建压缩管道
# -----------------------------------------------------------------------------
def build_compressor(cfg: Config):
    llm = ChatOpenAI(
        base_url=cfg.base_url,
        model=cfg.model_name,
        api_key=cfg.api_key,
        temperature=0.1,
    )
    reranker = ColBERTReranker(model_name=cfg.colbert_name, top_k=cfg.rerank_k)
    extractor = LLMChainExtractor.from_llm(llm)
    pipeline = DocumentCompressorPipeline(transformers=[reranker, extractor])
    return pipeline


# -----------------------------------------------------------------------------
# 5. 运行示例查询
# -----------------------------------------------------------------------------
def run_query(retriever, final_retriever, query: str):
    logging.info("开始执行查询")
    logging.info("查询内容: %s", query)

    logging.info("--- (1) 基础检索结果 (Top %s) ---", retriever.search_kwargs["k"])
    base_results = retriever.get_relevant_documents(query)
    for idx, doc in enumerate(base_results, start=1):
        logging.info("  [%d] %s...", idx, doc.page_content[:100].strip())

    logging.info("--- (2) 管道压缩后结果 (ColBERT重排 + LLM压缩) ---")
    final_results = final_retriever.get_relevant_documents(query)
    for idx, doc in enumerate(final_results, start=1):
        logging.info("  [%d] %s", idx, doc.page_content.strip())


def main():
    cfg = Config()

    docs = load_documents(cfg)
    base_retriever = build_retriever(cfg, docs)
    compressor = build_compressor(cfg)

    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    run_query(base_retriever, final_retriever, cfg.query)


if __name__ == "__main__":
    main()
