import logging
from typing import List, Dict, Any, Iterable

from collections import defaultdict
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from operator import itemgetter

logger = logging.getLogger(__name__)

class RetrievalOptimizationModule:
    """Responsible for hybrid retrieval and filtering
    
    attrs:
        vectorstore: vector store object
        chunks: a list of document chunks
    """
    
    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_retrievers()

    def setup_retrievers(self):
        logger.info("Setting up the retrievers...")

        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5
        )

        logger.info("Successfully setted up retrievers.")
    
    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """Hybrid search using RRF rerank

        args:
            query: query text
            top_k: number of returned results

        returns:
            list of retrieved documents
        """
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)

        return reranked_docs[:top_k]
    
    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """Retrieval with metadata-based filtering
        
        args:
            query: query text
            filters: metadata filtering condition
            top_k: number of documents to return

        returns:
            list of filtered document
        """
        candidate_docs = self.hybrid_search(query, top_k * 3)
        if not filters:
            return candidate_docs[:top_k]
        
        def _matches_filters(doc: Document, filters: Dict[str, Any]) -> bool:
            """Check whether a document satisfies all metadata filters."""
            metadata = getattr(doc, "metadata", {}) or {}
            for key, expected in filters.items():
                if key not in metadata:
                    return False
                value = metadata[key]
                if isinstance(expected, list):
                    if value not in expected:
                        return False
                else:
                    if value != expected:
                        return False
            return True

        filtered_docs: List[Document] = []
        for doc in candidate_docs:
            if _matches_filters(doc, filters):
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break
        return filtered_docs

    def _rrf_rerank(
        self, vector_docs: List[Document], bm25_docs: List[Document],
        k: int = 60
    ) -> List[Document]:
        """Re-rank documents using RRF (Reciprocal Rank Fusion)

        args:
            vector_docs: retrieved documents from vector retrieval
            bm25_docs: retrieved documents from BM25 retrieval
            k: parameter for rank smoothing

        returns:
            lisk of re-ranked documents
        """

        doc_scores: defaultdict[int, float] = defaultdict(float)
        doc_objects: dict[int, Document] = {}

        def accumulate_scores(docs: Iterable[Document], source: str) -> None:
            for rank, doc in enumerate(docs, start=1):
                doc_id = hash(doc.page_content)
                doc_objects[doc_id] = doc
                rrf_score = 1.0 / (k + rank)
                doc_scores[doc_id] += rrf_score
                logger.debug("%s retrieval - doc%d: RRF score = %.4f", source, rank, rrf_score)

        accumulate_scores(vector_docs, "Vector")
        accumulate_scores(bm25_docs, "BM25")

        reranked_docs: List[Document] = []
        for doc_id, final_score in sorted(doc_scores.items(), key=itemgetter(1), reverse=True):
            doc = doc_objects[doc_id]
            doc.metadata = getattr(doc, "metadata", {}) or {}
            doc.metadata["rrf_score"] = final_score
            reranked_docs.append(doc)
            logger.debug(
                "Final rank - doc: %s... final RRF score: %.4f",
                doc.page_content[:50],
                final_score,
            )

        logger.info(
            "RRF re-rank completed: vector retrieved %d docs, BM25 retrieved %d docs, merged %d docs.",
            len(vector_docs),
            len(bm25_docs),
            len(reranked_docs),
        )

        return reranked_docs