import logging
from typing import List
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class IndexConstructionModule:
    """Responsible for vectorization and indexing construction

    attrs:
        model_name: embedding model name
        index_save_paath: local path for storing indexes
        embeddings: embedding model obejct
        vectorstore:
    """

    def __init__(
        self, model_name: str = "BAAI/bge-small-zh-v1.5",
        index_save_path: str = "./vector_index"
    ):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore = None
        self.setup_embeddings()
    
    def setup_embeddings(self):
        """Initialize embedding model"""
        logger.info("Initializing embedding model %s.", self.model_name)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info("Successfully initialized embedding model.")
    
    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """
        args:
            chunks: list of document chunks
            
        returns:
            FAISS vector store object
        """
        logger.info("Building FAISS vector index...")
        
        if not chunks:
            raise ValueError("The list of document chunks must not be empty.")
        
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        logger.info("Successfully construct vector index, containing %d vectors.", len(chunks))

        return self.vectorstore
    
    def add_documents(self, new_chunks: List[Document]):
        """Add new documents to existing index
        
        args:
            new_chunks: a list of new document chunks
        """
        if not self.vectorstore:
            raise ValueError("Please construct vector index first")
        
        logger.info("Adding %d new document chunks to index...", len(new_chunks))

        self.vectorstore.add_documents(new_chunks)

        logger.info("Successfully added new documents.")

    def save_index(self):
        if not self.vectorstore:
            raise ValueError("Please construct vector index first")

        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)

        self.vectorstore.save_local(self.index_save_path)

        logger.info("Successfully saved vector index in: %s", self.index_save_path)
    
    def load_index(self):
        """Load vector index from local path

        returns:
            loaded vector store object
        """
        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            logger.info("Local index path doesn't exist: %s, building a new vector index...", self.index_save_path)
            return None

        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Successfully loaded vector index from local path: %s", self.index_save_path)

            return self.vectorstore
        
        except Exception as e:
            logger.warning("Loaded vector index failed: %s, building a new vector index...", e)
            return None
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        args:
            query: query text
            k: number of returned results
            
        returns:
            list of similar documents
        """
        if not self.vectorstore:
            raise ValueError("Construc or load the vector index first.")
        
        return self.vectorstore.similarity_search(query, k=k)
