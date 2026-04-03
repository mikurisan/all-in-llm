import os

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # path config
    data_path: str = "./data"
    index_save_path: str = "./vector_index"

    # model config
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    model_name: str = os.getenv("MODEL_NAME", "")
    api_key: str = os.getenv("API_KEY", "")
    base_url: str = os.getenv("BASE_URL", "")

    # retrieval config
    top_k: int = 3

    # generation config
    temperature: float = 0.1
    max_tokens: int = 2048

    def __post_init__(self):
        if not self.model_name:
            raise ValueError("MODEL_NAME is required but not provided.")
        if not self.api_key:
            raise ValueError("API_KEY is required but not provided.")
        if not self.base_url:
            raise ValueError("BASE_URL is required but not provided.")
        
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'data_path': self.data_path,
            'index_save_path': self.index_save_path,
            'embedding_model': self.embedding_model,
            'model_name': self.model_name,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }

DEFAULT_CONFIG = RAGConfig()