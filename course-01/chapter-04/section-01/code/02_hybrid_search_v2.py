import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    RRFRanker,
    connections,
)
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoProcessor


# -----------------------------------------------------------------------------
# 0. 日志与配置
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


@dataclass
class Config:
    model_name: str = "google/siglip-base-patch16-256-multilingual"
    device: str = "cpu"  # 如果有 GPU，可设为 "cuda"
    collection_name: str = "dragon_siglip_demo"
    milvus_uri: str = "http://localhost:19530"
    data_path: Path = Path("./data/dragon.json")
    batch_size: int = 50
    search_query: str = "悬崖上的巨龙"
    search_filter: str = 'category in ["western_dragon", "chinese_dragon", "movie_character"]'
    top_k: int = 5


# -----------------------------------------------------------------------------
# 1. SigLIP 嵌入器
# -----------------------------------------------------------------------------
class SigLIPEmbeddingFunction:
    def __init__(self, model_name="google/siglip-base-patch16-256-multilingual", device="cpu"):
        self.model_name = model_name
        self.device = device

        logging.info("加载 SigLIP 模型: %s", model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.to(device).eval()

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10_000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.tfidf_fitted = False

        with torch.no_grad():
            dummy_text = ["test"]
            inputs = self.processor(text=dummy_text, padding="max_length", return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items() if k != "pixel_values"}
            outputs = self.model.text_model(**inputs)
            self.dense_dim = outputs.pooler_output.shape[-1]

        logging.info("SigLIP 模型加载完成，密集向量维度: %s", self.dense_dim)

    @property
    def dim(self):
        return {
            "dense": self.dense_dim,
            "sparse": self.tfidf_vectorizer.max_features if self.tfidf_fitted else 10_000,
        }

    def fit_sparse(self, docs):
        logging.info("拟合 TF-IDF 模型...")
        self.tfidf_vectorizer.fit(docs)
        self.tfidf_fitted = True
        logging.info("TF-IDF 词汇表大小: %s", len(self.tfidf_vectorizer.vocabulary_))

    def encode_text_dense(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        outputs = []
        step = 8
        with torch.no_grad():
            for i in range(0, len(texts), step):
                batch = texts[i : i + step]
                inputs = self.processor(
                    text=batch, padding="max_length", truncation=True, return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "pixel_values"}
                emb = self.model.text_model(**inputs).pooler_output
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                outputs.extend(emb.cpu().numpy())
        return np.array(outputs)

    def encode_text_sparse(self, texts):
        if not self.tfidf_fitted:
            raise ValueError("请先调用 fit_sparse() 方法拟合 TF-IDF 模型")
        if isinstance(texts, str):
            texts = [texts]
        return self.tfidf_vectorizer.transform(texts)

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if not self.tfidf_fitted:
            self.fit_sparse(texts)
        return {
            "dense": self.encode_text_dense(texts),
            "sparse": self.encode_text_sparse(texts),
        }


# -----------------------------------------------------------------------------
# 2. 数据与集合构建
# -----------------------------------------------------------------------------
def ensure_collection(cfg, ef):
    connections.connect(uri=cfg.milvus_uri)
    client = MilvusClient(uri=cfg.milvus_uri)

    if client.has_collection(cfg.collection_name):
        logging.info("删除已存在的 Collection: %s", cfg.collection_name)
        client.drop_collection(cfg.collection_name)

    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"]),
    ]

    schema = CollectionSchema(fields, description="使用 SigLIP 的龙混合检索示例")
    collection = Collection(name=cfg.collection_name, schema=schema, consistency_level="Strong")

    logging.info("创建稀疏/密集索引 ...")
    collection.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
    collection.create_index("dense_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})

    collection.load()
    logging.info("Collection '%s' 已加载到内存", cfg.collection_name)
    return collection, client


# -----------------------------------------------------------------------------
# 3. 数据准备与插入
# -----------------------------------------------------------------------------
def load_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"数据文件未找到: {path}")
    with path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    docs, metadata = [], []
    for item in dataset:
        parts = [
            item.get("title", ""),
            item.get("description", ""),
            item.get("location", ""),
            item.get("environment", ""),
            # *item.get('combat_details', {}).get('combat_style', []),
            # *item.get('combat_details', {}).get('abilities_used', []),
            # item.get('scene_info', {}).get('time_of_day', '')
        ]
        docs.append(" ".join(filter(None, parts)))
        metadata.append(item)
    logging.info("数据加载完成，共 %s 条", len(docs))
    return docs, metadata


def sparse_to_dict(matrix: csr_matrix):
    vectors = []
    for i in range(matrix.shape[0]):
        row = matrix.getrow(i)
        vectors.append({int(idx): float(val) for idx, val in zip(row.indices, row.data)})
    return vectors


def insert_data(collection, metadata, embeddings):
    logging.info("开始插入数据 ...")
    sparse_vectors = sparse_to_dict(embeddings["sparse"])
    dense_vectors = embeddings["dense"].tolist()

    fields = {
        "img_id": [doc["img_id"] for doc in metadata],
        "path": [doc["path"] for doc in metadata],
        "title": [doc["title"] for doc in metadata],
        "description": [doc["description"] for doc in metadata],
        "category": [doc["category"] for doc in metadata],
        "location": [doc["location"] for doc in metadata],
        "environment": [doc["environment"] for doc in metadata],
        "sparse_vector": sparse_vectors,
        "dense_vector": dense_vectors,
    }

    collection.insert([fields[k] for k in fields])
    collection.flush()
    logging.info("插入完成，总数: %s", collection.num_entities)


# -----------------------------------------------------------------------------
# 4. 搜索与展示
# -----------------------------------------------------------------------------
def run_search(collection, ef, cfg):
    logging.info("开始混合搜索")
    query_embeddings = ef([cfg.search_query])
    dense_vec = query_embeddings["dense"][0].tolist()

    sparse_row = query_embeddings["sparse"].getrow(0)
    sparse_dict = {int(idx): float(val) for idx, val in zip(sparse_row.indices, sparse_row.data)}

    logging.info("密集向量维度: %d, 稀疏非零: %d", len(dense_vec), sparse_row.nnz)

    search_params = {"metric_type": "IP", "params": {}}
    output = ["title", "path", "description", "category", "location", "environment"]

    dense_results = collection.search(
        [dense_vec], anns_field="dense_vector", param=search_params, limit=cfg.top_k, expr=cfg.search_filter, output_fields=output
    )[0]

    sparse_results = collection.search(
        [sparse_dict], anns_field="sparse_vector", param=search_params, limit=cfg.top_k, expr=cfg.search_filter, output_fields=output
    )[0]

    logging.info("--- [单独] 密集向量搜索结果 ---")
    for i, hit in enumerate(dense_results, 1):
        logging.info("%d. %s (Score: %.4f)", i, hit.entity.get("title"), hit.distance)

    logging.info("--- [单独] 稀疏向量搜索结果 ---")
    for i, hit in enumerate(sparse_results, 1):
        logging.info("%d. %s (Score: %.4f)", i, hit.entity.get("title"), hit.distance)

    rerank = RRFRanker(k=60)
    dense_req = AnnSearchRequest([dense_vec], "dense_vector", search_params, limit=cfg.top_k)
    sparse_req = AnnSearchRequest([sparse_dict], "sparse_vector", search_params, limit=cfg.top_k)

    results = collection.hybrid_search(
        [sparse_req, dense_req],
        rerank=rerank,
        limit=cfg.top_k,
        output_fields=output,
    )[0]

    logging.info("--- [混合] 稀疏 + 密集搜索结果 ---")
    for i, hit in enumerate(results, 1):
        logging.info("%d. %s (Score: %.4f)", i, hit.entity.get("title"), hit.distance)


# -----------------------------------------------------------------------------
# 5. 主流程
# -----------------------------------------------------------------------------
def main():
    cfg = Config()
    ef = SigLIPEmbeddingFunction(cfg.model_name, cfg.device)
    collection, client = ensure_collection(cfg, ef)

    docs, metadata = load_dataset(cfg.data_path)
    embeddings = ef(docs)
    insert_data(collection, metadata, embeddings)

    run_search(collection, ef, cfg)

    client.release_collection(collection_name=cfg.collection_name)
    client.drop_collection(cfg.collection_name)
    logging.info("已释放并删除 Collection: %s", cfg.collection_name)


if __name__ == "__main__":
    main()
