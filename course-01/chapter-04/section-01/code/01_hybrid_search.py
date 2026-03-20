import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
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
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

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
    collection_name: str = "dragon_hybrid_demo"
    milvus_uri: str = "http://localhost:19530"
    data_path: Path = Path("./data/dragon.json")
    batch_size: int = 50
    device: str = "cpu"
    use_fp16: bool = False
    top_k: int = 5
    search_filter: str = 'category in ["western_dragon", "chinese_dragon", "movie_character"]'
    search_query: str = "悬崖上的巨龙"


# -----------------------------------------------------------------------------
# 1. 连接 Milvus & 初始化模型
# -----------------------------------------------------------------------------
def init_milvus(cfg: Config) -> Tuple[MilvusClient, BGEM3EmbeddingFunction]:
    logging.info("连接 Milvus -> %s", cfg.milvus_uri)
    connections.connect(uri=cfg.milvus_uri)
    client = MilvusClient(uri=cfg.milvus_uri)

    logging.info("初始化 BGE-M3 嵌入模型 (device=%s, fp16=%s)", cfg.device, cfg.use_fp16)
    embedding_fn = BGEM3EmbeddingFunction(use_fp16=cfg.use_fp16, device=cfg.device)
    logging.info("模型加载完成， dense 维度=%s", embedding_fn.dim["dense"])
    return client, embedding_fn


# -----------------------------------------------------------------------------
# 2. 构建或重置 Collection
# -----------------------------------------------------------------------------
def prepare_collection(cfg: Config, client: MilvusClient, embedding_fn: BGEM3EmbeddingFunction) -> Collection:
    if client.has_collection(cfg.collection_name):
        logging.info("检测到已有 Collection，删除以保证干净状态 -> %s", cfg.collection_name)
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
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_fn.dim["dense"]),
    ]
    schema = CollectionSchema(fields, description="关于龙的混合检索示例")

    logging.info("创建新 Collection -> %s", cfg.collection_name)
    collection = Collection(name=cfg.collection_name, schema=schema, consistency_level="Strong")

    logging.info("创建稀疏向量索引 (SPARSE_INVERTED_INDEX)")
    collection.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})

    logging.info("创建密集向量索引 (AUTOINDEX)")
    collection.create_index("dense_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})

    return collection


# -----------------------------------------------------------------------------
# 3. 加载数据并生成嵌入
# -----------------------------------------------------------------------------
def load_dataset(cfg: Config) -> Tuple[List[str], List[dict]]:
    if not cfg.data_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {cfg.data_path}")

    logging.info("加载数据集 -> %s", cfg.data_path)
    with cfg.data_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    docs, metadata = [], []
    for item in dataset:
        parts = [
            item.get("title", ""),
            item.get("description", ""),
            item.get("location", ""),
            item.get("environment", ""),
        ]
        docs.append(" ".join(filter(None, parts)))
        metadata.append(item)

    logging.info("数据集加载完成，共 %d 条记录", len(docs))
    return docs, metadata


def generate_embeddings(docs: List[str], embedding_fn: BGEM3EmbeddingFunction):
    logging.info("生成向量嵌入 ...")
    embeddings = embedding_fn(docs)

    dense_vectors = embeddings["dense"]
    sparse_vectors = embeddings["sparse"]

    dense_count = len(dense_vectors)
    dense_dim = len(dense_vectors[0]) if dense_vectors else 0

    logging.info(
        "向量生成完成 | dense: %d 条 (dim=%d) | sparse: shape=%s",
        dense_count,
        dense_dim,
        sparse_vectors.shape,
    )
    return embeddings


# -----------------------------------------------------------------------------
# 4. 数据插入
# -----------------------------------------------------------------------------
def insert_data(collection: Collection, metadata: List[dict], embeddings) -> None:
    logging.info("插入数据到 Milvus ...")
    fields_data = [
        [doc["img_id"] for doc in metadata],
        [doc["path"] for doc in metadata],
        [doc["title"] for doc in metadata],
        [doc["description"] for doc in metadata],
        [doc["category"] for doc in metadata],
        [doc["location"] for doc in metadata],
        [doc["environment"] for doc in metadata],
        embeddings["sparse"],
        embeddings["dense"],
    ]
    collection.insert(fields_data)
    collection.flush()
    logging.info("插入完成，当前实体数=%d", collection.num_entities)


# -----------------------------------------------------------------------------
# 5. 执行混合检索
# -----------------------------------------------------------------------------
def run_search(cfg: Config, collection: Collection, embedding_fn: BGEM3EmbeddingFunction) -> None:
    logging.info("加载 Collection 到内存")
    collection.load()

    query_embeddings = embedding_fn([cfg.search_query])
    dense_vec = query_embeddings["dense"][0]
    sparse_vec = query_embeddings["sparse"]._getrow(0)

    logging.info(
        "查询: %s | dense-norm=%.4f | sparse-nnz=%d",
        cfg.search_query,
        np.linalg.norm(dense_vec),
        sparse_vec.nnz,
    )

    search_params = {"metric_type": "IP", "params": {}}
    top_k = cfg.top_k

    logging.info("执行密集向量搜索 ...")
    dense_hits = collection.search(
        [dense_vec],
        anns_field="dense_vector",
        param=search_params,
        limit=top_k,
        expr=cfg.search_filter,
        output_fields=["title", "path", "description", "category", "location", "environment"],
    )[0]

    logging.info("执行稀疏向量搜索 ...")
    sparse_hits = collection.search(
        [sparse_vec],
        anns_field="sparse_vector",
        param=search_params,
        limit=top_k,
        expr=cfg.search_filter,
        output_fields=["title", "path", "description", "category", "location", "environment"],
    )[0]

    logging.info("执行混合搜索 (RRF)")
    rerank = RRFRanker(k=60)
    dense_req = AnnSearchRequest([dense_vec], "dense_vector", search_params, limit=top_k)
    sparse_req = AnnSearchRequest([sparse_vec], "sparse_vector", search_params, limit=top_k)
    hybrid_hits = collection.hybrid_search(
        [sparse_req, dense_req],
        rerank=rerank,
        limit=top_k,
        output_fields=["title", "path", "description", "category", "location", "environment"],
    )[0]

    print_results("密集向量结果", dense_hits)
    print_results("稀疏向量结果", sparse_hits)
    print_results("混合结果 (RRF)", hybrid_hits)


def print_results(title: str, hits) -> None:
    print(f"\n--- {title} ---")
    for idx, hit in enumerate(hits, start=1):
        title_txt = hit.entity.get("title", "N/A")
        path = hit.entity.get("path", "N/A")
        desc = (hit.entity.get("description") or "")[:100]
        print(f"{idx}. {title_txt} | Score={hit.distance:.4f}")
        print(f"   Path: {path}")
        print(f"   Desc: {desc}...")


# -----------------------------------------------------------------------------
# 6. 清理资源
# -----------------------------------------------------------------------------
def cleanup(cfg: Config, client: MilvusClient) -> None:
    logging.info("释放并删除 Collection -> %s", cfg.collection_name)
    client.release_collection(cfg.collection_name)
    client.drop_collection(cfg.collection_name)

def main():
    cfg = Config()
    client, embedding_fn = init_milvus(cfg)
    collection = prepare_collection(cfg, client, embedding_fn)

    docs, metadata = load_dataset(cfg)
    embeddings = generate_embeddings(docs, embedding_fn)
    insert_data(collection, metadata, embeddings)

    run_search(cfg, collection, embedding_fn)
    cleanup(cfg, client)


if __name__ == "__main__":
    main()