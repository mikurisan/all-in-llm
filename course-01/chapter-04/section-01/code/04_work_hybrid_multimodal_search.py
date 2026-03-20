import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
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
from tqdm import tqdm
from visual_bge.modeling import Visualized_BGE

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
    visual_model_name: str = "BAAI/bge-base-en-v1.5"
    visual_model_path: Path = Path("../../../model/weight/Visualized_base_en_v1.5.pth")
    data_dir: Path = Path("./data/dragon")
    metadata_path: Path = Path("./data/dragon.json")
    collection_name: str = "hybrid_multimodal_dragon_demo"
    milvus_uri: str = "http://localhost:19530"
    milvus_consistency: str = "Strong"
    drop_existing: bool = True
    query_image: Path = Path("./data/dragon/query.png")
    query_text: str = "悬崖上的巨龙"
    top_k: int = 3
    device: str = "cpu"


# -----------------------------------------------------------------------------
# 1. 数据结构
# -----------------------------------------------------------------------------


@dataclass
class DragonImage:
    img_id: str
    path: str
    title: str
    description: str
    category: str
    location: str
    environment: str
    combat_details: Dict[str, Any] | None = None
    scene_info: Dict[str, Any] | None = None


# -----------------------------------------------------------------------------
# 2. 数据集管理
# -----------------------------------------------------------------------------


class DragonDataset:
    def __init__(self, data_dir: Path, metadata_path: Path):
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.images: List[DragonImage] = []
        self._load_metadata()

    def _load_metadata(self) -> None:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata file missing: {self.metadata_path}")
        data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        for payload in data:
            img_path = Path(payload["path"])
            if not img_path.is_absolute():
                img_path = self.data_dir / img_path.name
            payload["path"] = str(img_path)
            self.images.append(DragonImage(**payload))
        logging.info("加载了 %d 张龙类图像", len(self.images))

    def get_text_content(self, img: DragonImage) -> str:
        parts = [
            img.title,
            img.description,
            img.location,
            img.environment,
        ]
        if img.combat_details:
            parts.extend(img.combat_details.get("combat_style", []))
            parts.extend(img.combat_details.get("abilities_used", []))
        if img.scene_info:
            parts.append(img.scene_info.get("time_of_day", ""))
        return " ".join(filter(None, parts))


# -----------------------------------------------------------------------------
# 3. 编码器
# -----------------------------------------------------------------------------


class HybridMultimodalEncoder:
    def __init__(self, model_name: str, model_path: Path, device: str):
        logging.info("初始化 Visual-BGE 模型: %s", model_name)
        self.visual_model = Visualized_BGE(
            model_name_bge=model_name,
            model_weight=str(model_path),
        )
        self.visual_model.eval()

        logging.info("初始化 BGE-M3 模型")
        self.bge_m3 = BGEM3EmbeddingFunction(use_fp16=False, device=device)
        logging.info("BGE-M3 密集向量维度: %d", self.bge_m3.dim["dense"])

    def encode_multimodal(self, image_path: str, text: str) -> List[float]:
        with torch.no_grad():
            embedding = self.visual_model.encode(image=image_path, text=text)
        return embedding.tolist()[0]

    def encode_text(self, text: str) -> Dict[str, Any]:
        embeddings = self.bge_m3([text])
        return {
            "dense": embeddings["dense"][0],
            "sparse": embeddings["sparse"]._getrow(0),
        }


# -----------------------------------------------------------------------------
# 4. 可视化助手
# -----------------------------------------------------------------------------


def visualize_results(
    query_image_path: Path,
    retrieved_results: List[Dict[str, Any]],
    search_mode: str,
    img_height: int = 300,
    img_width: int = 300,
    row_count: int = 3,
) -> np.ndarray:
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, np.uint8)
    query_display = np.full((panoramic_height, img_width, 3), 255, np.uint8)

    if query_image_path.exists():
        query_pil = Image.open(query_image_path).convert("RGB")
        query_cv = np.array(query_pil)[:, :, ::-1]
        resized_query = cv2.resize(query_cv, (img_width, img_height))
        bordered_query = cv2.copyMakeBorder(
            resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0)
        )
        query_display[-img_height:, :] = cv2.resize(bordered_query, (img_width, img_height))
        cv2.putText(query_display, "Query", (10, panoramic_height - 40), 0, 0.8, (255, 0, 0), 2)
        cv2.putText(query_display, search_mode, (10, panoramic_height - 10), 0, 0.6, (0, 100, 0), 2)

    for i, result in enumerate(retrieved_results):
        row, col = divmod(i, row_count)
        start_row, start_col = row * img_height, col * img_width

        img_path = result["image_path"]
        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        resized = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered = cv2.copyMakeBorder(resized, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic_image[start_row : start_row + img_height, start_col : start_col + img_width] = bordered
        cv2.putText(panoramic_image, f"{i+1}", (start_col + 10, start_row + 30), 0, 1, (0, 0, 255), 2)
        cv2.putText(
            panoramic_image,
            f"{result['distance']:.3f}",
            (start_col + 10, start_row + img_height - 10),
            0,
            0.5,
            (0, 255, 0),
            1,
        )

    return np.hstack([query_display, panoramic_image])


# -----------------------------------------------------------------------------
# 5. 混合多模态搜索系统
# -----------------------------------------------------------------------------


class HybridMultimodalSearcher:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        logging.info("初始化数据集: %s", cfg.metadata_path)
        self.dataset = DragonDataset(cfg.data_dir, cfg.metadata_path)

        logging.info("初始化混合多模态编码器")
        self.encoder = HybridMultimodalEncoder(
            cfg.visual_model_name,
            cfg.visual_model_path,
            cfg.device,
        )

        logging.info("连接 Milvus: %s", cfg.milvus_uri)
        connections.connect(uri=cfg.milvus_uri)
        self.milvus_client = MilvusClient(uri=cfg.milvus_uri)
        self.collection: Collection | None = None

    def create_collection(self) -> None:
        if self.cfg.drop_existing and self.milvus_client.has_collection(self.cfg.collection_name):
            self.milvus_client.drop_collection(self.cfg.collection_name)
            logging.info("已删除存在的 Collection: %s", self.cfg.collection_name)

        sample_image = self.dataset.images[0]
        text_sample = self.dataset.get_text_content(sample_image)
        multimodal_dim = len(self.encoder.encode_multimodal(sample_image.path, text_sample))
        dense_dim = self.encoder.bge_m3.dim["dense"]

        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="multimodal_vector", dtype=DataType.FLOAT_VECTOR, dim=multimodal_dim),
            FieldSchema(name="text_sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="text_dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
        ]

        schema = CollectionSchema(fields, description="混合多模态龙类图像检索")
        self.collection = Collection(
            name=self.cfg.collection_name,
            schema=schema,
            consistency_level=self.cfg.milvus_consistency,
        )
        logging.info("Collection 创建成功: %s", self.cfg.collection_name)

        self.collection.create_index(
            "multimodal_vector",
            {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 256}},
        )
        self.collection.create_index(
            "text_sparse_vector",
            {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"},
        )
        self.collection.create_index(
            "text_dense_vector",
            {"index_type": "AUTOINDEX", "metric_type": "IP"},
        )
        self.collection.load()
        logging.info("Collection 已加载到内存")

    def insert_data(self) -> None:
        if not self.collection:
            raise RuntimeError("collection 未创建")

        if not self.collection.is_empty:
            logging.info("Collection 已包含 %d 条数据，跳过插入", self.collection.num_entities)
            return

        payloads = {
            "img_id": [],
            "image_path": [],
            "title": [],
            "description": [],
            "category": [],
            "location": [],
            "environment": [],
            "multimodal_vector": [],
            "text_sparse_vector": [],
            "text_dense_vector": [],
        }

        for img in tqdm(self.dataset.images, desc="生成向量嵌入"):
            text_content = self.dataset.get_text_content(img)
            payloads["img_id"].append(img.img_id)
            payloads["image_path"].append(img.path)
            payloads["title"].append(img.title)
            payloads["description"].append(img.description)
            payloads["category"].append(img.category)
            payloads["location"].append(img.location)
            payloads["environment"].append(img.environment)
            payloads["multimodal_vector"].append(self.encoder.encode_multimodal(img.path, text_content))

            text_embeddings = self.encoder.encode_text(text_content)
            payloads["text_dense_vector"].append(text_embeddings["dense"])
            payloads["text_sparse_vector"].append(text_embeddings["sparse"])

        entities = [
            payloads["img_id"],
            payloads["image_path"],
            payloads["title"],
            payloads["description"],
            payloads["category"],
            payloads["location"],
            payloads["environment"],
            payloads["multimodal_vector"],
            payloads["text_sparse_vector"],
            payloads["text_dense_vector"],
        ]

        self.collection.insert(entities)
        self.collection.flush()
        logging.info("数据插入完成，总数: %d", self.collection.num_entities)

    def search(self, query_image_path: Path, query_text: str, mode: str, top_k: int):
        if not self.collection:
            raise RuntimeError("collection 未创建")

        cosine_params = {"metric_type": "COSINE", "params": {"ef": 128}}
        search_params = {"metric_type": "IP", "params": {}}
        output_fields = ["img_id", "image_path", "title", "description", "category", "location", "environment"]

        if mode == "multimodal":
            query_vector = self.encoder.encode_multimodal(str(query_image_path), query_text)
            return self.collection.search(
                [query_vector],
                "multimodal_vector",
                param=cosine_params,
                limit=top_k,
                output_fields=output_fields,
            )[0]

        embeddings = self.encoder.encode_text(query_text)
        dense_vec, sparse_vec = embeddings["dense"], embeddings["sparse"]

        if mode == "dense":
            return self.collection.search(
                [dense_vec],
                "text_dense_vector",
                param=search_params,
                limit=top_k,
                output_fields=output_fields,
            )[0]

        if mode == "sparse":
            return self.collection.search(
                [sparse_vec],
                "text_sparse_vector",
                param=search_params,
                limit=top_k,
                output_fields=output_fields,
            )[0]

        if mode == "hybrid":
            rerank = RRFRanker(k=60)
            dense_req = AnnSearchRequest([dense_vec], "text_dense_vector", search_params, limit=top_k)
            sparse_req = AnnSearchRequest([sparse_vec], "text_sparse_vector", search_params, limit=top_k)
            return self.collection.hybrid_search(
                [sparse_req, dense_req],
                rerank=rerank,
                limit=top_k,
                output_fields=output_fields,
            )[0]

        raise ValueError(f"未知模式: {mode}")

    def compare_search_modes(self, query_image_path: Path, query_text: str, top_k: int) -> Dict[str, List[Dict[str, Any]]]:
        modes = ["multimodal", "dense", "sparse", "hybrid"]
        results: Dict[str, List[Dict[str, Any]]] = {}

        logging.info("=" * 50)
        logging.info("查询图像: %s", query_image_path)
        logging.info("查询文本: %s", query_text)
        logging.info("=" * 50)

        for mode in modes:
            logging.info("--- %s 搜索结果 ---", mode.upper())
            hits = self.search(query_image_path, query_text, mode, top_k)
            mode_results = []
            for idx, hit in enumerate(hits, start=1):
                title = hit.entity.get("title")
                desc = (hit.entity.get("description") or "")[:80]
                path = hit.entity.get("image_path")
                logging.info("%d. %s (Score: %.4f)", idx, title, hit.distance)
                logging.info("   路径: %s", path)
                logging.info("   描述: %s...", desc)
                mode_results.append({"image_path": path, "distance": hit.distance, "title": title})
            results[mode] = mode_results

        return results

    def visualize_comparison(self, query_image_path: Path, query_text: str, top_k: int) -> None:
        for mode in ["multimodal", "dense", "sparse", "hybrid"]:
            hits = self.search(query_image_path, query_text, mode, top_k)
            retrieved = [{"image_path": hit.entity.get("image_path"), "distance": hit.distance} for hit in hits]

            if not retrieved:
                continue

            panoramic = visualize_results(query_image_path, retrieved, mode.upper())
            output_path = self.cfg.data_dir / f"{mode}_search_result.png"
            cv2.imwrite(str(output_path), panoramic)
            logging.info("%s 搜索结果已保存到: %s", mode.upper(), output_path)

    def cleanup(self) -> None:
        if self.collection:
            self.collection.release()
            logging.info("已释放 Collection: %s", self.cfg.collection_name)
            self.milvus_client.drop_collection(self.cfg.collection_name)
            logging.info("已删除 Collection: %s", self.cfg.collection_name)


# -----------------------------------------------------------------------------
# 6. 流程控制
# -----------------------------------------------------------------------------


def main():
    cfg = Config()
    searcher = HybridMultimodalSearcher(cfg)

    try:
        searcher.create_collection()
        searcher.insert_data()
        searcher.compare_search_modes(cfg.query_image, cfg.query_text, cfg.top_k)
        searcher.visualize_comparison(cfg.query_image, cfg.query_text, cfg.top_k)

        logging.info("=" * 50)
        logging.info("搜索模式分析:")
        logging.info("- MULTIMODAL: 结合图像与文本的多模态向量检索")
        logging.info("- DENSE     : 基于语义的密集向量检索")
        logging.info("- SPARSE    : 基于关键词的稀疏向量检索")
        logging.info("- HYBRID    : 稀疏 + 密集向量的 RRF 融合检索")
        logging.info("=" * 50)
    finally:
        searcher.cleanup()


if __name__ == "__main__":
    main()
