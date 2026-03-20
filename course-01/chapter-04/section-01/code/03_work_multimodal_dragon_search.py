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
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
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
    model_name: str = "BAAI/bge-base-en-v1.5"
    model_path: Path = Path("../../../model/weight/Visualized_base_en_v1.5.pth")
    data_dir: Path = Path("./data/dragon")
    metadata_path: Path = Path("./data/dragon.json")
    collection_name: str = "multimodal_dragon_demo"
    milvus_uri: str = "http://localhost:19530"
    query_image_name: str = "query.png"
    query_text: str = "悬崖上的巨龙"
    text_query_top_k: int = 3
    multimodal_top_k: int = 6
    search_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.search_params is None:
            self.search_params = {"metric_type": "COSINE", "params": {"ef": 128}}


# -----------------------------------------------------------------------------
# 1. 数据实体与数据集
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
    combat_details: Dict[str, Any] = None
    scene_info: Dict[str, Any] = None


class DragonDataset:
    def __init__(self, data_dir: Path, metadata_path: Path):
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.images: List[DragonImage] = []
        self._load_metadata()

    def _load_metadata(self):
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")
        with open(self.metadata_path, "r", encoding="utf-8") as fp:
            for entry in json.load(fp):
                img_path = Path(entry["path"])
                if not img_path.is_absolute():
                    img_path = self.data_dir / img_path.name
                entry["path"] = str(img_path)
                self.images.append(DragonImage(**entry))

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
# 2. 编码器
# -----------------------------------------------------------------------------

class Encoder:
    def __init__(self, model_name: str, model_path: Path):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=str(model_path))
        self.model.eval()

    def encode_query(self, image_path: str | None = None, text: str | None = None) -> List[float]:
        with torch.no_grad():
            if image_path and text:
                embedding = self.model.encode(image=image_path, text=text)
            elif image_path:
                embedding = self.model.encode(image=image_path)
            elif text:
                embedding = self.model.encode(text=text)
            else:
                raise ValueError("必须提供图像路径或文本内容")
        return embedding.tolist()[0]

    def encode_multimodal(self, image_path: str, text: str) -> List[float]:
        with torch.no_grad():
            embedding = self.model.encode(image=image_path, text=text)
        return embedding.tolist()[0]


# -----------------------------------------------------------------------------
# 3. 可视化
# -----------------------------------------------------------------------------

def visualize_results(
    query_image_path: Path,
    retrieved_results: List[Dict[str, Any]],
    img_height: int = 300,
    img_width: int = 300,
    row_count: int = 3,
) -> np.ndarray:
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    if query_image_path.exists():
        query_pil = Image.open(query_image_path).convert("RGB")
        query_cv = np.array(query_pil)[:, :, ::-1]
        resized_query = cv2.resize(query_cv, (img_width, img_height))
        bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
        query_display_area[img_height * (row_count - 1) :, :] = cv2.resize(bordered_query, (img_width, img_height))
        cv2.putText(
            query_display_area,
            "Query",
            (10, panoramic_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

    for i, result in enumerate(retrieved_results):
        row, col = divmod(i, row_count)
        start_row, start_col = row * img_height, col * img_width
        img_path = Path(result["image_path"])
        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic_image[start_row : start_row + img_height, start_col : start_col + img_width] = bordered_retrieved
        cv2.putText(
            panoramic_image,
            f"{i + 1}",
            (start_col + 10, start_row + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            panoramic_image,
            f"{result['distance']:.3f}",
            (start_col + 10, start_row + img_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return np.hstack([query_display_area, panoramic_image])


# -----------------------------------------------------------------------------
# 4. Milvus 相关操作
# -----------------------------------------------------------------------------

def prepare_collection(client: MilvusClient, cfg: Config, encoder: Encoder, dataset: DragonDataset) -> int:
    if not dataset.images:
        raise ValueError("数据集中没有任何图像，无法创建 Collection")

    if client.has_collection(cfg.collection_name):
        client.drop_collection(cfg.collection_name)
        logging.info("已删除已存在的 Collection: %s", cfg.collection_name)

    sample = dataset.images[0]
    dim = len(encoder.encode_multimodal(sample.path, dataset.get_text_content(sample)))

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),
    ]
    schema = CollectionSchema(fields, description="多模态龙类图像检索")
    client.create_collection(collection_name=cfg.collection_name, schema=schema)
    logging.info("成功创建 Collection: %s", cfg.collection_name)
    return dim


def insert_embeddings(client: MilvusClient, cfg: Config, dataset: DragonDataset, encoder: Encoder):
    records = []
    for img in tqdm(dataset.images, desc="生成多模态嵌入"):
        text_content = dataset.get_text_content(img)
        records.append(
            {
                "vector": encoder.encode_multimodal(img.path, text_content),
                "img_id": img.img_id,
                "image_path": img.path,
                "title": img.title,
                "description": img.description,
                "category": img.category,
                "location": img.location,
                "environment": img.environment,
            }
        )
    if records:
        result = client.insert(collection_name=cfg.collection_name, data=records)
        logging.info("成功插入 %s 条数据", result["insert_count"])


def create_index_and_load(client: MilvusClient, cfg: Config):
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256},
    )
    client.create_index(collection_name=cfg.collection_name, index_params=index_params)
    client.load_collection(collection_name=cfg.collection_name)
    logging.info("索引创建完成并已加载 Collection")


def search_with_logging(
    client: MilvusClient,
    cfg: Config,
    query_vector: List[float],
    top_k: int,
    label: str,
) -> List[Dict[str, Any]]:
    results = client.search(
        collection_name=cfg.collection_name,
        data=[query_vector],
        output_fields=["img_id", "image_path", "title", "description", "category", "location", "environment"],
        limit=top_k,
        search_params=cfg.search_params,
    )[0]
    logging.info("%s 检索结果:", label)
    formatted = []
    for idx, hit in enumerate(results, start=1):
        entity = hit["entity"]
        logging.info(
            "  Top %d | 距离=%.4f | 标题=%s | 类别=%s | 路径=%s",
            idx,
            hit["distance"],
            entity["title"],
            entity["category"],
            entity["image_path"],
        )
        formatted.append({"image_path": entity["image_path"], "distance": hit["distance"]})
    return formatted


# -----------------------------------------------------------------------------
# 5. 主流程
# -----------------------------------------------------------------------------

def main():
    cfg = Config()
    logging.info("--> 正在初始化数据集与编码器")
    dataset = DragonDataset(cfg.data_dir, cfg.metadata_path)
    logging.info("加载了 %d 张龙类图像", len(dataset.images))

    encoder = Encoder(cfg.model_name, cfg.model_path)
    milvus_client = MilvusClient(uri=cfg.milvus_uri)

    logging.info("--> 正在创建 Collection")
    prepare_collection(milvus_client, cfg, encoder, dataset)

    logging.info("--> 正在插入数据")
    insert_embeddings(milvus_client, cfg, dataset, encoder)

    logging.info("--> 正在创建索引并加载 Collection")
    create_index_and_load(milvus_client, cfg)

    logging.info("--> 正在执行多模态检索示例")
    query_image_path = cfg.data_dir / cfg.query_image_name
    multimodal_vector = encoder.encode_query(image_path=str(query_image_path), text=cfg.query_text)
    retrieved_results = search_with_logging(
        milvus_client,
        cfg,
        query_vector=multimodal_vector,
        top_k=cfg.multimodal_top_k,
        label="多模态查询",
    )

    text_vector = encoder.encode_query(text=cfg.query_text)
    search_with_logging(
        milvus_client,
        cfg,
        query_vector=text_vector,
        top_k=cfg.text_query_top_k,
        label="纯文本查询",
    )

    image_vector = encoder.encode_query(image_path=str(query_image_path))
    search_with_logging(
        milvus_client,
        cfg,
        query_vector=image_vector,
        top_k=cfg.text_query_top_k,
        label="纯图像查询",
    )

    if retrieved_results:
        output_path = cfg.data_dir / "multimodal_search_result.png"
        panorama = visualize_results(query_image_path, retrieved_results)
        cv2.imwrite(str(output_path), panorama)
        logging.info("检索结果图像已保存到: %s", output_path)
        if os.environ.get("DISPLAY"):
            Image.open(output_path).show()

    logging.info("--> 正在清理 Collection")
    milvus_client.release_collection(collection_name=cfg.collection_name)
    milvus_client.drop_collection(cfg.collection_name)
    logging.info("已释放并删除 Collection: %s", cfg.collection_name)


if __name__ == "__main__":
    main()
