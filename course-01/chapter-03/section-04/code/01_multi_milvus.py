import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from tqdm import tqdm
from visual_bge.modeling import Visualized_BGE

# -----------------------------------------------------------------------------
# 1. 通用配置
# -----------------------------------------------------------------------------
@dataclass
class Config:
    model_name: str = "BAAI/bge-base-en-v1.5"
    model_path: str = "../../../model/weight/Visualized_base_en_v1.5.pth"
    data_dir: Path = Path("./data")
    collection_name: str = "multimodal_demo"
    milvus_uri: str = "http://localhost:19530"


cfg = Config()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

# -----------------------------------------------------------------------------
# 2. 定义工具 (编码器和可视化函数)
# -----------------------------------------------------------------------------
class Encoder:
    """编码器类：负责将图像/文本转为向量。"""

    def __init__(self, model_name: str, model_path: str):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    def encode_query(self, image_path: str, text: str) -> List[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]

    def encode_image(self, image_path: str) -> List[float]:
        with torch.no_grad():
            embed = self.model.encode(image=image_path)
        return embed.tolist()[0]


def visualize_results(
    query_image_path: Path,
    retrieved_images: List[Path],
    img_height: int = 300,
    img_width: int = 300,
    row_count: int = 3,
) -> np.ndarray:
    """将查询图像与检索结果拼成对比图。"""

    def load_and_resize(path: Path, w: int, h: int) -> np.ndarray:
        return cv2.resize(np.array(Image.open(path).convert("RGB"))[:, :, ::-1], (w, h))

    pano_h, pano_w = img_height * row_count, img_width * row_count
    panoramic = np.full((pano_h, pano_w, 3), 255, dtype=np.uint8)
    query_panel = np.full((pano_h, img_width, 3), 255, dtype=np.uint8)

    query_img = load_and_resize(query_image_path, img_width, img_height)
    query_img = cv2.copyMakeBorder(
        query_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0)
    )
    query_panel[-img_height:] = cv2.resize(query_img, (img_width, img_height))
    cv2.putText(query_panel, "Query", (10, pano_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    for idx, image_path in enumerate(retrieved_images):
        row, col = divmod(idx, row_count)
        start_row, start_col = row * img_height, col * img_width
        retrieved = load_and_resize(image_path, img_width - 4, img_height - 4)
        retrieved = cv2.copyMakeBorder(retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic[start_row : start_row + img_height, start_col : start_col + img_width] = retrieved
        cv2.putText(
            panoramic,
            str(idx),
            (start_col + 10, start_row + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    return np.hstack((query_panel, panoramic))


# -----------------------------------------------------------------------------
# 3. 主流程
# -----------------------------------------------------------------------------
def main() -> None:
    logging.info("初始化 Encoder 与 Milvus 客户端 ...")
    encoder = Encoder(cfg.model_name, cfg.model_path)
    milvus_client = MilvusClient(uri=cfg.milvus_uri)

    # 若存在同名 Collection 则先删除
    if milvus_client.has_collection(cfg.collection_name):
        milvus_client.drop_collection(cfg.collection_name)
        logging.warning("检测到同名 Collection，已删除旧的 '%s'", cfg.collection_name)

    # 准备数据
    image_dir = cfg.data_dir / "dragon"
    image_list = sorted(image_dir.glob("*.png"))
    if not image_list:
        raise FileNotFoundError(f"在 {image_dir} 中未找到任何 .png 图像。")

    dim = len(encoder.encode_image(str(image_list[0])))

    # 定义 Schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
    ]
    schema = CollectionSchema(fields, description="多模态图文检索")
    milvus_client.create_collection(collection_name=cfg.collection_name, schema=schema)
    logging.info("成功创建 Collection '%s'", cfg.collection_name)

    # 插入数据
    logging.info("正在生成图像嵌入并插入数据 ...")
    data_to_insert = []
    for image_path in tqdm(image_list, desc="Encoding"):
        vector = encoder.encode_image(str(image_path))
        data_to_insert.append({"vector": vector, "image_path": str(image_path)})

    if data_to_insert:
        result = milvus_client.insert(collection_name=cfg.collection_name, data=data_to_insert)
        logging.info("成功插入 %d 条数据。", result["insert_count"])

    # 创建索引
    logging.info("正在创建 HNSW 索引 ...")
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256},
    )
    milvus_client.create_index(collection_name=cfg.collection_name, index_params=index_params)
    milvus_client.load_collection(collection_name=cfg.collection_name)
    logging.info("集合已加载至内存。")

    # 检索
    query_image_path = image_dir / "query.png"
    query_text = "一条龙"
    query_vector = encoder.encode_query(str(query_image_path), query_text)

    search_results = milvus_client.search(
        collection_name=cfg.collection_name,
        data=[query_vector],
        output_fields=["image_path"],
        limit=5,
        search_params={"metric_type": "COSINE", "params": {"ef": 128}},
    )[0]

    retrieved_images: List[Path] = []
    logging.info("检索结果：")
    for i, hit in enumerate(search_results):
        img_path = Path(hit["entity"]["image_path"])
        logging.info("Top %d | ID=%s | 距离=%.4f | 路径=%s", i + 1, hit["id"], hit["distance"], img_path)
        retrieved_images.append(img_path)

    # 可视化与清理
    if retrieved_images:
        panoramic_image = visualize_results(query_image_path, retrieved_images)
        combined_image_path = cfg.data_dir / "search_result.png"
        cv2.imwrite(str(combined_image_path), panoramic_image)
        logging.info("结果图像已保存到: %s", combined_image_path)
        Image.open(combined_image_path).show()
    else:
        logging.warning("未检索到任何图像。")

    milvus_client.release_collection(collection_name=cfg.collection_name)
    milvus_client.drop_collection(cfg.collection_name)
    logging.info("已释放并删除 Collection '%s'", cfg.collection_name)


if __name__ == "__main__":
    main()
