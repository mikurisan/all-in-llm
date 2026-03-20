import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
import torch
from PIL import Image
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from visual_bge.modeling import Visualized_BGE

# -----------------------------------------------------------------------------
# 日志与配置
# -----------------------------------------------------------------------------
@dataclass
class Config:
    model_name: str = "BAAI/bge-base-en-v1.5"
    model_path: str = "../../../model/weight/Visualized_base_en_v1.5.pth"
    data_dir: Path = Path("./data/dragon")
    collection_name: str = "multimodal_demo"
    milvus_uri: str = "http://localhost:19530"
    search_query: str = "一条龙"
    search_limit: int = 5
    ef_search: int = 128
    index_params: dict = None

    def __post_init__(self) -> None:
        self.index_params = {"M": 16, "efConstruction": 256}


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

# -----------------------------------------------------------------------------
# 1. 定义工具：编码器
# -----------------------------------------------------------------------------
class Encoder:
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


# -----------------------------------------------------------------------------
# 2. 定义可视化函数：将查询图像与检索结果拼成对比图
# -----------------------------------------------------------------------------
def visualize_results(
    query_image_path: Path,
    retrieved_images: List[Path],
    img_height: int = 300,
    img_width: int = 300,
    row_count: int = 3,
) -> np.ndarray:
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
# 3. 教学示例流程
# -----------------------------------------------------------------------------
def ensure_images_exist(image_dir: Path) -> List[Path]:
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"在 {image_dir} 中未找到任何 .png 图像。")
    return image_paths


# -----------------------------------------------------------------------------
# 4. Collection 管理：重置集合
# -----------------------------------------------------------------------------
def reset_collection(client: MilvusClient, collection_name: str) -> None:
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        logging.warning("检测到同名 Collection，已删除旧的 '%s'", collection_name)


# -----------------------------------------------------------------------------
# 5. Collection 构建：定义 Schema 并创建
# -----------------------------------------------------------------------------
def create_collection(client: MilvusClient, collection_name: str, dim: int) -> None:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
    ]
    schema = CollectionSchema(fields, description="多模态图文检索")
    client.create_collection(collection_name=collection_name, schema=schema)
    logging.info("成功创建 Collection '%s'", collection_name)


# -----------------------------------------------------------------------------
# 6. 数据插入：生成并写入向量
# -----------------------------------------------------------------------------
def insert_embeddings(
    client: MilvusClient,
    encoder: Encoder,
    image_paths: Iterable[Path],
    collection_name: str,
) -> None:
    data = [
        {"vector": encoder.encode_image(str(path)), "image_path": str(path)}
        for path in image_paths
    ]
    result = client.insert(collection_name=collection_name, data=data)
    logging.info("成功插入 %d 条数据。", result["insert_count"])


# -----------------------------------------------------------------------------
# 7. 索引构建：创建 HNSW 并加载集合
# -----------------------------------------------------------------------------
def build_index(client: MilvusClient, collection_name: str, index_params: dict) -> None:
    params = client.prepare_index_params()
    params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params=index_params,
    )
    client.create_index(collection_name=collection_name, index_params=params)
    client.load_collection(collection_name=collection_name)
    logging.info("集合已加载至内存。")


# -----------------------------------------------------------------------------
# 8. 检索：执行相似度搜索
# -----------------------------------------------------------------------------
def search_similar_images(
    client: MilvusClient,
    collection_name: str,
    query_vector: List[float],
    limit: int,
    ef: int,
) -> List[dict]:
    return client.search(
        collection_name=collection_name,
        data=[query_vector],
        output_fields=["image_path"],
        limit=limit,
        search_params={"metric_type": "COSINE", "params": {"ef": ef}},
    )[0]


# -----------------------------------------------------------------------------
# 9. 结果可视化：保存并展示拼图
# -----------------------------------------------------------------------------
def save_visualization(image: np.ndarray, output_path: Path) -> None:
    cv2.imwrite(str(output_path), image)
    logging.info("结果图像已保存到: %s", output_path)
    Image.open(output_path).show()


def main() -> None:
    logging.info("初始化 Encoder 与 Milvus 客户端 ...")
    cfg = Config()
    encoder = Encoder(cfg.model_name, cfg.model_path)
    milvus_client = MilvusClient(uri=cfg.milvus_uri)

    reset_collection(milvus_client, cfg.collection_name)

    image_paths = ensure_images_exist(cfg.data_dir)
    feature_dim = len(encoder.encode_image(str(image_paths[0])))

    create_collection(milvus_client, cfg.collection_name, feature_dim)
    logging.info("正在生成图像嵌入并插入数据 ...")
    insert_embeddings(milvus_client, encoder, image_paths, cfg.collection_name)

    logging.info("正在创建 HNSW 索引 ...")
    build_index(milvus_client, cfg.collection_name, cfg.index_params)

    query_image_path = cfg.data_dir / "query.png"
    query_vector = encoder.encode_query(str(query_image_path), cfg.search_query)
    search_results = search_similar_images(
        milvus_client,
        cfg.collection_name,
        query_vector,
        cfg.search_limit,
        cfg.ef_search,
    )

    retrieved_images = []
    logging.info("检索结果：")
    for rank, hit in enumerate(search_results, start=1):
        img_path = Path(hit["entity"]["image_path"])
        logging.info("Top %d | ID=%s | 距离=%.4f | 路径=%s", rank, hit["id"], hit["distance"], img_path)
        retrieved_images.append(img_path)

    if retrieved_images:
        visualization = visualize_results(query_image_path, retrieved_images)
        save_visualization(visualization, cfg.data_dir / "search_result.png")
    else:
        logging.warning("未检索到任何图像。")

    milvus_client.release_collection(collection_name=cfg.collection_name)
    milvus_client.drop_collection(cfg.collection_name)
    logging.info("已释放并删除 Collection '%s'", cfg.collection_name)


if __name__ == "__main__":
    main()
