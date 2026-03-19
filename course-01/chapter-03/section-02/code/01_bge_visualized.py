import torch
from visual_bge.visual_bge.modeling import Visualized_BGE


# 加载 model 和 weight
model = Visualized_BGE(
    model_name_bge="BAAI/bge-base-en-v1.5", 
    model_weight="./model/Visualized_base_en_v1.5.pth"
)

# 将 model 切换到推测/测试模式
model.eval()

# 不再计算 gradient 用于反向传播
with torch.no_grad():
    text_emb = model.encode(text="datawhale开源组织的logo")
    img_emb_1 = model.encode(image="./data/datawhale01.png")
    multi_emb_1 = model.encode(image="./data/datawhale01.png", text="datawhale开源组织的logo")
    img_emb_2 = model.encode(image="./data/datawhale02.png")
    multi_emb_2 = model.encode(image="./data/datawhale02.png", text="datawhale开源组织的logo")

# 相似度计算(点积): 使用矩阵乘法计算余弦相似度, 所有 embedded vector 都被标准化
sim_1 = img_emb_1 @ img_emb_2.T
sim_2 = img_emb_1 @ multi_emb_1.T
sim_3 = text_emb @ multi_emb_1.T
sim_4 = multi_emb_1 @ multi_emb_2.T

print("=== 相似度计算结果 ===")
print(f"纯图像 vs 纯图像: {sim_1}")
print(f"图文结合1 vs 纯图像: {sim_2}")
print(f"图文结合1 vs 纯文本: {sim_3}")
print(f"图文结合1 vs 图文结合2: {sim_4}")

# 向量信息分析
print("\n=== 嵌入向量信息 ===")
print(f"多模态向量维度: {multi_emb_1.shape}")
print(f"图像向量维度: {img_emb_1.shape}")
print(f"多模态向量示例 (前10个元素): {multi_emb_1[0][:10]}")
print(f"图像向量示例 (前10个元素):   {img_emb_1[0][:10]}")