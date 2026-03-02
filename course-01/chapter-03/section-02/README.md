## 1 为什么需要多模态嵌入

Text 和 image 的 vector space 维度不同, 而多模态嵌入(Multimodal Embedding)是为了将二者 map 到同一个共享的 vector space.

实现这一目标需要跨模态对齐(Cross-modal Alignment).

## 2 CLIP 模型浅析

以奠定基础的 OpenAI 的 CLIP (Contrasive Language-Image Pre-training) 为例.

其采用 Dual-Encoder Architecture, 分别为一个 image encoder 和 text encoder, 将 image 和 text 映射到一个共享的 vector space.

![alt text](./img/image.png)

CLIP 在 training 时采用了对比学习(Contrastive Learning): 最大化正确 text-image pair 的 vector similarity, 最小化所有错误 text-image pair 的 similarity.

> 正如 image 中 (1) 所示的蓝色对角线就是正确的 text-image pair.

通过大规模的 training 使得 CLIP 获得了零样本(Zero-shot) 识别能力.

## 3 常用多模态嵌入模型

随着领域发展, 涌现了许多针对不同目标和场景优化的 Multimodal embedding model.

例如 bge-visualized-m3, 其核心特性有:

- 多语言性(Multi-Linguality): 支持 100 多种 languages.

- 多功能性(Multi-Functionality): 文本检索场景下提供密集检索(Dense Retrieval), 多向量检索(Multi-Vector Retrieval) 等范式.

- 多粒度性(Multi-Granularity): 最高支持 8192 toekn 长度的 text.

技术架构略过.