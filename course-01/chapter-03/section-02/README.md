## 1 为什么需要多模态嵌入

Text 和 images 的 vector spaces 之间是 inherently disconnected. Multimodal embedding 是为了将二者 map 到同一个 shared vector space.

Achieving 这一目标需要 cross-modal alignment.

## 2 CLIP 模型浅析

以奠定基础的 OpenAI 的 CLIP (Contrasive Language-Image Pre-training) 为例.

其采用 dual-encoder architecture, 分别为一个 image encoder 和 text encoder, 将 image 和 text 映射到一个 shared vector space.

![alt text](./img/image.png)

CLIP 在 training 时采用了 contrastive learning: Maximize 正确 text-image pair 的 vector similarity, minimize 错误 text-image pair 的 similarity.

> 正如 image 中 (1) 所示的 the blue diagonal line 就是 correct text-image pair.

通过 large-scale training 使得 CLIP 获得了 zero-shot recognition capability.

## 3 常用多模态嵌入模型

随着 evolution of field, emerged 许多针对 different objectives 和 scenarios 优化的 multimodal embedding model.

例如 bge-visualized-m3, 其 core features 有:

- **Multi-Linguality**: 支持 100 多种 languages.

- **Multi-Functionality**: Text retrieval scenarios 下提供 dense retrieval 和 multi-vector retrieval paradigms.

- **Multi-Granularity**: Text length 最高达 8192 tokens.

*Its technical architecture is omitted here.*

## 参考代码

[使用 `Visualized_BGE` 模型进行多模态嵌入.](./code/01_bge_visualized.py)