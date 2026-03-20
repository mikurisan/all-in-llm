Hybrid Search 结合了 Sparse Vectors 和 Dense Vectors 的 advantage.

## 1 稀疏 VS 密集

### 1.1 稀疏向量

As known as "lexical vectors”, 是基于 term frequency statistics 的 traditional information retrieval method. 其通常是 high dimensionality (equal to the size of the vocabulary), 但 most dimensions 为 0 的 vector. 其将 documents 视为 words 的 collection, disregard 语义, each dimension 对应一个 word, 非 0 dimension 代表该 word 在 document 中的 weight.

先根据 all documents 计算出 each word 的 weight, 然后就可以给出 query 和 each document的 sparse vector representation, 从而 calculate 二者的 similarity.

该 method 无需 training, 具有 high interpretability, 能够实现 keyword matching. 对于 specialized terms 和 specified nouns 检索效果很好, 但是无法理解 semantic, 识别 synonym.

> 纯单词的字面统计维度, 每个维度就是一个单词.

### 1.2 密集向量

Referred to as “semantic vectors", 是通过 deep learning model 得到的 low-dimensional, dense floating-point representation. In an ideal semantic space, vector 之间的 distance 代表了二者之间的 semantic relationship.

A classic example 是 `vector('king') - vector('男人man') + vector('woman')` 的 calculation reuslt 在 vector space 中非常 proximate to `vector('queen')`.

该 method 能够理解 synonym, near-synonyms 和 contexual relationships, strong generalization capability. However, poor interpretability, 需要 significant computational resources 进行 training.

> 单词的语义维度, 向量的不可解释性.

> 对于未登录词 (OOV, Out-of-Vocabulary), 传统的稀疏向量会完全忽略, 而现代的密集向量能通过子词分割来处理.

## 2 混合检索

### 2.1 技术原理与融合方法

#### 2.1.1 倒数排序融合

Reciprocal Rank Fusion (RRF) 只关心每个 document 在其 result set 中的 rank, rank 越靠前 final score 越高.

#### 2.1.2 加权线性组合

Weighted Linear Combination 将 different retrieval systems 的 scores 归一化, 然后通过总和为 1 的 weighting coefficients 进行 linearly combined. 通过 adjust 权重可以 control 对应 retrieval system 在 final ranking 中的权重.

比如 e-commerce system 可以调高 sparse vector 的 weight.

### 2.2 优势与局限

| 优势	| 局限 |
| -- | -- |
| 召回率与准确率高: 能同时捕获关键词和语义, 显著优于单一检索. |	计算资源消耗大: 需要同时维护和查询两套索引. |
| 灵活性强: 可通过融合策略和权重调整, 适应不同业务场景. |	参数调试复杂: 融合权重等超参数需要反复实验调优. |
| 容错性好: 关键词检索可部分弥补向量模型对拼写错误或罕见词的敏感性 |	可解释性仍是挑战: 融合后的结果排序理由难以直观分析. |

## 代码示例

[使用 Milvus 实现混合检索.](./code/01_hybrid_search.py)

[使用 Milvus 实现混合检索 v2.](./code/02_hybrid_search_v2.py)

[多模态 + 混合检索.](./code/03_work_multimodal_dragon_search.py)

[多模态 + 混合检索 v2.](./code/04_work_hybrid_multimodal_search.py)