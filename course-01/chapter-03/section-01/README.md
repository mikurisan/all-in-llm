## 1 向量嵌入基础

### 1.1 基础概念

#### 1.1.1 什么是 Embedding

将 various data objects 转换为 dense, continuous numerical vector representations.

![](./img/image-01.png)

Consequently, 在一个 high-dimensional vector space 中得到 an object 的 coordinate representation.

The dimensionality 通常 range from hundreds to several thousand.

#### 1.1.2 向量空间的语义表示

Embedding 是对 semantic of data 的 encoding:

**Core Principle**: 在 vector space 中 semantically similar objects 应拥有 close spatial proximity, while 则应该 farther apart.

**Key Metrics**: 通过 following metrics 来表示 vector 之间的 distance 或者 similarity:

  - **Cosine Similarity**: 计算 vector 间 cosine of the angle, 越 close to 1 代表 more aligned direction, greater semantic similarity. (Most commonly used)

  - **Dot Product**: 计算 the sum of the products of corresponding vector components. 当 normalized 后, 等价于 cosine similarity.

  - **Euclidean Distance**: 计算 vector 在 space 中的直线 distance, 越小语义越相似.

> cosine similarity 只关心方向, dot product 关心方向和长度 (投影), euclidean distance 关注绝对距离.

### 1.2 Embedding 在 RAG 中的作用

#### 1.2.1 语义检索的基础

A typical workflow 如下:

- **Offline Index Construction**: Documents 被 split 后, embedding model 将各个 chunk 转换为 vector 存入 dedicated vector database.

- **Similarity Calculation**: 在 vector database 中计算 query vector 与所有 chunk vectors 的 similarity.

- **Context Retrieval**: 选取 similarity 最高的 top-k 个 chunks, 作为 supplementary context 与 original query 一同传给 LLM.

> 让 vector 作为 retrieval 到媒介, 在 query 与 docuemnt 之间建立了 relationship.

#### 1.2.2 决定检索质量的关键

Embedding 决定了 retrieval 的 accuracy 和 relevance.

In the best scenario, embedding 最好能 capture 在 query 与 chunk 之间的 deep semantic relationship, 即使二者的 words 并不完全 match.

In the worst-case scenario, embedding 无法理解 underlying semantic meaning, incorrect or irrelevant content 被 retrieved, 从而 给 context 引入 noise.

> Retrieval 的角度分为字面层面和语义层面, 二者并不 100% 相等的.

## 2 Embedding 技术发展

与 NLP 的 advancements 紧密相连, 尤其当 RAG 出现后, 对其 capabilities 要求 more higher.

### 2.1 静态词嵌入: 上下文无关的表示

- **Representative Models**: Word2Vec (2013), GloVe (2014)

- **Main Principle**: 为 vocabulary 中的每个 word 生成 a fixed, independent of its context 的 vector

- **Limitation**: 无法处理 the issue of polysemy (words with multiple meanings.)

> 通过 training 算好了 word vectors, 要使用的时候直接 lookup 即可.

### 2.2 动态上下文嵌入

**2017 Transformer & Self-Attention**: 使得 word vectors 的生成会受到 the same sentence 中其他 words 的 influence.

**2018 BERT**: 使用 Transformer 中的 encoder 通过 MLM 进行 pre-training, 来生成 deeply contextualized embeddings.

**Impact**: A single word 在不同 context 中有不同的 vector representation, 从而解决 the polysemy problem.

> 将包含 words 的 sentence 进行动态嵌入 (实时推理的), 最后得到每个位置的 word 及其对应的 vector.

### 2.3 RAG 对 Embedding 的新要求

- **Domain Adaptation Capability**: General-purpose embedding models 在 specialized domains 往往 underperform. Threfore, embedding model 具备一定的领域自适应能力. 比如通过 fine-tuning 或 guided instructions 的方式来 align with 特定领域的 terminology and semantic.

- **Multi-granularity and Multimodal Support**: 能够处理 data of varying lengths and types, 例如 long documents, code, images, 甚至是 tabular data.

- **Retrieval Efficiency**: The dimensionality of embedding vectors 和 the size of the model 直接影响 storage costs 和 retrieval speed.

- **Hybrid Retrieval**: 为了 leverage 语义 similarity (dense retrieval) 和 keyword matching (sparse retrieval) 二者的 strengths, 支持 hybrid retrieval 的 model 应运而生.

## 3 嵌入模型训练原理

现代 embedding model 的核心通常是 transformer 的 encoder 部分.

### 3.1 主要训练任务

Primarily 为 self-supervised learning, 允许 model 从 vast amounts of unlabled text data 中 internalize knowledge.

**Task 1: Masked Language Model (MLM)**

Procedure: 

- Randomly 将 input sentences 中的 15% 的 tokens 替换为 a special `[MASK]` token.

- 让 model 去 predict 被 masked 的 original token 是什么.

Objective: 让 model 学习每个 tokens 与其 context 间的 relationship, 从而 grasp 深层次的 contextual semantics.

**任务2: Next Sentence Prediction (NSP)**

Procedure:

- Construct 训练 samples, 每个 sample 包含 (Sentence A 和 Sentence B).

- 其中 50% 的 samples, Sentence B 是 actual next sentence (IsNext); 另外 50% 则是从 corpus 中randomly 抽取的 (NotNext).

Objective: 让 model 学习 sentence 之间的 logical relationships, coherence 和 topical relevance.

**Important Note**: NSP 在 subsequent research 中被found 可能 too simplistic, 甚至损坏 model 的 performance. Consequently, many modern embedding model 在 pre-training stage 移除了 NSP.

> 了解更多细节: [BERT 架构及其应用](https://github.com/datawhalechina/base-llm/blob/main/docs/chapter5/13_Bert.md)

> 从 word 和 sentence 的层面分别进行 learning.

### 3.2 效果增强策略

现代 embedding model 通常会引入 specifically tailored training strategies 以 enhance 在 retrieval task 中的 performance.

**Metric Learning**

- **Idea**: Directly 以优化 similarity 为 target.

- **Method**: 用 large-scale relevant text pairs (e.g., question-answer, title-content) 训练, 以optimize 向量空间中的 relative distances. 让 positive pairs 在 space 中拉近, negative pairs 推远. 这里并非 pursuing 绝对的 similarity, 因为 extreme values 可能导致 overfitting.

**Contrastive Learning**

- **Idea**: 在 vector space 中将 similar samples 拉近, dissimilar samples 推远.

- **Method**: 构建 triplets (Anchor, Positive, Negative), 其中 anchor 与 positive 是 relevant 的, 与 negative 是 irrelevant 的. Trainning objective 是让 distance(anchor, positive) 尽量小, distance(anchor, negative) 尽量大.

> 针对性地增强两个相关 part 之间的 similarity.

## 4 嵌入模型选型指南

### 4.1 从 [MTEB](https://huggingface.co/spaces/mteb/leaderboard) rankings 开始

![](./img/image-02.png)

MTEB (Massive Text Embedding Benchmark) 是由 Hugging Face 维护的, a comprehensive evaluation benchmark for embedding models. 涵盖了 classification, clustering, retrieval and reranking.

![](./img/image-03.png)

The char above 提供了 open-source embedding model 需要权衡的 4 个 key dimensions:

- **Horizontal Axis (Number of Parameters)**: 代表 model size, 越大 potential capability 越强, 也需要更多的 computational resources.

- **Vertical Axis (Mean Task Score)**: 代表 model 的 overall performance, 越大说明 model 在各种 NLP task 上的 average performance 更佳, general semantic understanding capability 更强.

- **Bubble Size (Embedding Size)**: 代表 model 输出 vector 的 dimensionality, higer dimensionality 理论上能 encode 更丰富的 semantic details, 当然也更消耗 computational and storage resources.

- **Bubble Color (Max Tokens)**: 代表 model 能 process 的 maximum text length.

**Importang Note**: Benchmark 的 score 是在 general-purpose datasets 上 evaluated, 可能无法 reflect 在 specific business scenarios 的 performance.

### 4.2 关键评估维度

Additional dimension to consider:

- **Task**: 对于 RAG 更需关注 model 在 retrieval task 下的 ranking.

- **Language**: 有些 model 并不是 all language 都 support 的.

- **Score & Publisher**: 考虑 model 的 score 与 publisher 的 reputation.

- **Cost**: 调用 api 考虑 price, self-hosting 考虑 hardware resouce consumpation 和 maintenance and operational cost.

### 4.3 迭代测试与优化

不要 rely solely on 公开 leaderboards.

1. **Establish a Baseline**: 根据 aforementioned criteria 选择 several candidate models 作为 initial benchmarks.

2. **Build a Private Evaluation Dataset**: 根据 authentic business data, create 一批 high-quality QA samples.

3. **Iterate and Optimize**: 使用 baseline models 在 dataset 运行, 进行一定的 RAG 调优, 评估其 recall 的 accuracy 和 relevance, ultimately 选择 optimal model.

> 根据业务和数据选择适合的 model, 并尽量考虑 cost.

## 参考文献

[Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.](https://arxiv.org/abs/2005.11401)

[RoBERTa: A Modified BERT Model for NLP.](https://www.comet.com/site/blog/roberta-a-modified-bert-model-for-nlp/)