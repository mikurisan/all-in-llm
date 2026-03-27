Basic RAG 依靠的是 similarity 进行 retrieval, 其 limitation 是 the most relevant document 并不总是 rank 靠前, 或者 semantic understanding 有偏差等.

## 1 重排序 (Re-ranking)

### 1.1 RRF (Reciprocal Rank Fusion)

如果一个 document 在 multiple retrieval sets 中都 rank 靠前, 那么它 likely more significant. RRF 通过计算 reciprocal rank 来为 document 打分, leveraging 来自 various retrieval strategies 的 strength.

只考虑了 rank information, 会 disregard 原本的 similarity score.

> 以不同 strategies 中 document 的 rank 为依据.

### 1.2 RankLLM / LLM-based Reranker

既然 LLM 负责生成 answer, 就直接让 LLM 来 determine 哪些 documents 是 relevant.

该方法在 prompt 包含 query 与一系列 document 的 summary, 然后让 LLM 以 specific format 输出一组 reordered document list, 并给出 relevance scores.

> 让 LLM itself 作评分.

### 1.3 Cross-Encoder 重排

通过 Cross-Encoder 给出 query 与 document 之间的 relevance. 其 principle 是将 query 与 document 拼接为 a single input, 然后将其 fed into 一个 pre-trained transformer model, model 最终会输出一个 0-1 之间的 score 表示二者之间的 similarity.

The workflow:

1. **Initial Retrieval**: search engine 从 knowledge base 召回 initial document list.

2. **Indivisual Scoring**: 将每个 document 与原始 query 配对, 发送给 Cross-Encoder model.

3. **Independent Inference**: 对每个 document-query pair 进行一次 complete, independent 计算, 得到 a precise similarity score.

4. **Return Re-ranked Results**: 根据 score 对 document list 进行 re-rank, 返回 re-ranked results.

可以看出其 characteristic 为 high precision, high latency.

> 引入了 external specific model 用于实时计算 query-docuemnt pair 的 similarity.

### 1.4 ColBERT 重排

ColBERT (Contextualized Late Interaction over BERT) 在 Cross-Encoder 的 high accuracy 和 Bi-Encoder 的 high efficiency 之间 strike a balance, 采用了 a late interaction mechanism.

> Bi-Encoder 可分别将 query 和 document 压缩为 vector, 然后计算二者 similarity, document vector 可以提前计算存储.

The workflow:

- **Independent Encoding**: 分别为 query 和 document 中的每个 token 生成 context 相关的 embedding vector. Document vector 可以 precompute and store 以 accelerate 处理速度.

- **Late Interaction**: Search 时 model 会计算 query 中每个 token 与 document 中每个 token 之间的 max similarity.

- **Score Aggregation**: 将 query 中所有 tokens 得到的 max similarity 相加得到最终的 similarity score.

这样 ColBERT 避免了 query-document 拼接的 costly joint encoding, 又比 simply 比较 compressed document vector 捕捉了 finer-grained lexical-level interaction information.

> 计算 token 间的 similarity 比跑 model 计算快得多.

> 综上的方式都是为了计算 query 与 document 之间的 similarity, 最精确的是 query-document 作为 pair 实时计算, 其次是将 query 单独拉出来, 对 document vecort 进行独立处理, 处理的方式有比较精细的 each token vector 计算, 或者直接整个 document 级的 vector 计算.

### 1.5 重排方法对比

| 特性 |	RRF	| RankLLM |	Cross-Encoder |	ColBERT |
| --- | --- | --- | --- | --- |
|核心机制 |	融合多个排名 |	LLM 推理，生成排序列表	| 联合编码查询与文档，计算单一相关 分	 | 独立编码，后期交互 |
| 计算成本 |	低 (简单数学计算)|	中 (API 费用与延迟) |	高 (N次模型推理) |	中(向量点积计算) |
| 交互粒度 |	无 (仅排名) |	概念/语义级	| 句子级 (Query-Doc Pair) |	Token 级 |
| 适用场景 |	多路召回结果融合 |	高价值语义理解场景 |	Top-K 精排 |	Top-K 重排 |

## 2 压缩 (Compression)

Aims to address: 初步 retrieved chunks 可能 contain 大量 irrelevant text.

The purpose of compression 是 “compress” 与 “refine”, 尽可能 retain 与用户 query 最相关的 text. 可通过 2 种方式 achive:

- **Content Extraction**: 从 document 中 extract 与 query 相关的 content.

- **Document Filtering**: 经过 fine-grained assessment 后, 完全 discard 不相关的 entire document.

### 2.1 LangChain 的 ContextualCompressionRetriever

该 component 用于 wrap 基础 retriever, 当基础 retriever 返回 documents 之后, 会使用 a specified `DocumentCompressor` 对这些 documents 进行 processing.

`DocumentCompressor` 有 various types:

- `LLMChainExtractor` : 遍历 documents, 用 LLM 来 assess 并 extract 与 query 相关的 content.

- `LLMChainFilter` : 遍历 documents, 用 LLM 判断整个 document 是否与 query 相关; 相关则 retain, 否则直接 discard.

- `EmbeddingFilter` : 计算 query 每个 document 之间的 vector similarity, 只保留超过 threshold 的 documents.

### 2.2 自定义重排器与压缩管道

参考 `ContextualCompressionRetriever`, 进行 a layer-by-layer source code analysis, 实现 a custom, 符合其 standard 的 ColBERT 重排器.

### 示例代码

[按照 LangChain 标准自定义重排器与压缩管道.](./code/01_rerank_and_refine.py)

### 2.3 LlamaIndex 中的检索压缩

A representative example 是 `SentenceEmbeddingOptimizer`, 是一个 Node Postprocessor, 工作在 retrieval 之后.

其 work principle: 将 document 分割为 sentences, 然后 calculate 每个 sentence 与 query 之间的 similarity, 最后 retain 超过 threshold 的 sentences.

> 该部分的工作是在 retrieval 之后的, 本质是对 retrieved documents 进行 refine.

## 3 校正 (Correcting)

Traditional RAG 有 an implicit assumption: retrieved documents 总是 contain 与 query 相关的 content. In reality, retrieval system 可能会 fail, 会返回 irrelevant, outdated, or even entirely incorrect documents.

Corrective-RAG (C-RAG) 引入一个 “self-correction” loop, 对 the quality of retrieved document 进行 evaluating, 根据评估结果采取不同的行动.

C-RAG 的工作流分为:

![alt text](./img/image01.png)

1. **Retrieve**: 与 traditional RAG 一样根据 query 检索出 documents.

2. **Assess**: 通过 Retrieval Evaluator 判断 each document 与 query 的 relevance, 并给出 correct, incorrect 或 ambiguous 的 tag.

3. **Act**: 根据 assessment results, 进入 different corrective processes:

    - correct: 进入 knowledge refinement phase. 将 document 分解为 smaller knowledge fragments (strips), filter 掉 irrelevant parts, reassemble 为 more precise and focused context.

    - incorrect: 触发 knowledge searching. 对 query 进行 rewriting, 生成 a more suitable search engine 的 query, 然后进行 web search, 获取 external information.

    - ambiguous: 同时触发 knowledge searching, 但 typically 使用 original query 进行 web search, 获取 external information.

> 根据 retrieved document 的正确度, 使用 knowledge base 或 external information 对其进行 enhancement.

> 在 LangChain 的 `langgraph` 库中, 可以使用其图结构来灵活构建这种带有条件判断和循环的复杂 RAG 流程.

## 参考代码

2.1 中“按照 LangChain 标准自定义重排器与压缩管道.”的代码运行后可能会出现重复的情况.

[对“按照 LangChain 标准自定义重排器与压缩管道”进行修正.](./code/02_work_rerank_and_refine.py)

## 参考文献

[Using LLM’s for Retrieval and Reranking.](https://www.llamaindex.ai/blog/using-llms-for-retrieval-and-reranking-23cf2d3a14b6)

[Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT.](https://arxiv.org/abs/1901.04085)

[Advanced RAG: ColBERT Reranker.](https://www.pondhouse-data.com/blog/advanced-rag-colbert-reranker)

[How to do retrieval with contextual compression.](https://docs.langchain.com/oss/python/langchain/overview)

[Sentence Embedding Optimizer.](https://developers.llamaindex.ai/python/examples/node_postprocessor/OptimizerDemo/)

[Jiang, Z. et al. (2024). Corrective Retrieval Augmented Generation.](https://arxiv.org/pdf/2401.15884)

[Corrective-RAG (CRAG).](https://docs.langchain.com/oss/python/langgraph/agentic-rag)