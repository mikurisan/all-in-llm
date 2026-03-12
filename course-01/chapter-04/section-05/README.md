基础的 RAG 依靠的是 similarity 进行检索, 其局限性在于, 最相关的 document 并不总是排名靠前, 或者语义的理解有偏差等. 因而这里引入更高级的检索技术.

## 1 重排序 (Re-ranking)

### 1.1 RRF (Reciprocal Rank Fusion)

在混合检索中已接触过, 如果一个 document 在多个检索结果中都排名靠前, 那么它可能就越重要. RRF 通过计算排名的倒数来为 document 打分, 融合来自不同检索策略的优势.

只考虑排名信息, 会忽略原本的 similarity score, 丢失部分参考信息.

### 1.2 RankLLM / LLM-based Reranker

既然 LLM 负责生成 answer, 就直接让 LLM 来判断哪些 documents 最相关.

该方法在 prompt 包含 query 与一系列 document 的 summary, 然后让 LLM 以特定格式输出一组排序后的 document list, 并给出相关性分数.

例如:

```md
以下是一个文档列表，每个文档都有一个编号和摘要。同时提供一个问题。请根据问题，按相关性顺序列出您认为需要查阅的文档编号，并给出相关性分数（1-10分）。请不要包含与问题无关的文档。

示例格式:
文档 1: <文档1的摘要>
文档 2: <文档2的摘要>
...
文档 10: <文档10的摘要>

问题: <用户的问题>

回答:
Doc: 9, Relevance: 7
Doc: 3, Relevance: 4
Doc: 7, Relevance: 3
```

### 1.3 Cross-Encoder 重排

通过 Cross-Encoder 给出 query 与 document 之间的相关性. 其原理是将 query 与 document 拼接为一个单一 input, 然后将其给到一个 pre-trained transformer model, model 最终会输出一个 0-1 之间的 score 表示二者之间的 similarity.

该方式的工作流程为:

1. 初步检索: search engine 从知识库中召回初始 document list.

2. 逐一评分: 将每个 document 与原始 query 配对, 发送给 Cross-Encoder model.

3. 独立推理: 对每个 document-query pair 进行一次完整, 独立的计算, 得到一个精确的 similarity score.

4. 返回重排结果: 根据 score 对 document list 进行重排, 返回重排后的结果.

可以看出其特点为高精度, 高延迟.

### 1.4 ColBERT 重排

ColBERT (Contextualized Late Interaction over BERT) 在 Cross-Encoder 的高精度和双编码器 (Bi-Encoder) 的高效率之间取得平衡, 采用了一种后期交互机制.

> 这里的 Bi-Encoder 第一次提, 简单理解为将 query 和 document 分别压缩为 vector, 然后计算 similarity.

其工作流程如下:

- 独立编码: 分别为 query 和 document 中的每个 token 生成 context 相关的 embedding vector. 该步骤是独立完成的, 可预先计算并存储 document 的 vector, 加快查询速度.

- 后期交互: search 时 model 会计算 query 中每个 token 与 document 中每个 token 之间的 max similarity.

- 分数聚合: 将 query 中所有 tokens 得到的 max similarity 相加得到最终的 similarity score.

这样 ColBERT 避免了 query-document 拼接的昂贵编码, 又比单纯比较单个 `[CLS]` vector 捕捉了更细粒度的词汇级交互信息.

> `[CLS]` 可以理解为代表了整个 sentence 语义的单一 vector.

### 1.5 重排方法对比

| 特性 |	RRF	| RankLLM |	Cross-Encoder |	ColBERT |
| --- | --- | --- | --- | --- |
|核心机制 |	融合多个排名 |	LLM 推理，生成排序列表	| 联合编码查询与文档，计算单一相关 分	 | 独立编码，后期交互 |
| 计算成本 |	低 (简单数学计算)|	中 (API 费用与延迟) |	高 (N次模型推理) |	中(向量点积计算) |
| 交互粒度 |	无 (仅排名) |	概念/语义级	| 句子级 (Query-Doc Pair) |	Token 级 |
| 适用场景 |	多路召回结果融合 |	高价值语义理解场景 |	Top-K 精排 |	Top-K 重排 |

## 2 压缩 (Compression)

旨在解决: 初步检索到的 chunks 可能包含大量无关的 text.

压缩的目的是“压缩”与“提炼”, 尽可能保留与用户 query 最相关的 text. 可以通过 2 种方式实现:

- 内容提取: 从 document 中抽出与 query 相关的 content.

- 文档过滤: 完全丢弃经过精细判断后, 认为不相关的整个 document.

### 2.1 LangChain 的 ContextualCompressionRetriever

该组件用于包装基础 retriever, 当基础 retriever 返回 documents 之后, 其会使用一个指定的 `DocumentCompressor` 对这些 documents 进行处理再返回给调用者.

`DocumentCompressor` 有多种类型:

- `LLMChainExtractor` : 遍历 documents, 用 LLM 来判断并提取出与 query 相关的 content.

- `LLMChainFilter` : 遍历 documents, 用 LLM 判断整个 document 是否与 query 相关; 相关则保留, 否则直接丢弃.

- `EmbeddingFilter` : 计算 query 每个 document 之间的 vector similarity, 只保留超过阈值的 documents.

### 2.2 自定义重排器与压缩管道

参阅 LangChain 的 `ContextualCompressionRetriever` 源码, 一层层分析, 自定义实现一个符合其标的 ColBERT 重排器.

### 代码示例

[按照 LangChain 标准自定义重排器与压缩管道.](./code/01_rerank_and_refine.py)

### 2.3 LlamaIndex 中的检索压缩

代表性的是 `SentenceEmbeddingOptimizer`, 也是一个后处理器 (Node Postprocessor), 工作在检索之后.

其工作原理为, 将 document 分割为 sentences, 然后计算每个 sentence 与 query 之间的 similarity, 最后只保留超过阈值的 sentences.

## 3 校正 (Correcting)

传统 RAG 有一个隐含假设: 检索到的 document 总是包含与 query 相关的 content. 现实中, 检索系统可能会失败, 会返回不相关, 过时甚至完全错误的 document.

校正检索 (Corrective-RAG, C-RAG) 引入一个“自我校正”的循环, 对检索到的 document 质量进行评估, 根据评估结果采取不同的行动.

C-RAG 的工作流分为:

![alt text](./img/image01.png)

1. 检索 (Retrieve): 与传统 RAG 一样根据 query 检索出 documents.

2. 评估 (Assess): 通过 Retrieval Evaluator 判断每个 document 与 query 的相关性, 并给出 correct, incorrect 或 ambiguous 的 tag.

3. 行动 (Act): 根据评估结果, 进入不同的修正流程:

    - correct: 进入 knowledge refinement 环节. 将 document 分解为更小的知识片段 (strips), filter 掉无关部分, 重新组合为更精准聚焦的 context.

    - incorrect: 触发 knowledge searching. 对 query 进行 rewriting, 生成一个更适合 search engine 对 query, 然后进行 web search, 获取外部信息.

    - ambiguous: 同时触发 knowledge searching, 但通常使用原始 query 进行 web search, 获取外部信息.

> 在 LangChain 的 `langgraph` 库中, 可以使用其图结构来灵活构建这种带有条件判断和循环的复杂 RAG 流程.

## 参考代码

2.1 中“按照 LangChain 标准自定义重排器与压缩管道.”的代码运行后可能会出现重复的情况.

[对“按照 LangChain 标准自定义重排器与压缩管道”进行修正.]

## 参考文献

[Using LLM’s for Retrieval and Reranking.](https://www.llamaindex.ai/blog/using-llms-for-retrieval-and-reranking-23cf2d3a14b6)

[Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT.](https://arxiv.org/abs/1901.04085)

[Advanced RAG: ColBERT Reranker.](https://www.pondhouse-data.com/blog/advanced-rag-colbert-reranker)

[How to do retrieval with contextual compression.](https://docs.langchain.com/oss/python/langchain/overview)

[Sentence Embedding Optimizer.](https://developers.llamaindex.ai/python/examples/node_postprocessor/OptimizerDemo/)

[Jiang, Z. et al. (2024). Corrective Retrieval Augmented Generation.](https://arxiv.org/pdf/2401.15884)

[Corrective-RAG (CRAG).](https://docs.langchain.com/oss/python/langgraph/agentic-rag)