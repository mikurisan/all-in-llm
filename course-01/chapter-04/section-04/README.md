用户的 query 往往是不够精确的, 因而可以对 query 进行预处理, 这就是 query 重构与分发.

主要包括两个关键技术:

1. 查询翻译 (Query Translation): 将 query 转换为更合适的检索形式.

2. 查询路由 (Query Routing): 根据 query 的性质, 将其分发给更适合的数据源或检索器.

## 1 查询翻译

弥补 query 与 document 之间的语义鸿沟.

### 1.1 提示词工程

通过 prompt 将 query 改写得更清晰, 具体, 更利于检索的格式.

### 示例代码

[通过 prompting 改写 query. (对 c4s2 的 improve)](./code/01_text_to_metadata_filter_v2.py)

### 1.2 多查询分解 (Multi-query)

当一个 query 较为复杂时, 可以将其拆解为多个 sub-query, 然后分别进行检索, 最后汇总检索的结果.

### 1.3 退步提示 (Step-Back Prompting)

当面对一个细节繁多或过于复杂的 query, LLM 直接回答往往容易出错, 这时可以引导模型 step-back.

其核心流程分为:

1. 抽象化: 引导 LLM 从 query 中生成一个更高层次, 更概括的 step-back question. 该 question 旨在探索原始 query 别后抽象层次更高, 更通用的原理或核心概念.

2. 推理: 先获取 step-back question 的 answer, 随后将其作为 context 并结合原始 query 传递给 LLM.

![alt text](./img/image01.png)

### 1.4 假设性文档嵌入 (Hypothetical Document Embeddings, HyDE)

HyDE 旨在解决: 用户的 query 往往简短, 关键词有限, 而 docuemnt 又详实而丰富, 二者在语义 vector space 中可能存在鸿沟, 导致检索效果不佳.

其工作流可分为:

1. 生成: 调用 LLM 根据 query 生成一个详细的, 可能是 answer 的 docuemnt. 该 document 无需完全符合事实, 但是在语义上要与一个好的 answer 高度相关.

2. 编码: 将生成的 document 输入给对比编码器 (如 Contriever), 将其转化为高维 vector embedding, 其在语义上代表了一个理想答案的 position.

> 这里需要去理解一下对比编码器, 才知道这里使用它的原因, 简单的说对比编码器关注的是句子维度的 embedding.

3. 检索: 使用该 document 的 vector 在 db 中检索, 找到最接近的真实的 document 作为 context.

通过上述过程, 将 query 与 document 匹配, 转换为 document 与 document 匹配, 从而提升了检索的准确率.

## 2 查询路由

当系统接入了多个数据源或拥有多种不同的处理路径时, 通过智能路由替代硬编码规则, 从语义层度将 query 分发给最佳的下游处理.

### 2.1 应用场景

1. 数据源路由: 根据 query 的意图, 将其 route 到不同的知识库.

2. 组件路由: 根据 query 的复杂程度, 将其分配给不同的处理组件, 以平衡成本和效果.

3. 提示模板路由: 根据 query 的类型, 将其分配给最优的提示词模板, 以优化生成效果.

### 2.2 实现方法

有 2 种主流的方法.

#### 2.2.1 基于 LLM 的意图识别

设计一个包含 route options 的 prompt, 让 LLM 根据 query 进行分类并输出一个 route option.

![alt text](./img/image02.png)

#### 代码示例

[菜谱问答, 根据菜系调用不同的专家模型. ](./code/02_llm_based_routing.py)

#### 2.2.2 嵌入相似性路由

通过计算 query 与预设的 route 示例语句之间的 similarity 来做出决策.

![alt text](./img/image03.png)

#### 代码示例

[菜谱问答, 根据菜系调用不同的专家模型. ](./code/03_embedding_based_routing.py)

### 2.3 LlamaIndex 拓展

其思路是将不同的 data source 或 query strategy 包装为 tool, 然后通过 router 进行动态选择:

- 基于 LLM 的意图识别: 每个 tool 包含一个 query engine 和一段描述其功能的 description, router 利用 selector 来让 LLM 根据 user query 与 description 进行语义匹配, 选择对应的 tools.

- 嵌入相似性路由: 没有直接提供基于 similarity 计算的独立 route 组件.

## 参考文献

[How to use the MultiQueryRetriever.](https://docs.langchain.com/oss/python/langchain/overview)

[Zheng, H. S. et al. (2023). Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models.](https://arxiv.org/abs/2310.06117)

[Gao, L. et al. (2022). Precise Zero-Shot Dense Retrieval without Relevance Labels.](https://arxiv.org/abs/2212.10496)

[使用假设性文档嵌入（HyDE）改进信息检索和 RAG.](https://zilliz.com.cn/blog/improve-rag-and-information-retrieval-with-hyde-hypothetical-document-embeddings)

[How to route between sub-chains.](https://docs.langchain.com/oss/python/langchain/overview)

[LangChain Expression Language.](https://docs.langchain.com/oss/python/langchain/overview)

[LlamaIndex Routing.](https://developers.llamaindex.ai/python/framework/module_guides/querying/router/)