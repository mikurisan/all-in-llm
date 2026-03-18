## 1 什么是 RAG ?

### 1.1 核心定义

RAG(Retrieval-Augmented Generation), 即通过 retrieve 外部 knowledge 以增强 LLM 的 factual accuracy. 其中:

- **Prametric Knowledge**: LLM 被 trained 而固化的 weights.

- **Non-Parametric Knowledge**: 在 training step 中 LLM 没有接触过的 knowledge, 也就是上述所说的 external knowledge.

### 1.2 技术原理

**Retrieval Step**:

- **Knowledge Vectorization**: 通过 Embedding Model 将外部知识 encoded 为向量索引 (index) 存入 vector db.
- **Semantic Retrieval**: 通过同样的 Embedding Model 将 query 向量化, 通过 similarity search 从 db 中找到与 query 最 relevant 的 knowledge snippets.

**Generation Step**:

- 将 query 与 knowledge 按照 predefined prompt template 进行 integration, 引导 LLM output.

### 1.3 技术演进分类

| | Naive RAG | Advanced RAG | Modular RAG |
| -- | -- | -- | -- |
| 流程 | 离线: 索引 <br> 在线: 检索 -> 生成 | 离线: 索引 <br> 在线: ... -> 检索前 -> ... -> 检索后 -> ... | 积木式可编排流程 |
| 特点 | 基础线性流程 | 增加检索前后的优化步骤 | 模块化, 可组合, 可动态调整 |
| 关键技术 | 基础向量检索 | 查询重写 (Query Rewrite) <br> 结果重排 (Rerank) | 动态路由 (Routing) <br> 查询转换 (Query Transformation) <br> 多路融合 (Fusion) | 
| 局限性 | 效果不稳定, 难以优化 | 流程相对固定, 优化点有限 | 系统复杂性高 |

> 上述部分简单了解, 后序会熟悉.

## 2 为什么使用 RAG ?

### 2.1 RAG VS 微调

从 cost-benefit perspective 考虑: Prompt Engineering -> RAG -> Fine-tuning.

Based on 以下 scenarios 考虑:

- **Prompt Enginnering**: 引导 LLM output, 适用于 simple tasks, 模型已 possesses 相关 knowledge.

- **RAG**: LLM 缺乏 specific knowledge.

- **Fine-tuning**: 改变 LLM 如何 operates, 如 behavior, style, 或 output format.

最 powerful 的是 hybrid method, 即将 RAG 用于 Fine-tuning.

> LLM 通过 training 具备基础的 semantic comprehension capability 后, 可以基于 external knowledge 进行 answer.

RAG 解决了 LLM 的几个 limitations:

- **Static Knowledge** Constraint: 实时检索 external knowledge

- **幻觉 (Hallucination)**: 基于 retrieved knowledge, 降低 error rate.

- **Domain Expertise Gap**: 引入 domain-specific knowledge.

- **Data Privacy Risks**: 本地化部署 knowledge bases.

> 核心就是使得 knowledge 从 static 变为 dynamic 而无需 fine-tuning.

### 2.2 关键优势

1. **Accuracy & Credibility**: Knowledge augments 以 reduce 幻觉, 使得 answer 具备 traceability.

2. **Timeliness**: Dynamic knowledge update, 解决 the issue of knowledge latency, 这也 referred 为“索引热拔插“ (Index Hot-swapping).

3. **Significant Cost-effectiveness**: Avoid 高频 fine-tuning 的 substantial computational costs; 对于 domain-specific taks, 也可以 smaller LLM 来达到 comparable results.

4. **Modular Scalability**: Knowledge source 是diverse 和 modular, 与 retrieval component 是 decoupled.

### 2.3 适用场景的风险分级

| 风险等级 | 案例 | 适用性 |
| -- | -- | -- |
| 低 | 翻译/语法检查 | 高可靠 |
| 中 | 合同起草/法律咨询 | 需结合人工审核 |
| 高 | 证据分析/签证决策 | 需严格控制质量 |

## 3 如何上手 RAG ?

### 3.1 基础工具链选择

- **Established Frameworks**: LangChain, LlamaIndex; 当然也可以 native development.

- **Vector DB**: 大型用 Milvus, Pinecone, 轻量用 FAISS, Chroma

- **Evaluation Tools**: RAGAS, TruLens

### 3.2 四步构建最小可行系统 (MVP)

1. **Data Preparation & Cleansing**: Standardize 多源异构 data sources, 采用 chunking strategies 对 text 进行 segmentation.

2. **Index Construction**: 通过 embedding model 将 chunk 向量化并存入 vector db, 在此 stage 可以关联 metadata.

3. **Retrieval Strategy Optimization**: 可采用 hybrid retrieval methods, 并引入 re-ranking 优化 retrieval outcomes.

4. **Answer & Prompt Enginnering**: 引导 LLM 基于 retrieved content 给出 target answer.

> 准备数据 -> 存入 vector db -> 检索 -> 生成 answer.

### 3.3 新手友好方案

希望 quickly 验证 ideas, consider:

- **Visual Knowledge Base Platforms**: FastGPT, Dify

- **Open-soure Template**: LangChain4j, TinyRAG

### 3.4 进阶与挑战

**Dimensions and Challenges of Evaluation**

首先是 retrieval relevance,  其次是 generation quality, 这又可细分为 semantic accuracy (回答的意思是否正确)和 lexical appropriateness (专业术语是否使用得当).

如果 retrieve 到 erroneous knowledge, LLM 就会输出 nonsensical answer. Furthemore, cross-document multi-hop reasoning 也是一项 significant challenges.

**Optimization Direction and Architectural Evolution**

Performance enhancement 可通过 index stratification 和 multimodal extension (支持图像和表格检索).

Architecture 方面有 more complex的 design patterns. 例如可以通过 branching pattern 并行处理 multi-path retrieval, 或通过 looping pattern 进行 self-correction.

> 像是在说 RAG 的弱点以及 some 对应的 enhancement, 有个印象即可, 后续会提及.

## 4 RAG 已死

No, RAG 现已成为 LLM development 的一种 fundamental paradigm, 并正在 rapidly evolving.

## 参考代码

[使用 LangChain 实现一个 RAG demo.](./code/01_langchain_example.py)

[使用 LLamaIndex 实现一个 RAG demo.](./code/02_llamaIndex_example.py)

## 参考文献

[Genesis, J. (2025). Retrieval-Augmented Text Generation: Methods, Challenges, and Applications.](https://www.researchgate.net/publication/391141346_Retrieval-Augmented_Generation_Methods_Applications_and_Challenges)

[Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey.](https://arxiv.org/abs/2312.10997)

[Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.](https://arxiv.org/abs/2005.11401)

[Gao et al. (2024). Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks.](https://arxiv.org/abs/2407.21059)

[TinyRAG: GitHub项目.](https://github.com/KMnO4-zx/TinyRAG)