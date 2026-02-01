## 1 什么是 RAG ?

### 1.1 核心定义

RAG(Retrieval-Augmented Generation), 即通过检索外部知识来用于增强 LLM 的准确性. 其中:

- 参数化知识: 即 LLM 已经学习固化的权重.
- 非参数化知识: LLM 没有学习过的知识, 也就是这里的外部知识.

### 1.2 技术原理

**检索阶段**:

- 知识向量化: 通过 Embedding Model 将外部知识编码为向量索引 (index) 存入向量数据库.
- 语义召回: 通过同样的 Embedding Model 将 user input 进行向量化, 通过相似度搜索 (Similarity Search) 从数据库中找到与 input 最匹配的知识片段.

**生成阶段**:

- 将 input 与知识按照预设 prompt 指令进行整合, 引导 LLM output.

### 1.3 技术演进分类

| | Naive RAG | Advanced RAG | Modular RAG |
| -- | -- | -- | -- |
| 流程 | 离线: 索引 <br> 在线: 检索 -> 生成 | 离线: 索引 <br> 在线: input -> 检索前 -> xx -> 检索后 ->  new input | 积木式可编排流程 |
| 特点 | 基础线性流程 | 增加检索前后的优化步骤 | 模块化, 可组合, 可动态调整 |
| 关键技术 | 基础向量检索 | 查询重写 (Query Rewrite) <br> 结果重排 (Rerank) | 动态路由 (Routing) <br> 查询转换 (Query Transformation) <br> 多路融合 (Fusion) | 
| 局限性 | 效果不稳定, 难以优化 | 流程相对固定, 优化点有限 | 系统复杂性高 |

> 上述部份简单了解, 后序会熟悉

## 2 为什么使用 RAG ?

### 2.1 RAG VS 微调

从成本和效益考虑: Prompt Engineering -> RAG -> Fine-tuning.

基于以下场景考虑:

- Prompt Enginnering: 引导 LLM output, 适用于任务简单, 模型已有相关知识的场景.

- RAG: LLM 缺乏特定的知识.

- Fine-tuning: 改变 LLM 如何做, 如行为, 风格, 格式等.

最厉害的是 hybrid way 将 RAG 用于 Fine-tuning.

RAG 解决了 LLM 的几个局限性:

- 静态知识局限: 实时检索外部知识

- 幻觉 (Hallucination): 基于检索知识, 错误率降低

- 领域专业不足: 引入领域特定知识库

- 数据隐私风险: 本地化部署知识库

### 2.2 关键优势

1. 准确性与可信度: 补充知识, 减少幻觉, output 可溯源.

2. 时效性保障: 知识动态更新, 解决知识时滞问题, 这也称为“索引热拔插“ (Index How-swapping) 技术.

3. 显著的综合成本效益: 避免高频 fine-tuning 的巨额算力成本; 在处理特定领域问题时, 也可以采用参数量更小的 LLM 来达到目标效果.

4. 模块化扩展性: 知识来源是多样的, 模块化的, 与检索组件是节耦的, 二者之间的优化互不干扰.

### 2.3 适用场景风险分级

| 风险等级 | 案例 | 适用性 |
| -- | -- | -- |
| 低 | 翻译/语法检查 | 高可靠 |
| 中 | 合同起草/法律咨询 | 需结合人工审核 |
| 高 | 证据分析/签证决策 | 需严格控制质量 |

## 3 如何上手 RAG ?

### 3.1 基础工具链选择

- 成熟框架: LangChain, LlamaIndex; 当然也可以原生开发

- 向量数据库: 大型用 Milvus, Pinecone, 轻量用 FAISS, Chroma

- 评估工具: RAGAS, TruLens

### 3.2 4 步构建最小可行系统 (MVP)

1. 数据准备与清洗: 异构数据源标准化, 采用分块策略对文本进行分割

2. 索引构建: 通过 embedding model 将文本块向量化, 并存入数据库, 在此阶段可以关联元数据

3. 检索策略优化: 可以采用混合检索, 并引入重排序优化检索结果

4. Output 与 Prompt Enginnering: 引导 LLM 基于检索内容给出目标结果

### 3.3 新手友好方案

希望快速验证可以使用:

- 可视化知识库平台: FastGPT, Dify

- 开源模板: LangChain4j, TinyRAG

### 3.4 进阶与挑战

- 评估唯独与挑战: 检索相关度与生成质量, 又分为语义准确性 (回答的意思是否正确)和词汇匹配度 (专业术语是否使用得当). 如果检索到错误知识, LLM 就会胡说八道; 此外跨文档的多跳推理也具有挑战性.

- 优化方向与架构演进: 性能方面可通过索引分层和多模态扩展 (支持, 图像表格检索). 架构方面有更加复杂的设计模式, 例如可以通过分支模式并行处理多路检索, 或通过循环模式进行自我修正.

## 4 RAG 已死

并不, 现已成为 LLM 开发中的一种基本范式, 并正在快速发展.

## 参考文献

[Genesis, J. (2025). Retrieval-Augmented Text Generation: Methods, Challenges, and Applications. ↩](https://www.researchgate.net/publication/391141346_Retrieval-Augmented_Generation_Methods_Applications_and_Challenges)

[Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. ↩](https://arxiv.org/abs/2312.10997)

[Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. ↩](https://arxiv.org/abs/2005.11401)

[Gao et al. (2024). Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks. ↩](https://arxiv.org/abs/2407.21059)

[TinyRAG: GitHub项目. ↩](https://github.com/KMnO4-zx/TinyRAG)
