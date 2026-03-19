## 1 向量数据库的作用

To quickly and accurately 从  vector 中找到与 query 最相似的 top-n.

### 1.1 向量数据库的主要功能

核心 lies in 高效 handle 海量 high-dimensional vector 的 capability. Can be summarized as:

- **Similarity Search** (Most important): 利用 spcialized indexing techniques (e.g. HNSW, IVF) 在 billions of vectors 中实现 ms 级的 Approximate Nearest Neighbor (ANN) query.

- **Storage and Management of High-Dimensional Vectors**: Dimensions 可达成 hundreds or thousands, 支持 CRUD operation.

- **Query Capability**: 支持按 scalar fields 过滤查询, range queries, cluster analysis 等.

- **Scalability and High Availability**: Distributed architecture, horizontal scaling and fault tolerance.

- **Ecosystem Integration**: 支持 mainstream LLM framework, 可 seamlessly 与 machine learning workflows 集成.

### 1.2 向量数据库 VS 传统数据库

主要差异通常为:

| 维度 | 向量数据库 |	传统数据库 (RDBMS) |
| --- | --- | --- |
| 核心数据类型	 | 高维向量 (Embeddings) |	结构化数据 (文本、数字、日期) |
| 查询方式	 |相似性搜索 (ANN)	 |精确匹配 |
|索引机制	 |HNSW, IVF, LSH 等 ANN 索引	 |B-Tree, Hash Index |
|主要应用场景 |	AI 应用, RAG, 推荐系统, 图像/语音识别 |	业务系统 (ERP, CRM), 金融交易, 数据报表 |
|数据规模	 |轻松应对千亿级向量 |	通常在千万到亿级行数据, 更大规模需复杂分库分表 |
|性能特点 |	高维数据检索性能极高, 计算密集型 |	结构化数据查询快, 高维数据查询性能呈指数级下降 |
|一致性	 | 通常为最终一致性	 | 强一致性 (ACID 事务)|

## 2 工作原理

Vector database 通常采用 4-layer achitecture (top-down):

- **Service Layer**: 管理 client connections, 提供 monitoring 和 logging, 实现 security management.

- **Query Layer**: 处理 query requests, 支持 hybrid queries, 实现 query optimization.

- **Index Layer**: 维护 indexing algorithms, index的 creation 和 optimization, 支持 index tuning.

- **Storage Layer**: 存储 vector 和 metadata, 优化 storage efficiency, 支持 distributed storage.

主要技术手段:

- **Tree-based Methods**: 如 Annoy 使用的 random projection trees, 通过 a tree structure 实现 logarithmic search complexity.

- **Hash-based Methods**: 如 LSH (Locality Sensitive Hashing), 通过 hash functions 将 similar vector 映射到 the same bucket.

- **Graph-based Methods**: 如 HNSW (Hierachical Navigable Small World), 通过 multi-layer proximity graph structure 实现 fast search.

- **Quantization-based Methods**: 如 Faiss 的 IVF 和 PQ, 通过 clustering 和 quantization 压缩向量.

## 3 主流数据库介绍

![alt text](./img/image.png)

当前的主流产品有:

- **Pinecone** 是一款完全托管的向量数据库服务, 采用 Serverless 架构设计. 它提供存储计算分离, 自动扩展和负载均衡等企业级特性, 并保证 99.95% 的 SLA. 其支持多种语言 SDK, 提供极高可用性和低延迟搜索(<100ms), 特别适合企业级生产环境, 高并发场景和大规模部署.

- **Milvus** 是一款开源的分布式向量数据库, 采用分布式架构设计, 支持 GPU 加速和多种索引算法. 它能够处理亿级向量检索, 提供高性能GPU加速和完善的生态系统. 其特别适合大规模部署, 高性能要求的场景, 以及需要自定义开发的开源项目.

- **Qdrant** 是一款高性能的开源向量数据库, 采用 Rust 开发, 支持二进制量化技术. 它提供多种索引策略和向量混合搜索功能, 能够实现极高的性能 (RPS>4000) 和低延迟搜索. Qdrant 特别适合性能敏感应用, 高并发场景以及中小规模部署.

- **Weaviate** 是一款支持 GraphQL 的 AI 集成向量数据库, 提供 20+ AI 模块和多模态支持. 它采用 GraphQL API 设计, 支持 RAG 优化, 特别适合 AI 开发, 多模态处理和快速开发场景. Weaviate具有活跃的社区支持和易于集成的特点.

- **Chroma** 是一款轻量级的开源向量数据库, 采用本地优先设计, 无依赖. 它提供零配置安装, 本地运行和低资源消耗等特性, 特别适合原型开发, 教育培训和小规模应用. Chroma的部署简单, 适合快速原型开发.

Selection Recommendations:

- **For Small Project**: 选择 Chroma 或 FAISS, 与 mainstream framework 紧密 integrated, a few lines of code 即可满足 basic storage 与 retrieval.

- **For Large Projects**: 超过 millions of vectors, 需要 high concurrency, real-time updates, complex metadata filtering, 可考虑 Milvus, Weaviate, Pinecone.

## 参考代码

[使用 Langchian 封装的 FAISS 完成 vector 的创建, 保存, 加载和查询.  ↩](./code/01_langchain_faiss.py)

[使用 LlamaIndex 完成 vector 的创建, 保存, 并查看保存到本地的 json 文件.  ↩](./code/02_llamaindex_vector.py)