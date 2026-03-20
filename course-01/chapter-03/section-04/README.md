## 1 简介

An open-source, 专为 massive-scale similarity search 和 analysis 而 designed 的 vector database.

## 2 部署安装

下面以 Standlone(单机版) 为例.

### 2.1 Docker 安装

略

### 2.2 下载和启动

下载配置文件:

```shell
wget https://github.com/milvus-io/milvus/releases/download/v2.5.14/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

启动:

```shell
docker compose up -d
```

### 2.3 验证安装

```shell
docker ps
```

### 2.4 常用管理命令

停止并移除 container (保留 volumnes):

```shell
docker compose down
```

停止并移除 container (不保留 volumnes, 彻底移除):

```shell
docker compose down -v
```

### 3 核心组件

### 3.1 Collection

The most fundamental organizational unit for data, 所有的 CRUD 都围绕 Collection 展开.

一个 Collection 包含 several Partition, 用于 isolate 数据, 以便于 retrieval; Partition 中包含 multiple Entities, 即数据本身.

Schema 定义了 the structure of Entities.

Alias 指向 a specific Collection 以便于 application layer 的调用.

> 类似于结构化数据库中的 table 概念.

#### 3.1.1 Schema

Schema 定义了 Collection 的 fields 及其属性.

通常包括以下 fields:

- **Primary Key Field**: Must 拥有 exactly one, 作为 Entity 的 unique identifier.

- **Vector Field**: 存储 core vector data, 可以有多个, 以满足 multimodal scenarios.

- **Scalar Field**: 存储 metadata, 用于 filtering queries.

![alt text](./img/image01.png)

> 类似于结构化数据库中表的 schema 概念.

#### 3.1.2 Partition

是 Collection 的一个 logical subdivision, 默认会有一个 `_default` partition. 

**Why use Partition?**

- **Enhance Query Performance**: Search 时指定 partition, 减少 the volume of data scanned.

- **Data Management**: 按需 load 指定 partition data, prevent 加载 entire Collection.

一个 Collection 最多有 1024 个 Partition.

> 类似于结构化数据库中的分区 table.

#### 3.1.3 Alias

在 programs 中对 Alias 进行操作, avoiding 对 Collection 名的 direct references.

**Why use Alias?**

- **Safe Data Updates**: 更新 an underlying copy of the original Collection 后, 执行 atomic switch 到 new Collection, 不 impacting 上层 application.

- **Code Decoupling**: The entire switching process 对 upper layer 是 transparent, code 无需 modification.

### 3.2 Index

Index 本身是一种为 accelerate 查询而 designed 的 complex data structure, 其 significantly improve 了 the speed of similarity searches, 代价便是占用 additional storage and computational resources.

![alt text](./img/image02.png)

说明:

- **Data Structure**: Index 的 framework, defines 如何organize 向量.

- **Quantization**: A compression technique, 降低 vector precision 以 save 计算和存储资源.

- **Refiner**: 找到 initial candidate set 后, 进行 more precise calculations 以 refine 结果.

Milvus 支持 scalar fields 和 vector fields 分别创建 index:

- **Scalar Field Index**: 用于加速 metadata 过滤. Typically 使用 recommended index type 即可.

- **Vector Field Index**: The core component. 合适的 vector index 是在 query performance, recall rate 和 memory consumption 之间的 art of balance.

> Index 是为了加速 query 的, 并关注 query 的速度, 质量和资源消耗.

#### 3.2.1 主要 vector index 类型

Overview of Core Index Types:

- **FLAT (Exact Search)**

  - **Principle**: Brute-force Search. 计算 query vector 与所有 vectors 的 exact distance, 返回 the most precise results.

  - **Advantage**: 100% recall rate.

  - **Disadvantages**: Slow search speed, high memory consumption, unsuitable 对于海量数据.

  - **Applicable Scenarios**: Extremely high accuracy requirements, and relatively small-scale data (within millions).

- **IVF Series (Inverted File Index)**

  - **Principle**: 先通过 cluster 将 vectors 分成不同的 buckets, query 时找到几个 most similar buckets, 然后只在这几个 buckets 内执行 exact search. 其有几种不同 variants, 主要 differ 在于是否对 vectors within buckets 进行了 quantization.

  - **Advantage**: 缩小 search scope, 提高 search speed, 在 performance 和 effectiveness 之间取得 a good balance.

  - **Disadvantages**: 非 100% recall rate, 相关 vectors 可能 reside in 未被 search 的 buckets.

  - **Applicable Scenarios**: General-purpose, especially 适合 high-throughput 的 large-scale datasets.

- **HNSW (Hierarchical Navigable Small World)**

  - **Principle**: 构建 a multi-layer 的 proximity graph, 从 the top sparse layer 开始 search, 快速 locate 到 the target region, 随后在 denser lower layers 进行 precise search.

  - **Advantage**: Exremely fast retrieval speed, high recall rate, excels in 处理 high-dimensional data 和 law-latency queries.

  - **Disadvantages**: Very high memory consumption, longer index build time.

  - **Applicable Scenarios**: Strict requirements for query latency.

- **DiskANN (Disk-based Index)**

  - **Principle**: 一种为在 high-speed disks 上运行而 optimized 的 index.
  
  - **Advantage**: 支持 massive datasets, 同时 mmaintain 较低的 search latency.

  - **Disadvantages**: 相比 memory-based indices, latency 较高.

  - **Applicable Scenarios**: Extremely large datasets, 无法全部 loaded into 内存.

### 3.2.2 如何选择 Index?

根据 business 场景从 data volume, memory constraints, query performance 和 recall rate 之间进行 trade-off.

| 场景| 	推荐索引| 	备注| 
| --- | --- | --- |
| 数据可完全载入内存, 追求低延迟| 	HNSW| 	内存占用较大, 但查询性能和召回率都很优秀. | 
| 数据可完全载入内存, 追求高吞吐| 	IVF_FLAT / IVF_SQ8| 性能和资源消耗的平衡之选. | 
| 数据量巨大, 无法载入内存| 	DiskANN| 	在 SSD 上性能优异, 专为海量数据设计. | 
| 追求 100% 准确率, 数据量不大| FLAT| 	暴力搜索, 确保结果最精确. | 

In practice, 往往需要通过 trial 来找到适合 specific data 和 query pattern 的 index type 及其 parameters.

## 3.3 Search

### 3.3.1 ANN Search

Approximate Nearest Neighbor Search, 利用 pre-built index, 从 datasets 中找出与 query vector 最相似的 top-K vecctors.

该 strategy 在 speed 和 precision 之间取得 optimal compromise.

### 3.3.2 Enhanced Search

在基础 ANN Search 之上提供多种 advanced research capabilities.

**Filtered Search**

将 similarity search 与 scalar field search 结合.

- **Mechanism**: 根据 filter expression 过滤出 entities, 然后再执行 ANN search.

- **Examples**: 

  - **E-commerce**: Search 与 "red skirt" 最相似的 products, 但只看 price under 500 的.
  
  - **Knowledge Base**: 查找与 AI 相关的 documents, 但只从 “technology” category 下, 且年份大于 2024 的 articles 中找.

**Range Search**

Focuses on 所有与 query vectors 相似度在 specified range 内的 vectors.

- **Mechanism**: 定义一个 similarity threshold, 返回 similarity 在该 range 内的 entities.

- **Examples**:

  - **Facial Recognition**: Retrieve 与 target face 相似度超过 0.9 的 faces, 用于 authentication purposes.

  - **Anomaly Detection**: Identify 所有与 normal samples 的 vector distance 差距过大的 points, 用于 spot anomalies.

**Hybrid Search**

在一个 query 中同时检索 multiple vector fields, 并将结果 rerank.

- **Mechanism**: 

  1. **Parallel Retrieval**: 针对 different vector fields 分别 simultaneously 发起 ANN search 请求.
  
  2. **Fusion & Re-ranking**: Employs 一个 reranker 将 separate search results 合并为 a single, higher-quality ranked list.

- **Examples**:

  - **Multi-modal Product Search**: query ”quiet and comfortable white headphones“, 同时检索 products 的 textual description 和 image content vectors, 返回 best match 的 products.

  - **Enhanced RAG**: 结合 dense vectors (to capture semantic meaning) 和 sparse vectors (to match keywords), 实现 more accurate and contextually relevant document retrieval.

**Grouping Search**

解决 the issue of limited diversity in search results.

- **Mechanism**: 通过 a specified field 对 results 进行 group, each group 只返回组内与 query 相似度最高的 entity.

- **Examples**:

  - **Video Retrieval**: 检索“cute cats”, ensure 返回的 vedio 来自 different creator.

  - **Document Retrieval**: 检索“AI”, ensure 返回的 results 来自 distinct books or publications.

## 参考代码

[使用 Milvus 和 Visualized-BGE 构建一个 eng-to-end 图文多模态检索引擎.](./code/01_multi_milvus.py)