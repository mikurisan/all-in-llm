需要处理的数据源是多样的, 例如结构化数据, 非结构化数据, 图数据. 用户的 query 也不仅仅是简单的语义检索, 往往也包含复杂的 filter, 聚合操作或关系查询.

查询构建 (Query Construction) 利用 LLM 将用户的自然语言 query 翻译为特定数据源的结构化查询或者带有 filter condition 的 request.

![alt text](./img/image01.png)

## 1 文本到元数据过滤器

构建 index 的时常常为 chunk 添加 metadata, 这些 metadata 为结构化 fitler 提供了可能.

自查询检索器 (Self-Query Retriever) 是 LangChain 中实现该功能的核心组件, 其主要工作流程:

1. 定义元数据结构: 向 LLM 描述 document content 和每个元数据 field 的含义和类型.

2. 查询解析: 调用 LLM 将 query 分解为两部分:

    - 查询字符串 (Query String): 用于进行语义检索的部分

    - 元数据过滤器 (Metadata Filter): 从 query 中提取出的结构化 filter

3. 执行查询: 将上述两部分传递给 vector database 进行查询. 当然这里会将上述的查询再翻译为 db 能够理解的原生语法.

## 2 文本到 Cypher

查询构建技术还能够应用到更复杂的数据结构, 比如图数据库.

### 2.1 什么是 Cypher

Cypher 是图数据库 (如 Neo4j) 中最常见的查询语言.

### 2.2 文本到 Cypher 的原理

利用 LLM 将 query 直接翻译为一句精准的 cyphter 语句查询.

以 LangChain 中的 `GraphCypherQAChain` 为例, 其大致工作流程为:

1. 接收自然语言 query

2. LLM 根据预先提供的图谱 schema, 将问题转换为 cypher 查询

3. 在图数据库中执行查询, 获取结构化数据

4. (optional) 将结果传递给 LLM 生成 answer.

生成 cypher 查询复杂度较高, 需要更强的 LLM .

## 参考代码

[text 转换为元数据 filter](./code/01_text_to_metadata_filter.py)

## 参考文献

[LangChain Blog: Query Construction. ↩](https://blog.langchain.ac.cn/query-construction/)
