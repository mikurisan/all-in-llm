文本到 SQL (Text-to-SQL) 将自然语言 query 转换为 sql 查询语句.

## 1 业务挑战

- “幻觉”问题: LLM 会想象出不存在的 fields, 导致 sql 失效.

- 对 schema 理解不足: LLM 需要准确理解 schema, fileds 的含义, 以及 talbes 间的关联关系, 才能保证 sql 准确.

- 处理用户输入的模糊性: query 可能存在拼写错误或不规范的表达等, LLM 需要具备一定的容错和推理能力.


## 2 优化策略

1. 提供准确的 schema: 向 LLM 提供 create table 语句.

2. 提高少而高质量的 example: 在 prompt 中添加一些 question-sql pairs 可以极大提升 query 的准确性.

3. 利用 RAG 增强 context: 为 database 建立一个知识库, 除了 tabl 的 ddl 还可以包括:

    - table 和 fields 的详细描述: 解释每个 table 是干什么的, 每个 field 的字段含义.

    - 同义词和业务术语: 例如“花费“映射到的是“lost”字段.

    - 复杂的查询示例: 提供一些相对复杂的 sql 查询 exmaples.

4. 错误修正与反思 (Error Correction and Reflection): 生成 sql 后尝试执行, 发生错误后将其返回给 LLM, 让其自行修正后重试.

## 参考代码

[实现一个简单的 Text2Sql 框架](./code/01_text2sql_demo.py)

## 参考文献

[LangChain Docs: Text to SQL. ↩](https://docs.langchain.com/oss/python/langchain/rag)

[RAGFlow Blog: Implementing Text2SQL with RAGFlow. ↩](https://ragflow.io/blog/implementing-text2sql-with-ragflow)