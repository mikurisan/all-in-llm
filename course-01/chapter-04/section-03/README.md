Text-to-SQL 指的是将 natural language queries 转换为 sql statements.

## 1 业务挑战

- **Hallucination**: LLM 会 invent 出 non-existent fields, 导致 invalid sql.

- **Insufficient Schema Understanding**: LLM 需要 a precise understanding of schema, fileds 的 meaning, 以及 relationships between tables, 才能生成 accurate SQL.

- **Handling Input Ambiguity**: User queries 可能存在 typos, colloquialisms or vague expressions. LLM 需要具备一定的 fault tolerance 和 reason capability.

## 2 优化策略

1. **Provide Accurate Schema**: 向 LLM 提供 create table statements.

2. **Use High-Quality, Few-Shot Examples**: 在 prompt 中添加一些 question-sql pairs 可以 significantly 提升 query 的 accuracy.

3. **Enhance Context with RAG**: 为 database 建立一个 knowledge base, 除了 tabl 的 ddl 还可以 include:

    - **Detailed Description**: Explanations of what each table and field represents.

    - **Synonyms and Business Terminology**: 例如“花费“映射到的是 “lost”.

    - **Complex Query Examples**: 提供一些相对 exmaples of sophisticated sql queries .

4. **Error Correction and Reflection**: 生成 sql 后尝试 execute, error occurs 后将其返回给 LLM, 让其自行 correct 后 retry.

> 和 section-02 一样, 利用传递给 llm 到 metadata 让其输出 specifc query statement, 很依赖 llm 到能力.

## 参考代码

[实现一个简单的 Text2Sql 框架.](./code/01_text2sql_demo.py)

## 参考文献

[LangChain Docs: Text to SQL.](https://docs.langchain.com/oss/python/langchain/rag)

[RAGFlow Blog: Implementing Text2SQL with RAGFlow.](https://ragflow.io/blog/implementing-text2sql-with-ragflow)