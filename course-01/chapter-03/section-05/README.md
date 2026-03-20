## 1 上下文扩展

Small chunk 缺乏 sufficient context, large chunk 又容易引入 noise. LlamaIndex 提出 Sentence Window Retrieval 以处理这种 trade-off. 其 advantage 在于: 检索时 focus on 单个 sentence, 送入 LLM 之前又 intelligently 将 context 拓展成 wider window.

### 1.1 主要思路

Main workflow:

1. **Indexing**: 将 document 分割为 individual sentences. 每个 sentence 作为 a separate node 存入 vector database. 每个 node 会在其 metadata 中存储其 context window, 即该 sentence 的前后 n 句.

2. **Retrieval**: 只针对 node 对 query 进行 similarity search.

3. **Post-processing**: Retrieve 到 relevant node 后, 再 fetches 对应的 context window 中的 sentences

4. **Generation**: Pass 这些 sentences 给 LLM.

### 示例代码

[加载一份 PDF 格式的 IPCC 气候报告，并就其中的专业问题进行提问.](./code/01_sentence_window_retrieval.py)

## 2 结构化索引

当 knowledge base 不断扩大, 每次 retrieval 都要查找 all documents 是 inefficient.

One solutionn 是利用 structured indexing. 其 principle 是索引该 chunk 的同时, 为其 attach 结构化的 metadata. 这些 metadata 可以帮助 filter. Such as:

- File name

- Creation date

- Chapter title

- Author

- Custom tags

这样在 retrieval 的时候, 可以先过滤 metadata 再进行 vector similarity search.

A similart approach 还有对 document 作 summary, 然后先从 summary 进行 similarity search.

### 示例代码

[Excel 多 sheets 递归检索.](./code/02_recursive_retrieval.py)

[更安全的 Excel 多 sheets 递归检索.](./code/03_recursive_retrieval.py)

## 参考文献

[Building Performant RAG Applications for Production.](https://developers.llamaindex.ai/python/framework/optimizing/production_rag/)

[LlamaIndex - Sentence Window Retrieval.](https://developers.llamaindex.ai/python/examples/node_postprocessor/MetadataReplacementDemo/#metadata-replacement-node-sentence-window)

[Recursive Retriever + Query Engine Demo.](https://developers.llamaindex.ai/python/examples/query_engine/pdf_tables/recursive_retriever)

[Structured Hierarchical Retrieval.](https://developers.llamaindex.ai/python/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval/)

