## 1 上下文扩展

小 chunk 缺乏 context, 大 chunk 又容易引入噪声. LlamaIndex 提出句子窗口检索(Sentence Window Retrieval) 以解决该矛盾. 其优点在于: 检索时聚焦于单个 sentence, 送入 LLM 之前又智能地将 context 拓展成更宽的 window.

### 1.1 主要思路

其主要工作流程:

1. 索引阶段: 将 document 分割为单个 sentence. 每个 sentence 作为独立的 node 存入 vector database. 每个 node 会在其 metadata 中存储其 context window, 即该 sentence 的前后 n 句.

2. 检索阶段: 只针对 node 对 query 进行 similarity 检索.

3. 后处理阶段: 检索到相关 node 后, 再读取对应的 context window 中的 sentences

4. 生成阶段: 传递这些 sentences.

### 1.2 代码示例

[加载一份 PDF 格式的 IPCC 气候报告，并就其中的专业问题进行提问. ↩](./code/01_sentence_window_retrieval.py)

## 2 结构化索引

当知识库不断扩大, 每次检索都要查找所有 documents, 效率是很低的.

解决这种问题的一个方式是利用结构化索引. 其原理是索引该文本块的同时, 为其添加结构化的 metadata. 这些 metadata 可以帮助 filter, 例如:

- 文件名

- 创建日期

- 章节标题

- 作者

- 自定义的标签

这样在检索的时候, 可以先过滤 metadata 再进行 vector 检索.

类似的思路还有对 document 作 summary, 然后先从 summary 进行 similarity 检索.

### 2.1 代码示例

[Excel 多 sheets 递归检索. ↩](./code/02_recursive_retrieval.py)

[更安全的 Excel 多 sheets 递归检索. ↩](./code/03_recursive_retrieval.py)

## 参考文献

[Building Performant RAG Applications for Production. ↩](https://developers.llamaindex.ai/python/framework/optimizing/production_rag/)

[LlamaIndex - Sentence Window Retrieval. ↩](https://developers.llamaindex.ai/python/examples/node_postprocessor/MetadataReplacementDemo/#metadata-replacement-node-sentence-window)

[Recursive Retriever + Query Engine Demo. ↩](https://developers.llamaindex.ai/python/examples/query_engine/pdf_tables/recursive_retriever)

[Structured Hierarchical Retrieval. ↩](https://developers.llamaindex.ai/python/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval/)

