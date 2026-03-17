The most critical component, 谨记 “Garbage In, Garbage Out”.

## 1 文档加载器

### 1.1 主要功能

将 unstructured documents 转化为 structured data, typically 分为 3 个 steps:

1. **Parse Documents**: 从 documents in various formats 中提取 the raw textual content.

2. **Extract Metadata**: Concurrently 抽取 doc source, page number, and author.
3. **Structure Data**: Consolidate 上述 2 者为 a unified format.

> 简单的说就是统一"多源异构数据“的 format.

### 1.2 主流加载器

| 工具名称 |	特点 |	适用场景 |	性能表现 |
| --- | --- | --- | --- |
| PyMuPDF4LLM | PDF → Markdown转换，OCR + 表格识别 | 	科研文献, 技术手册 | 	开源免费, GPU加速 | 
| TextLoader	| 基础文本文件加载 | 	纯文本处理	| 轻量高效 | 
| DirectoryLoader | 	批量目录文件处理 | 	混合格式文档库 | 	支持多格式扩展 | 
| Unstructured | 	多格式文档解析 | 	PDF, Word, HTML 等 | 	统一接口，智能解析 | 
| FireCrawlLoader | 	网页内容抓取 | 	在线文档, 新闻 | 	实时内容获取 | 
| LlamaParse | 	深度PDF结构解析	 | 法律合同, 学术论文 | 	解析精度高, 商业API | 
| Docling | 	模块化企业级解析 | 	企业合同, 报告 | 	IBM生态兼容 | 
| Marker | 	PDF→Markdown, GPU加速 | 	科研文献, 书籍 | 	专注PDF转换 | 
| MinerU | 	多模态集成解析 | 	学术文献, 财务报表 | 	集成LayoutLMv3 + YOLOv8 | 

## 2 Unstructured 文档处理库

### 2.1 核心优势

- **Unified Processing**: Single interface for multiple document types.

- **Advanced Parsing**: Superior format support and content extraction.

- **Structure Recognition**: Auto-detection of titles, paragraphs, tables, lists.

- **Metadata Preservation**: Retention of all structual metadata.

### 2.2 支持的文档元素类型

| 元素类型 | 	描述 | 
| --- | --- |
| Title | 	文档标题 | 
|  NarrativeText | 	由多个完整句子组成的正文文本, 不包括标题, 页眉, 页脚和说明文字 | 
| ListItem | 	列表项, 属于列表的正文文本元素 | 
| Table | 	表格 | 
| Image | 	图像元数据 | 
| Formula | 	公式 | 
| Address | 	物理地址 | 
| EmailAddress | 	邮箱地址 | 
| FigureCaption | 	图片标题/说明文字 | 
| Header | 	文档页眉 | 
|  Footer | 	文档页脚 | 
|  CodeSnippet | 	代码片段 | 
|  PageBreak | 	页面分隔符 | 
| PageNumber | 	页码 | 
|  UncategorizedText | 	未分类的自由文本 | 
| CompositeElement | 	分块处理时产生的复合元素* | 

`CompositeElement` 是通过 chunking 产生的 special element type, 由 one or more contiguous text elements 组合而成. For example, 多个 list items 可能会被combined 为 a single chunk.

## 3 从 LangChain 封装到原始 `Unstructured`

"Chapter 1" 中使用了 LangChain 对 `Unstructured` 进行封装的 [UnstructuredMarkdownLoader](../../chapter-01/section-01/code/01_langchain_example.py#L13), 这里是其 [native usage ](./code/01_unstructured_example.py)
.

`partition` 会 automatically 检测 file type, 随后 routes 到 corresponding specialized function. 例如 pdf 会 invoke `partition_pdf`, therefore 也可 directly 使用:

```py
from unstructured.partition.pdf import partition_pdf
```

> 实际应用中, 更多是使用 PaddleOCR, MinerU 等 model 或 tool 处理 pdf.

## 代码示例

[使用 Unstructured 解析 PDF.](./code/01_unstructured_example.py)

## 参考文献

[Unstructured Open-Source Documentation. ↩](https://docs.unstructured.io/open-source/introduction/overview)

[Unstructured Open-Source: Document Elements. ↩](https://docs.unstructured.io/open-source/concepts/document-elements)