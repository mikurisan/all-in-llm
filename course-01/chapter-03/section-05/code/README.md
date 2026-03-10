## 1 [句子窗口检索](./01_sentence_window_retrieval.py)

其底层代码实现过程可分解为:

1. 句子切分: 解析器接收一个 document, 将其 split 为一个 sentences list.

2. 创建基础节点: 为 sentences list 中的每个 sentence 创建独立的 node.

3. 构建窗口并填充元数据(主要循环): 解析器遍历所有 node, 对每个 node 执行:

  - 定位窗口: 获取该 node 及其前后 n 个邻近 nodes 并存储在 node list 中.

  - 组合窗口文本: 将 node list 中的 text 用空格拼接成一个长字符串.

  - 填充元数据: 将长字符串即 context text 存入到当前 node 到 metadata.

  - 设置元数据排除项: 让后续的 embedding 或 llm 只处理 node 的 text.

  - 后处理器处理: 当 query 到该 node 后, 使用 metadata 中的 context text 替代原本 node 中的 text.

## 2 [递归检索](./02_recursive_retrieval.py)

主要原理:

- 创建专门 engine: 遍历每个 sheet 转换为 dataframe, 并为其创建一个 `PandasQueryEngine`, 该 engine 能够将自然语言问题转换为实际的 pandas 代码执行.

- 创建摘要节点: 对每个 sheet 创建一段 summary, 以作为顶层检索对指针.

- 构建顶层索引: 对所有摘要节点创建索引.

- 创建 retriever: 提供一个从 index id 到其 query engine 的 mapping 给到 retriever, 这样当 node 被检索到的时候就知道该调用哪个 engine 了.

> 安全警告: 其原理是让 LLM 生成 python code, 然后使用 `eval()` 在本地执行, 理论上可能执行任意 code.

## 2 [更安全的递归检索](./03_recursive_retrieval_v2.py)

创建 2 个独立的向量索引:

- 摘要索引 (用于路由): 为每个 sheet 创建一个 summary 用于检索.

- 内容索引 (用于回答): 将 sheet 转换为 text 形式并创建 index, 同时附加 metadate 用于 filter.

当发生 query 时首先通过摘要索引找到相关 nodes, 再根据 metadata 过滤, 最后在根据内容索引进行检索.

> 弯弯绕绕, 关键就是调用 `df.to_string()` 将 dataframe 转换为 string, 从而避免上面的生成 code.