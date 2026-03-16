## 1 项目背景

[How To Cook](https://github.com/Anduin2017/HowToCook) 是一个 open-source recipe project, 使用 markdown 记录了各种 recipe 的 cooking methods, 并且 documents 都 strictly 使用了 unified title format.

基于此 start 一个 Recipe QA RAG Project.

## 2 项目架构

### 2.1 项目目标

用户可以:

- 询问具体菜品的制作方法

- 寻求菜品推荐

- 获取食材信息

### 2.2 数据分析

#### 2.2.1 文档分析

"How to cook" project 包含超 300 多个的 markdown recipe files, 其 content 也是 well-structured 和 concise, 很适合 structural segmentation.

#### 2.2.2 结构分块局限

按照标题结构分块会将 conten 分割得太细, 导致 context 不完整, 从而无法给出 perfect answer.

为了解决这一矛盾, 可采用父子文本块策略: 使用小的子 chunk 进行 retrieval, 然后传递整个 document 给 LLM.

为什么直接用整个 document 进行 retrieval 的原因在于, query 如果在整个 document 的占比很小, 会导致 retrieval 的 precision 降低, 从而影响 retrieval 的结果.

### 2.3 整体架构

略.

### 2.4 项目结构

```text
code
├── config.py                   # 配置管理
├── main.py                     # 主程序入口
├── requirements.txt            # 依赖列表
├── rag_modules/               # 核心模块
│   ├── __init__.py
│   ├── data_preparation.py    # 数据准备模块
│   ├── index_construction.py  # 索引构建模块
│   ├── retrieval_optimization.py # 检索优化模块
│   └── generation_integration.py # 生成集成模块
└── vector_index/              # 向量索引缓存 (自动生成)
```

## 3 "数据准备模块"实现

父子 chunk 映射 relationship:

```
父 chunk (完整 document)
├── 子 chunk 1: 菜品介绍 + 难度评级
├── 子 chunk 2: 必备原料和工具
├── 子 chunk 3: 计算 (用量配比)
├── 子 chunk 4: 操作 (制作步骤)
└── 子 chunk 5: 附加内容 (变化做法)
```

基本流程:

- Retrieval: 使用子 chunk 进行匹配, 提高 retrieval precision.

- Generation: 传递完整的父 document 给 LLM, 提供丰富 context.

- Deduplication: 若多个 chunks 同属一个父级, 则进行 merge.

Metadata Enhancement:

- 菜品分类: 从文件路径推断

- 难度等级: 从 document content 中提取.

- 菜品名称: 从 file name 提取.

- 文档关系: 建立父子 document 的 ID mapping relationship.

### 代码示例

[“数据准备模块“实现](./code/rag_modules/data_preparation.py)

## 4 索引构建与检索优化

### 4.1 索引构建

选择 BGE-small-zh-v1.5 作为 embedding model, 使用 FAISS 作为 vector database.

为了提升启动速度, 构建后的 index 会保存到本地.

### 4.2 混合检索

采用 vector 和 keyword 的混合检索方式, 使用 RPF (Reciprocal Rank Fusion) 融合 retrieval result.

此外, system 还支持基于 metadata 的智能过滤.

### 代码示例

[“索引构建模块”实现](./code/rag_modules/index_construction.py)

["检索优化模块“实现](./code/rag_modules/retrieval_optimization.py)

## 5 生成集成

负责理解用户意图, 路由查询类型, 并生成高质量 answer.

### 5.1 设计思路

- 智能查询路由: 根据用户 query 自动判断是列表查询, 详细查询还是一般查询.

- 查询重写优化: 对模糊不清的 query 进行重写.

- 多模式生成: 

    - 列表模式: 适用于推荐类 query, 返回简介的 recipe list

    - 详细模式: 适用于制作类 query, 提供分步骤的详细指导.

    - 基础模式: 适用于一般性 query, 提供常规回答.

### 代码示例

[“生成集成模块“实现](./code/rag_modules/generation_integration.py)

## 6 系统整合

负责协调各个 module, 实现 “数据准备 -> 索引构建 -> 检索优化 -> 生成集成“ 的完整 RAG 流程. 提供索引缓存, 交互式问答等功能.

### 示例代码

["系统整合“实现](./code/main.py)

## 7 优化方向

- 集成图数据库: 将 recipe data 构建为知识图谱, 以此揭示食材, 菜品与烹饪方法间的复杂关联. 进而支持复杂的关系查询, 发掘潜在的食材组合, 实现基于图的智能推荐.

- 融合多模态数据: 组合菜品图片等视觉信息, 利用多模态 model 进行图文联合 retrieval. 或者通过图像识别食材来推荐相关 recipe.

- 增强专业知识: 比如集成营养成份数据库, 烹饪技巧知识图谱, 以及食材替换规则等外部知识源. 从而, 提供营养分析, 烹饪指导, 并灵活适应用户的饮食习惯和偏好.
