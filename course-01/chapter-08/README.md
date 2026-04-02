## 1 Project Background

[How To Cook](https://github.com/Anduin2017/HowToCook) is an open-source recipe project. It use markdown to record the cooking methods for various recipes, and all documents strictly follow a unified title format.

Based on this, we are starting a Recipe RAG Project.

## 2 Project Architecture

### 2.1 Project Goals

Users will be able to:

- Ask for the method to cook a specific dish.

- Request dish recommendations.

- Get information about ingredients.

### 2.2 Data Analysis

#### 2.2.1 Document Analysis

"How to cook" project contains over 300 markdown recipe files. The content is well-structured and concise, making it suitable for structural segmentation.

#### 2.2.2 Limitations of Structural Chunking

Splitting content purely based on headings can lead to overly granular chunks. This fragments the context, resulting in incomplete information being retrieved. An incomplete context makes it difficult for the LLM to provide a perfect answer.

To address this issue, a parent-child chunking strategy can be employed: smaller child chunks are used for retrieval, but once retrieved, the entire parent document (or a larger containing section) is passed to the LLM for context.

The reason for not using the entire document directly for retrieval is that if the user's query relates to only a small part of a large document, retrieval precision decreases. The relevant information may get lost among less relevant text, negatively impacting the retrieval results.

### 2.3 Overall Architecture

Skipped.

### 2.4 Project Structure

```text
code
├── config.py                   # Config management
├── main.py                     # Main program entry point
├── requirements.txt            # Dependencies list
├── rag_modules/               # Core modules
│   ├── __init__.py
│   ├── data_preparation.py    # Data preparation module
│   ├── index_construction.py  # Index construction module
│   ├── retrieval_optimization.py # Retrieval optimization module
│   └── generation_integration.py # Generation & Integration modules
└── vector_index/              # Vector index cache (auto-generated)
```

## 3 Implementation of the "Data Preparation Module"

Parent-Child Chunk Relationship:

```
Parent Chunk (Complete Document)
├── Child Chunk 1: Dish Introduction + Difficulty Rating
├── Child Chunk 2: Required Ingredients & Tools
├── Child Chunk 3: Calculation (Quantities & Ratios)
├── Child Chunk 4: Operations (Step-by-step Instructions)
└── Child Chunk 5: Additional Content (Variations)
```

Basic Process:

- **Retrieval**: Use child chunk for matching to improve retrieval precision.

- **Generation**: Pass the complete parent document to the LLM to provide rich context.

- **Deduplication**: If multiple retireved chunks belong to the same parent, merge them.

Metadata Enhancement:

- **Dish Category**: Inferred from the file path.

- **Difficulty Level**: Extracted from the document content.

- **Dish Name**: Extracted from the filename.

- **Document Relationships**: Establish ID mapping relationships between parent and child document.

### Example Code

[Implementation of the "Data Preparation Module"](./code/rag_modules/data_preparation.py)

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
