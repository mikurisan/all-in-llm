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

- **Deduplication**: If multiple retireved chunks belong to the same parent, merge them.

- **Generation**: Pass the complete parent document to the LLM to provide rich context.

Metadata Enhancement:

- **Dish Category**: Inferred from the file path.

- **Difficulty Level**: Extracted from the document content.

- **Dish Name**: Extracted from the filename.

- **Document Relationships**: Establish ID mapping relationships between parent and child document.

### Example Code

[Implementation of the "Data Preparation Module"](./code/rag_modules/data_preparation.py)

## 4 Index Construction & Retrieval Optimization

### 4.1 Building the Index

- Using BGE-small-zh-v1.5 model to create embedding.

- Using FAISS as the vector database.

- To make startup faster, saving the built index locally to a file.

### 4.2 Hybrid Retrieval

- Combining 2 search methods:

    1. Vector Search

    2. Keyword Search

- Merging the results from both methods using RPF (Reciprocal Rank Fusion).

- The system also supports smart filtering using metadata.

### Example Code

["Index Construction Module" Implementation](./code/rag_modules/index_construction.py)

["Retrieval Optimization Module" Implementation](./code/rag_modules/retrieval_optimization.py)

## 5 Generation Integration

Responsible for unerstanding user intent, routing query types, and generating high-quality answers.

### 5.1 Design Approach

- **Intelligent Query Routing**: Automatically determines whether the query is a list query, detail query, or general query.

- **Query Rewrite & Optimization**: Rewrite ambiguous or unclear queries.

- **Multi-Mode Generation**:

    - **List Mode**: Suitable for recommendation-type queries, returns a concise recipe list.

    - **Detail Mode**: Suitable for instructional/creation-type queries, returns a detailed step-by-step guidance.

    - **Basic Mode**: Suitable for general queries, provide standard responses.

### Example Code

["Generation Integration Module" Implementation](./code/rag_modules/generation_integration.py)

## 6 System Integration

Coordinates all modules to achieve the full RAG workflow: "data preparation -> index construction -> retrieval optimization -> generation integration". Provides index caching and interactive QA functionality.

### Example Code

["System Integration" Implementation](./code/main.py)

## 7 Optimization Direction

- **Integrate a Graph Database**: Construct recipe data into a knowledge graph to reveal complex relationships between ingredients, dishes, and cooking methods. This supports complex relational queries, uncovers potential ingredients combinations, and enables graph-based intelligent recommendations.

- **Incorporate Multimodal Data**: Combine visual information like dish images and use multimodal models for joint text-image retrieval. Alternatively, utilize image recognition to identify ingredients and recommend relevant recipes.

- **Enhance Domain-Specific Knowledge**: Integrate external knowledge base such as nutritional databases, cooking technique knowledge graphs, and ingredient substitution rules. This enables nutritional analysis, cooking guidance, and flexible adaptation to user dietary habits and preferences.

> This recipe RAG project is poorly implemented and has numerous optimization opportunities. It is only useful for understanding the basic architecture and workflow of a simple RAG project.