<div align="center">
  <img src="./img/nousresearch.png" width="180" />
  <br>
  <h1>RAG</h1>
</div>

## [简单认识 RAG](./chapter-01/README.md)

如题.

## [数据准备](./chapter-02/README.md)

数据加载, 文本分块.

## [索引构建](./chapter-03/README.md)

向量嵌入, 多模态嵌入, 向量数据库, Milvus 实践, 索引优化.

## [检索优化](./chapter-04/README.md)

混合检索, 查询构建, 

- [chapter-05: 生成集成](./chapter-05/README.md)

- [chapter-06: RAG 系统评估](./chapter-06/README.md)

- [chapter-07: 高级 RAG 架构 (选修)](./chapter-07/README.md)

- [chapter-08: 项目实战 (基础篇)](./chapter-08/README.md)

## 必要提醒

### 配置模型 API
运行 code 前引入以下 env variables:

```shell
export BASE_URL=""
export API_KEY=""
export MODEL_NAME=""
```

### 安装模型

```shell
cd /opt/repo/all-in-llm/course-01/model/visual_bge

uv pip install -e .

cd /opt/repo/all-in-llm/course-01/model

python download_model.py
```