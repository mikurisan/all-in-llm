让 LLM 特定结构的数据, 比如 JSON, XML 等.

## 1 为什么需要格式化生成?

What can I say?

## 2 格式化生成的实现方法

### 2.1 Output Parsers

LangChain 中的一个用于处理 LLM output 的组件, 其主要思想是将“如何格式化输出的 instruction“注入到 prompt 中, 并将返回到 output text 解析成预期的格式化数据.

举例几个 parsers:

- `StrOutputParser`: 将 LLM output 作为 str 返回.

- `JsonOutputParser`: 解释 json str.

- `PydanticOutputParser`: 与 Pydantic 模型结合, 实现 output format 的严格定义和验证.

### 示例代码

[分析 `PydanticOutputParser` 工作原理. ](./code/01_pydantic.py)

### 2.2 LlamaIndex 的输出解析

有 2 大核心组件紧密结合, 分别是响应合成 (Response Synthesis) 和结构化输出 (Structured Output).

在 RAG 时, retriever 召回一系列文本块 (nodes) 后, 并不是将其简单拼接. Response Synthesizer 会接收这些 nodes 和原始 query, 然后以某种更智能的方式将其传递给 LLM, 例如:

- refine 模式: 逐块处理信息, 并迭代优化 answer.

- compact 模式: 将尽可能多的 nodes 压缩进单次 LLM 调用

该阶段的目标是生成一段高质量的 text answer.

当需要返回 structured output, 将会使用 pydantic programs, 其思路与 LangChain 相关:

- 定义 Schema: 定义一个 pydantic model.

- 引导生成: 将 pydantic model 转换为 LLM 能够理解的 formart output instruction. 如果底层 LLM 支持 function calling, 会优先调用.

- 解析验证: 解析 text output 并用 pydantic model 进行验证, 返回 pydantic object instance.

### 2.3 不依赖框架的简单实现思路

主要就是通过 prompt enginnering:

- 明确要求 JSON format: 直接强硬要求返回 JSON format, 不包含任何解释性 text.

- 提供 JSON Schema: 在 prompt 中给出指定的 Json Schema, 描述每个 key 的含义和类型.

- 提供 few-shot examples: 给出几个 "input -> json format output" 的 examples.

- 使用语法约束: 一些本地部署的开源 model, 可以使用 GBNF 等语法文件强制约束 model output, 确保生成的每个 token 都严格符合预定义的 JSON 语法. 这是最严格可靠的非 function calling 方法.

## 3 Function Calling

也称 Tool Calling, 近年来一个重要进展.

### 3.1 概念与工作流程

其本质是一个多轮对话流程, 让 model, code 和 external tools 之间协同工作. 其核心工作流:

1. 定义 tool: 在 code 中以特定 format (通常是 json) 定义好可用的 tools, 包括 tool 的 name, function description 以及需要的 parameters.

2. 用户提问: user 发起一个需要调用 tool 才能回答的 request.

3. 模型决策: model 分析 user 的意图, 匹配合适的工具, 返回一个包含 `tool_calls` 的特殊 response.

4. 代码执行: 从该 response 中解析出 tool name, parameters, 然后在 code 中实际执行该 tool.

5. 结果反馈: 将工具的执行结果包装为一个 rool 为 tool 的 message, 返回给 model.

6. 最终生成: model 结合原始 query 和返回到 tool message, 生成最终 answer.

### 代码示例

[模拟 function calling.](./code/02_function_calling_example.py)

### 3.2 Function Calling 的优势

相较于单纯通过 prompt engineering 而言, 其优势在于:

- 可靠性高: model 的原生能力, 得到的 structured format data 更稳定和精确.

- 意图识别: 能够根据 query 选择最合适的工具.

- 与外部世界交互: 是构建能够实现执行 task 的 agent 的核心基础.