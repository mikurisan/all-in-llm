Letting LLMs output secific structured format data (e.g., JSON, XML, etc.)

## 1 Why do we need formatted generation?

What can I say?

## 2 Implementation methods for formatted generation

### 2.1 Output Parsers

In LangChain, a component designed to handle LLM output. The core concept is injecting "instructions for formatting the output" into the prompt and parsing the returned text into the expected structured data.

Few examples:

- `StrOutputParser`: Return the output as string.

- `JsonOutputParser`: Parses the output into a JSON string.

- `PydanticOutputParser`: Integrates with Pydantic models to achieve strict definition and validation of output format.

### Example Code

[Analyzing how `PydanticOutputParser` works. ](./code/01_pydantic.py)

### 2.2 Output Parsing in LlamaIndex

Threre are 2 core compoenents that work closely together: **Response Synthesis** and **Structured Output**.

When retrieving docs for RAG, the systm fetches a series of text chunks(nodes). These aren't just concatenated together. The **Response Synthesizer** takes these nodes and the original query, then passes them to the LLM in smarter ways, such as:

- **Refine Mode**: Processes information chunk by chunk and iteratively improves the answer.

- **Compact Mode**: Compresses as many nodes as possible into a single LLM call.

The goal here is to produce a high-quality text answer.

When structured output is required, Pydantic programs are used. This approach is similar to LangChain:

- **Define Schema**: Create a Pydantic model.

- **Guide Generation**: Convert a Pydantic model into a format instruction that the LLM understands. If the LLM supports function calling, that will be used first.

- **Parse & Validate**: Parse the text output, validate it with the Pydantic model, and return a Pydantic object instance.

### 2.3 Simple Implementation without Frameworks

The main approach is through prompt engineering:

- **Explicitly Request JSON Format**: Directly and firmly instruct the model to return JSON format only, without any explanatory text.

- **Provide Json Schema**: Include a specified json schema in the prompt, describing the meaning and type of each key.

- **Provide Few-Shot Examples**: Give several examples of "input -> json format output".

- **Use Grammer Constraits**: For some locally deployed open-source models, you can use grammer files (like GBNF) to strictly constrain the model's output, ensuring every generated token strictly complies with predefined JSON grammar. This is the most rigorous and reliable method without using function calling.

> The essence of frameworks is also about prompt engineering.

## 3 Function Calling

As known as Tool Calling, a recent major development.

### 3.1 Concept and Workflow

It's essentially a multi-turn conversation process that corordinates the model, code and external tools. The main workflow is:

1. **Define Tools**: In code, define available tools in a specific format (usually json), including the tool's name, function description, and required parameters.

2. **User Query**: A user makes a request that requires a tool to answer.

3. **Model Decision**: The model analyzes the user's intent, mataches a suitable tool, and returns a special response containing `tool_calls`.

4. **Code Execution**: Parse the tool name and parameters from that response, and actually execute that tool in the code.

5. **Result Feedback**: Package the tool's execution results as a meesage with the role `tool` and return it to the model.

6. **Final Generation**: The model combines the original query and the returned tool message to generate the final answer.

### Example Code

[Simulating function calling.](./code/02_function_calling_example.py)

### 3.2 Adavantage of Function Calling

Compared to relying solely on prompt engineering, its advantages include:

- **High Reliability**: It's a native capability of the model, resulting in more stable and precise structured data output.

- **Intent Recognition**: Can select the best appropriate tool based on the user's query.

- **Interaction with the External World**: Forms the core fundation for building agents capable of executing real-world tasks.

> Function calling involves fine-tuning a model to give it this capability.