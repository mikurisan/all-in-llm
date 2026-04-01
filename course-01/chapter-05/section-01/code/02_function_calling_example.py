from openai import OpenAI
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL", ""),
)

def send_messages(messages, tools=None):
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", ""),
        messages=messages,
        tools=tools,
        # Let the model decide when to use tools
        tool_choice="auto",
    )
    return response.choices[0].message

# Define function schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定地点的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市和省份，例如：杭州市, 浙江省",
                    }
                },
                "required": ["location"]
            },
        }
    },
]

messages = [{"role": "user", "content": "杭州今天天气怎么样？"}]
logging.info(f"User> {messages[0]['content']}")
message = send_messages(messages, tools=tools)

if message.tool_calls:
    logging.info(f"message 内容: {message.content}")
    tool_call = message.tool_calls[0]
    function_info = tool_call.function
    logging.info(f"工具名称: {function_info.name}")
    logging.info(f"工具参数: {function_info.arguments}")
    # Add the model's response (including any tool call requests)
    # to the message history.
    messages.append(message)

    # Simulating tool execution
    tool_output = "24℃，晴朗"
    logging.info(f"工具执行结果: {tool_output}")

    # Add the tool's execution resutls as a new message to the history
    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_output})

    final_message = send_messages(messages, tools=tools)
    logging.info(f"Model> {final_message.content}")
else:
    logging.info(f"Model> {message.content}")