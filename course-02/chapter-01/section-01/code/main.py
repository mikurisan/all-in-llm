from pathlib import Path
from llm import OpenAICompatibleClient
from tools import get_weather, get_attraction

import os
import re
import logging
import sys

sys.path.append(str(Path(__file__).parent))


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

with open("agent.md", "r", encoding="utf-8") as f:
    md_content = f.read()

AGENT_SYSTEM_PROMPT = md_content

llm = OpenAICompatibleClient(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
)

user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"User Query: {user_prompt}"]

logging.info("User Prompt: %s", user_prompt)

for i in range(5):
    logging.info("Loop %d", i+1)
    
    full_prompt = "\n".join(prompt_history)
    
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)

    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            logging.warning("Truncated redundant Thought-Action pairs.")

    logging.info("Model Output:\n%s", llm_output)

    prompt_history.append(llm_output)
    
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = "Error: Failed to parse Action field. Please ensure your response strictly follows the 'Thought: ... Action: ...' format."
        observation_str = f"Observation: {observation}"
        logging.info("Observation:\n%s", observation_str)
        prompt_history.append(observation_str)
        continue

    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        logging.info("Final Answer: %s", final_answer)
        break

    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"Error: Undefined tool '{tool_name}'"

    observation_str = f"Observation: {observation}"
    logging.info("Observation:\n%s", observation_str)
    prompt_history.append(observation_str)