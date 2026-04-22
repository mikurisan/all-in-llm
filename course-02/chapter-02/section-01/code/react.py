import re
import logging
from client import BaseLLMClient
from tools import ToolExecutor, search
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REACT_PROMPT_TEMPLATE = """
You have become an intelligent assistant with the ability to call external tools.

System information:

- Current date and time: {current_datetime}

Available tools:
{tools}

Must strictly follow the following format for your response:

Thought: Current thought process. Understand the question, analyze the current situation, break down tasks, plan the next steps.

Action: The action to be taken must comply with one of the following formats:

- `{{tool_name}}[{{tool_input}}]`: Call an external tool, where `tool_name` is the name of the tool to call, and `tool_input` is the input for that tool.

- `Finish[Final Answer]`: When you have gathered enough information to answer the user's final question, you must use `Finish[Final Answer]` after the `Action:` field to output the final answer.

Now, please start solving the following problem based on the question and the history of your previous actions and observations:

- History: {history}

- Question: {question}
"""


class ReActAgent:
    def __init__(self, llm_client: BaseLLMClient, tool_executor: ToolExecutor, max_steps: int = 10):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1

            logger.info("Step %d", current_step)

            tools_desc = self.tool_executor.get_available_tools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str,
                current_datetime=datetime.now()
            )

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                logger.error("Cannot get valid response from LLM."); break

            thought, action = self._parse_output(response_text)
            if thought:
                print(f"🤔 Thought: {thought}")
            if not action:
                logger.warning("Failed to parse valid Action, terminating process."); break

            if action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                print(f"🎉 Final Answer: {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                self.history.append("Observation: Invalid Action format, please check."); continue

            print(f"🎬 Action: {tool_name}[{tool_input}]")
            tool_function = self.tool_executor.get_tool(tool_name)
            observation = tool_function(tool_input) if tool_function else f"Error: Tool '{tool_name}' not found."
            
            print(f"👀 Observation: {observation}")
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        logger.info("Reached maximum steps without finding a final answer.");
        return None

    def _parse_output(self, text: str):
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        return (match.group(1), match.group(2)) if match else (None, None)

    def _parse_action_input(self, action_text: str):
        match = re.match(r"\w+\[(.*)\]", action_text, re.DOTALL)
        return match.group(1) if match else ""

if __name__ == '__main__':
    llm = BaseLLMClient()
    tool_executor = ToolExecutor()
    search_desc = (
        "A web search engine. "
        "You should use this tool when you need to answer questions about "
        "current events, facts, and information not found in your knowledge base."
    )
    tool_executor.register_tool("Search", search_desc, search)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "What is the latest iPhone model? What are its main selling points?"
    agent.run(question)