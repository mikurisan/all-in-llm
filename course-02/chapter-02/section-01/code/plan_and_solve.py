import ast
import logging

from client import BaseLLMClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PLANNER_PROMPT_TEMPLATE = """
As a professional planner, treat the use's question as a task and break it down into an action plan consisting of multiple execution steps.

Ensure that each execution step in the plan is an independent, executable subtask and is strictly arranged in logical order.

Output format must be a Python list, where each element is a string describing a subtask.

Question: {question}

Please strictly follow the following format to output your plan, ```python and ``` as prefix and suffix are necessary:
```python
["step 1", "step 2", "step 3", ...]
```
"""

class Planner:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]
        
        logger.info("Generating plan for question: %s", question)
        response_text = self.llm_client.think(messages=messages) or ""
        logger.info("Plan generated for question: %s\nResponse: %s", question, response_text)

        try:
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            logger.error("❌ Error parsing plan: %s", e)
            logger.error("Original response: %s", response_text)
            return []
        except Exception as e:
            logger.error("❌ Unknown error occurred while parsing plan: %s", e)
            return []

EXECUTOR_PROMPT_TEMPLATE = """
As an executor. Your task is to strictly follow the given plan and solve the problem step by step.

You will receive the original question, the complete plan, and the steps and results that have been completed so far.

Please focus on solving the "current step" and only output the final answer for that step, without outputting any additional explanations or dialogue.

# Original Question:
{question}

# Complete Plan:
{plan}

# History of Steps and Results:
{history}

# Current Step:
{current_step}

Please output only the answer for the "Current Step":
"""

class Executor:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        history = ""
        final_answer = ""
        
        logger.info("Executing plan for question: %s", question)
        for i, step in enumerate(plan, 1):
            logger.info("Executing step %d/%d: %s", i, len(plan), step)
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question, plan=plan, history=history if history else "无", current_step=step
            )
            messages = [{"role": "user", "content": prompt}]
            
            response_text = self.llm_client.think(messages=messages) or ""
            
            history += f"步骤 {i}: {step}\n结果: {response_text}\n\n"
            final_answer = response_text
            logger.info("Step %d completed with result: %s", i, final_answer)
            
        return final_answer

class PlanAndSolveAgent:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question: str):
        logger.info("Starting to process question: %s", question)

        plan = self.planner.plan(question)
        if not plan:
            logger.warning("Failed to generate a valid action plan for question: %s", question)
            return
        
        final_answer = self.executor.execute(question, plan)
        logger.info("Task completed for question: %s\nFinal answer: %s", question, final_answer)

if __name__ == '__main__':
    llm_client = BaseLLMClient()
    agent = PlanAndSolveAgent(llm_client)
    question = "A fruit store sold 15 apples on Monday. The number of apples sold on Tuesday was twice that of Monday. The quantity sold on Wednesday was 5 less than on Tuesday. How many apples were sold in total over these three days?"
    agent.run(question)