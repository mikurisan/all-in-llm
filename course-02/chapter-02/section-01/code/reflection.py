from typing import List, Dict, Any
from client import BaseLLMClient
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Memory:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        self.records.append({"type": record_type, "content": content})
        logger.info("📝 Memory added record of type '%s'", record_type)

    def get_trajectory(self) -> str:
        trajectory = ""
        
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"--- Last attempt (code) ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- Reviewer feedback ---\n{record['content']}\n\n"
        return trajectory.strip()

    def get_last_execution(self) -> str:
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None

INITIAL_PROMPT_TEMPLATE = """
You are a seasoned Python programmer. Please write a Python function based on the following requirements.

Your code must include a complete function signature, a docstring, and follow PEP 8 coding standards.

Requirements: {task}

Please output only the code, without any additional explanation.
"""

REFLECT_PROMPT_TEMPLATE = """
You are a strict code reviewer and senior algorithm engineer, with an extreme focus on code performance.
Your task is to review the following Python code and focus on identifying its main bottlenecks in **algorithmic efficiency**.

# Original Task:
{task}

# Code to Review:
```python
{code}
```

Please analyze the time complexity of this code and consider whether there is a more **algorithmically efficient** solution to significantly improve performance.

If so, clearly identify the shortcomings of the current algorithm and propose specific, feasible improvements (e.g., using the Sieve of Eratosthenes instead of trial division).

Only respond with "no need for improvement" if the code is already optimal at the algorithmic level.

Please output your feedback directly, without including any additional explanation.
"""

REFINE_PROMPT_TEMPLATE = """
You are a seasoned Python programmer. You are optimizing your code based on feedback from a code review expert.

# Original Task:
{task}

# Last Code Attempt:
{last_code_attempt}

# Reviewer Feedback:
{feedback}

Please generate an optimized version of the code based on the reviewer's feedback.

Your code must include a complete function signature, a docstring, and follow PEP 8 coding standards.

Please output only the code, without any additional explanation.
"""

class ReflectionAgent:
    def __init__(self, llm_client, max_iterations=3):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(self, task: str):
        logger.info("Start to process task: %s", task)

        logger.info("Starting initial execution for task: %s", task)
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record("execution", initial_code)

        for i in range(self.max_iterations):
            logger.info("Starting iteration %d/%d for task: %s", i+1, self.max_iterations, task)

            logger.info("Analyzing code for task: %s", task)
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code)
            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record("reflection", feedback)

            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                logger.info("Reflection indicates no further improvement needed for task: %s", task)
                break

            logger.info("Optimizing code for task: %s", task)
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback
            )
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_code)
        
        final_code = self.memory.get_last_execution()
        logger.info("Task completed. Final generated code:\n%s", final_code)
        return final_code

    def _get_llm_response(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm_client.think(messages=messages) or ""
        return response_text

if __name__ == '__main__':
    llm_client = BaseLLMClient()

    agent = ReflectionAgent(llm_client, max_iterations=2)

    task = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
    agent.run(task)
