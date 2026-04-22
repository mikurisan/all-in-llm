import os
import logging

from openai import OpenAI
from typing import List, Dict

logger = logging.getLogger(__name__)

class BaseLLMClient:

    def __init__(self, model: str = None, api_key: str = None, base_url: str = None, timeout: int = 60):
        self.model = model or os.getenv("MODEL_NAME")
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")
        self.timeout = timeout

        if not all([self.model, self.api_key, self.base_url]):
            raise ValueError("Please ensure MODEL_NAME, API_KEY, and BASE_URL are set in the environment variables.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:

        logger.info("🧠 Calling model: %s", self.model)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            logger.info("✅ Model response received.")

            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                content = delta.get("content") if isinstance(delta, dict) else delta.content

                if content:
                    logger.debug("📝 Model output: %s", content)
                    collected_content.append(content)

            return "".join(collected_content)

        except Exception as e:
            logger.error("❌ Error occurred while calling LLM API: %s", e)
            return None

if __name__ == '__main__':
    llm = BaseLLMClient()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that writes Python code."},
        {"role": "user", "content": "Write a Python function that takes a list of numbers and returns the sum of those numbers."}
    ]
    
    response = llm.think(messages)
    if response:
        print(response)
