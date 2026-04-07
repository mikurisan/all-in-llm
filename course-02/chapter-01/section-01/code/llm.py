from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

class OpenAICompatibleClient:
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        logger.info("Invoking LLM...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            logger.info("LLM response received successfully.")
            return answer
        except Exception as e:
            logger.error(f"Error occurred while calling LLM API: {e}")
            return "Error: Failed to call language model service."