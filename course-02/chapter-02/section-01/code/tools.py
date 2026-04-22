import os
import logging

from serpapi import SerpApiClient
from typing import Dict, Any

logger = logging.getLogger(__name__)

def search(query: str) -> str:

    logger.info("🔍 Executing [SerpApi] search: %s", query)

    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "❌ SERPAPI_API_KEY is not set in environment variables."

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "us",
            "hl": "en",
        }
        
        client = SerpApiClient(params)
        results = client.get_dict()
        
        # Extracting relevant information from the results
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"❌ Sorry, couldn't find relevant information for '{query}'."

    except Exception as e:
        return f"❌ Error occurred while searching: {e}"

class ToolExecutor:

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, name: str, description: str, func: callable):

        if name in self.tools:
            logger.warning("Tool '%s' already exists and will be overwritten.", name)

        self.tools[name] = {"description": description, "func": func}
        logger.info("Tool '%s' has been registered.", name)

    def get_tool(self, name: str) -> callable:
        return self.tools.get(name, {}).get("func")

    def get_available_tools(self) -> str:
        return "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
        ])

if __name__ == '__main__':
    tool_executor = ToolExecutor()

    search_description = "A tool to perform web searches using SerpApi. Input should be a search query string."
    tool_executor.register_tool("Search", search_description, search)

    print(f"Available Tools:\n{tool_executor.get_available_tools()}")

    tool_name = "Search"
    tool_input = "What is the latest GPU model from NVIDIA?"

    tool_function = tool_executor.get_tool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print(f"Observation: {observation}")
    else:
        print(f"Error: Tool named '{tool_name}' not found.")