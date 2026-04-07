import requests

import os
from tavily import TavilyClient

def get_weather(city: str) -> str:
    """Call the wttr.in API to get real-time weather information."""
    url = f"https://wttr.in/{city}?format=j1"
    
    try:
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()
        
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']
        
        return f"{city} current weather: {weather_desc}, temperature {temp_c}°C"

    except requests.exceptions.RequestException as e:
        return f"Error: Encountered a network issue while querying weather - {e}"
    except (KeyError, IndexError) as e:
        return f"Error: Failed to parse weather data, possibly due to an invalid city name - {e}"

def get_attraction(city: str, weather: str) -> str:
    """Find the best tourist attractions in a city based on the current weather using Tavily."""

    api_key = os.environ.get("TAVILY_API_KEY")

    if not api_key:
        return "Error: TAVILY_API_KEY is not configured."

    tavily = TavilyClient(api_key=api_key)
    
    query = f"'{city}' in '{weather}' weather conditions - best tourist attractions to visit and reasons why they are recommended."
    
    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        
        if response.get("answer"):
            return response["answer"]
        
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
        
        if not formatted_results:
             return "Sorry, no relevant tourist attraction recommendations found."

        return "Based on the search, here are some recommendations:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"Error: An issue occurred while searching for tourist attractions - {e}"
