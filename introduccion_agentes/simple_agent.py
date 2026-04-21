import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

@tool('get_weather', description='Get the weather for a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()


agent = create_agent(
    model = "gpt-4.1-mini",
    tools = [get_weather],
    system_prompt = "You are a weather agent who always cracks jokes and makes puns",
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Should I bring an umbrella today in Vienna?"
        }
    ]
})

print(response['messages'][-1].content)