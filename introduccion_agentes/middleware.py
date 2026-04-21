from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt

load_dotenv()

@dataclass
class Context:
    user_role: str


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role

    base_prompt = "You are a helpful assistant that can answer questions and help with tasks."

    match user_role:
        case "expert":
            return f"{base_prompt} Provide technical and detailed answers."
        case "beginner":
            return f"{base_prompt} Keep explanations simple and basic"
        case "child":
            return f"{base_prompt} As if explaining to a literal 5 year old"
        case _:
            return base_prompt

agent = create_agent(
    model = "gpt-4.1-mini",
    context_schema = Context,
    middleware = [user_role_prompt],
)


response = agent.invoke({
    "messages": [{"role": "user", "content": "Explain Machine Learning"}]
}, context = Context(user_role="child"))

print(response['messages'][-1].content)