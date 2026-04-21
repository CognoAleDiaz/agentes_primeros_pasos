from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model = "gpt-4.1-mini",
    temperature = 0.1
)

conversation = [
    SystemMessage(content="You are a helpful assistant that can answer questions and help with tasks."),
    HumanMessage(content="What is Python?"),
    AIMessage(content="Python is a programming language that can be used to create web applications, desktop applications, and mobile applications."),
    HumanMessage(content="When was python made?"),
]

response = model.invoke(conversation)

print(response.content)