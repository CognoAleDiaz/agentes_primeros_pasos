from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model = "gpt-4.1-mini",
    temperature = 0.1
)

response = model.invoke("What do you know about the company Cognodata?")

print(response.content)