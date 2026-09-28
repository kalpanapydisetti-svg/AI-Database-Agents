import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

openrouter_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env")

llm = ChatOpenAI(
    api_key=openrouter_key,
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1"
)

response = llm.invoke("What is a database agent? Explain in simple words.")

print("\nAI Response:\n")
print(response.content)