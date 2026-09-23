from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

openrouter_key = os.getenv("OPENROUTER_API_KEY")

llm_name ="openrouter/free"

model = ChatOpenAI(api_key=openrouter_key, model=llm_name,base_url ="https://openrouter.ai/api/v1")

messages = [
    SystemMessage(
        content="You are a helpful assistant who is extremely competent as a Computer Scientist! Your name is Rob."
    ),
    HumanMessage(content="who was the very first computer scientist?"),
]


# res = model.invoke(messages)
# print(res)


def first_agent(messages):
    res = model.invoke(messages)
    return res


def run_agent():
    print("Simple AI Agent: Type 'exit' to quit")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        print("AI Agent is thinking...")
        messages = [HumanMessage(content=user_input)]
        response = first_agent(messages)
        print("AI Agent: getting the response...")
        print(f"AI Agent: {response.content}")


if __name__ == "__main__":
    run_agent()