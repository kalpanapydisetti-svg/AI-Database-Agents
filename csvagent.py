from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd

# Load environment variables
load_dotenv()

# OpenRouter API key
openrouter_key = os.getenv("OPENROUTER_API_KEY")

# OpenRouter model
llm_name = "openrouter/free"

# Create model
model = ChatOpenAI(
    api_key=openrouter_key,
    model=llm_name,
    base_url="https://openrouter.ai/api/v1"
)

# Read CSV
df = pd.read_csv("./data/salaries_2023.csv").fillna(value=0)

# Create Pandas dataframe agent
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
)

agent = create_pandas_dataframe_agent(
    llm=model,
    df=df,
    verbose=True,
    allow_dangerous_code=True,
    agent_executor_kwargs={
        "handle_parsing_errors": True
    }
)

# Prompt
CSV_PROMPT_PREFIX = """
You are a careful data analyst working with a pandas dataframe.

First inspect the dataframe columns.

Solve the user's question using pandas calculations on the dataframe.
Do not use prior knowledge or invent values.

When practical, verify important numerical results using a second pandas
calculation method, but do not describe or expose internal tool/action syntax.
"""

CSV_PROMPT_SUFFIX = """
Before giving the final answer, verify that the calculated results are
consistent.

FORMAT NUMBERS OF 4 DIGITS OR MORE WITH COMMAS.

Your final answer must:
- Clearly answer the original question.
- Use Markdown.
- Include the calculated values.
- Include a section titled "Explanation".
- In the Explanation section, mention the dataframe column names used.
- Only report values obtained from calculations performed on the dataframe.
"""

QUESTION = """
Which grade has the highest average base salary,
and compare the average female pay vs male pay?
"""

# Run agent
res = agent.invoke(
    CSV_PROMPT_PREFIX +
    QUESTION +
    CSV_PROMPT_SUFFIX
)

# Print final result
print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)
print(res["output"])