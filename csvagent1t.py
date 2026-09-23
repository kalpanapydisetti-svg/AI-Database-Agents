import os
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
)

print("CSV AGENT STARTED")

# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

openrouter_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_key:
    raise ValueError(
        "OPENROUTER_API_KEY was not found. "
        "Check your .env file."
    )

print("OpenRouter API key found")


# ============================================================
# 2. OPENROUTER MODEL
# ============================================================

llm_name = "minimax/minimax-m3:free"

model = ChatOpenAI(
    api_key=openrouter_key,
    model=llm_name,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

print(f"Model selected: {llm_name}")


# ============================================================
# 3. LOAD CSV
# ============================================================

csv_file = "./data/salaries_2023.csv"

df = pd.read_csv(csv_file).fillna(0)

print("CSV loaded successfully")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print("Column names:")
print(list(df.columns))


# ============================================================
# 4. CREATE PANDAS AGENT
# ============================================================

agent = create_pandas_dataframe_agent(
    llm=model,
    df=df,
    verbose=True,
    allow_dangerous_code=True,
    agent_executor_kwargs={
        "handle_parsing_errors": True,
    },
)

print("Pandas agent created successfully")


# ============================================================
# 5. QUESTION
# ============================================================

QUESTION = """
Which grade has the highest average Base_Salary,
and compare the average total pay between female and male employees,
where total pay = Base_Salary + Overtime_Pay + Longevity_Pay?
"""

PROMPT = f"""
You are a Pandas data analyst.

Question:
{QUESTION}

Use the DataFrame to calculate the answer.

IMPORTANT RULES:
1. Use Python/Pandas for calculations.
2. Python tool inputs must contain ONLY valid Python code.
3. Never put explanations inside a Python tool call.
4. Never use Unicode mathematical symbols in Python code.
5. Do not calculate the same thing repeatedly.
6. After obtaining the required values, STOP using tools.
7. Return exactly ONE final answer.
8. Do not output a second Final Answer.
9. Do not make another tool call after your final answer.
10. Do not guess any values.
"""


# ============================================================
# 6. RUN AGENT
# ============================================================

print("\nRunning AI agent...\n")

try:

    result = agent.invoke(PROMPT)

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(result["output"])

except Exception as e:

    print("\n" + "=" * 70)
    print("ERROR")
    print("=" * 70)

    print("Error type:", type(e).__name__)
    print("Error:", e)