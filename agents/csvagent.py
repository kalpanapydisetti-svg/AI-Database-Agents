import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
)


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


# ============================================================
# 2. OPENROUTER MODEL
# ============================================================

llm_name = "deepseek/deepseek-v4.1-flash"

model = ChatOpenAI(
    api_key=openrouter_key,
    model=llm_name,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)
test_response = model.invoke(
    "Reply with exactly: DeepSeek connection successful"
)

print("MODEL TEST:")
print(test_response.content)

# ============================================================
# 3. LOAD CSV
# ============================================================

csv_file = "./data/salaries_2023.csv"

df = pd.read_csv(csv_file).fillna(0)

# Store original dataset information
original_column_count = len(df.columns)
original_columns = df.columns.tolist()


# ============================================================
# 4. CREATE PANDAS DATAFRAME AGENT
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


# ============================================================
# 5. DEFAULT QUESTION
# ============================================================

QUESTION = """
Which grade has the highest average Base_Salary,
and compare the average total pay between female and male employees,
where total pay = Base_Salary + Overtime_Pay + Longevity_Pay?
"""


# ============================================================
# 6. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Database AI Agent",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# 7. TITLE
# ============================================================

st.title("📊 Database AI Agent with LangChain")

st.write(
    "Ask questions about the salaries_2023.csv dataset "
    "using natural language."
)


# ============================================================
# 8. DATASET INFORMATION
# ============================================================

st.write("### Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Rows",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Columns",
        original_column_count
    )

with col3:
    st.metric(
        "Model",
        "DeepSeek"
    )


# ============================================================
# 9. DATASET PREVIEW
# ============================================================

st.write("### Dataset Preview")

st.dataframe(
    df.head(),
    use_container_width=True
)


# ============================================================
# 10. COLUMN NAMES
# ============================================================

with st.expander("View Column Names"):

    for column in original_columns:
        st.write(f"• `{column}`")


# ============================================================
# 11. USER QUESTION
# ============================================================

st.write("### Ask a Question")

question = st.text_input(
    "Enter your question about the dataset:",
    QUESTION,
)


# ============================================================
# 12. RUN QUERY
# ============================================================

if st.button(
    "🚀 Run Query",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        prompt = f"""
You are a Pandas data analyst.

Question:
{question}

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

        with st.spinner(
            "Analyzing the dataset..."
        ):

            try:

                res = agent.invoke(prompt)

                st.write("### Final Answer")

                st.markdown(
                    res["output"]
                )

            except Exception as e:

                st.error(
                    f"Agent error: {type(e).__name__}: {e}"
                )


# ============================================================
# 13. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📁 Dataset")

    st.write(
        "**File:** `salaries_2023.csv`"
    )

    st.write(
        f"**Rows:** {len(df):,}"
    )

    st.write(
        f"**Columns:** {original_column_count}"
    )

    st.divider()

    st.write("### Available Fields")

    for column in original_columns:

        st.write(
            f"• `{column}`"
        )