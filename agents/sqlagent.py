import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

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
     max_tokens=4096,
)

# ============================================================
# 3. CONNECT TO SQLITE DATABASE
# ============================================================

db = SQLDatabase.from_uri(
    "sqlite:///./db/salaries.db"
)

# ============================================================
# 4. CREATE SQL AGENT
# ============================================================

agent = create_sql_agent(
    llm=model,
    db=db,
    verbose=True,
    agent_executor_kwargs={
        "handle_parsing_errors": True,
    },
)

# ============================================================
# 5. STREAMLIT UI
# ============================================================

st.title("Database AI Agent with LangChain")

st.write("### Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", "10,291")
col2.metric("Columns", "8")
col3.metric("Model", "DeepSeek")

st.write("### Database")

st.write("**Database:** `salaries.db`")
st.write("**Table:** `salaries`")

st.write("### Available Fields")

columns = [
    "Department",
    "Department_Name",
    "Division",
    "Gender",
    "Base_Salary",
    "Overtime_Pay",
    "Longevity_Pay",
    "Grade",
]

for column in columns:
    st.write(f"• `{column}`")

# ============================================================
# 6. ASK A QUESTION
# ============================================================

st.write("### Ask a Question")

QUESTION = """
Which grade has the highest average Base_Salary?
"""

question = st.text_input(
    "Enter your question about the database:",
    QUESTION,
)

# ============================================================
# 7. RUN SQL AGENT
# ============================================================

if st.button("Run Query"):

    with st.spinner("Analyzing the database..."):

        try:
            result = agent.invoke(question)

            st.write("### Final Answer")

            st.markdown(result["output"])

        except Exception as e:

            st.error("An error occurred while running the SQL Agent.")

            st.write("Error type:", type(e).__name__)
            st.write("Error:", str(e))