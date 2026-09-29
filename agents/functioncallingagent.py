import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text

load_dotenv()

# --------------------------------------------------
# OpenRouter
# --------------------------------------------------

openrouter_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_key:
    raise ValueError("OPENROUTER_API_KEY was not found. Check your .env file.")

llm_name = "deepseek/deepseek-v4.1-flash"

model = ChatOpenAI(
    api_key=openrouter_key,
    model=llm_name,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=4096,
)

# --------------------------------------------------
# Database
# --------------------------------------------------

engine = create_engine("sqlite:///./db/salaries.db")


# --------------------------------------------------
# Functions
# --------------------------------------------------

def get_employee_count():
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT COUNT(*) FROM salaries")
        ).scalar()


def get_average_base_salary():
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT AVG(Base_Salary) FROM salaries")
        ).scalar()

    return round(result, 2)


def get_employee_count_by_department():
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT Department, COUNT(*) AS Employee_Count
                FROM salaries
                GROUP BY Department
                ORDER BY Employee_Count DESC
            """)
        ).fetchall()

    return [
        {
            "Department": row[0],
            "Employee_Count": row[1]
        }
        for row in result
    ]


def get_average_salary_by_gender():
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT Gender, AVG(Base_Salary) AS Average_Salary
                FROM salaries
                GROUP BY Gender
            """)
        ).fetchall()

    return [
        {
            "Gender": row[0],
            "Average_Salary": round(row[1], 2)
        }
        for row in result
    ]


def get_highest_salary():
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT MAX(Base_Salary) FROM salaries")
        ).scalar()


def get_lowest_salary():
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT MIN(Base_Salary) FROM salaries")
        ).scalar()


def get_highest_salary_by_department():
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT Department, MAX(Base_Salary) AS Highest_Salary
                FROM salaries
                GROUP BY Department
                ORDER BY Highest_Salary DESC
            """)
        ).fetchall()

    return [
        {
            "Department": row[0],
            "Highest_Salary": row[1]
        }
        for row in result
    ]


def get_average_salary_by_department():
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT Department, AVG(Base_Salary) AS Average_Salary
                FROM salaries
                GROUP BY Department
                ORDER BY Average_Salary DESC
            """)
        ).fetchall()

    return [
        {
            "Department": row[0],
            "Average_Salary": round(row[1], 2)
        }
        for row in result
    ]


# --------------------------------------------------
# Tools
# --------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_employee_count",
            "description": "Get the total number of employees in the dataset.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_average_base_salary",
            "description": "Get the average Base_Salary of all employees.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_employee_count_by_department",
            "description": "Get the number of employees in each department.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_average_salary_by_gender",
            "description": "Get the average Base_Salary for male and female employees.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_highest_salary",
            "description": "Get the highest Base_Salary in the dataset.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_lowest_salary",
            "description": "Get the lowest Base_Salary in the dataset.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_highest_salary_by_department",
            "description": "Get the highest Base_Salary for each department.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_average_salary_by_department",
            "description": "Get the average Base_Salary for each department.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

model_with_tools = model.bind_tools(tools)


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("Function Calling Agent")

st.write("### Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", "10,291")
col2.metric("Columns", "8")
col3.metric("Model", "DeepSeek")

st.write("### Database")

st.write("**Database:** `salaries.db`")
st.write("**Table:** `salaries`")

st.write("### Available Functions")

functions = [
    "Employee Count",
    "Average Base Salary",
    "Employee Count by Department",
    "Average Salary by Gender",
    "Highest Salary",
    "Lowest Salary",
    "Highest Salary by Department",
    "Average Salary by Department"
]

for function in functions:
    st.write(f"• {function}")


# --------------------------------------------------
# Ask Question
# --------------------------------------------------

st.write("### Ask a Question")

QUESTION = "What is the highest salary?"

question = st.text_input(
    "Enter your question:",
    QUESTION
)


# --------------------------------------------------
# Run Agent
# --------------------------------------------------

if st.button("Run Agent"):

    with st.spinner("Analyzing your question..."):

        try:

            response = model_with_tools.invoke(question)

            tool_calls = response.additional_kwargs.get(
                "tool_calls",
                []
            )

            # ------------------------------------------
            # Function calling
            # ------------------------------------------

            if tool_calls:

                tool_call = tool_calls[0]

                function_name = tool_call["function"]["name"]

                st.write("### Function Selected")

                st.code(function_name)


                if function_name == "get_employee_count":
                    result = get_employee_count()

                elif function_name == "get_average_base_salary":
                    result = get_average_base_salary()

                elif function_name == "get_employee_count_by_department":
                    result = get_employee_count_by_department()

                elif function_name == "get_average_salary_by_gender":
                    result = get_average_salary_by_gender()

                elif function_name == "get_highest_salary":
                    result = get_highest_salary()

                elif function_name == "get_lowest_salary":
                    result = get_lowest_salary()

                elif function_name == "get_highest_salary_by_department":
                    result = get_highest_salary_by_department()

                elif function_name == "get_average_salary_by_department":
                    result = get_average_salary_by_department()

                else:
                    result = "Unknown function"


                st.write("### Function Result")

                if isinstance(result, list):
                    st.dataframe(result)

                else:
                    st.write(result)


                # --------------------------------------
                # Final LLM answer
                # --------------------------------------

                final_prompt = f"""
The user asked:

{question}

The function {function_name} was executed.

The function returned:

{result}

Provide a clear and concise answer to the user.
"""

                final_response = model.invoke(final_prompt)

                st.write("### Final Answer")

                st.markdown(final_response.content)


            # ------------------------------------------
            # Normal question
            # ------------------------------------------

            else:

                st.write("### Final Answer")

                st.markdown(response.content)


        except Exception as e:

            st.error("An error occurred.")

            st.write("Error type:", type(e).__name__)

            st.write("Error:", str(e))