from dotenv import load_dotenv
import streamlit as st
import os
import sqlite3
import pandas as pd
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(question, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt[0], question])
    return response.text

def read_sql_query(sql, db_path):
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(sql, con)
    con.close()
    return df

def get_db_schema_df(db_path):
    schema_data = []
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cur.fetchall()]
    for table in tables:
        cur.execute(f"PRAGMA table_info({table});")
        columns = cur.fetchall()
        for col in columns:
            schema_data.append({"Table": table, "Column": col[1], "Data Type": col[2]})
    con.close()
    return pd.DataFrame(schema_data)

st.set_page_config(page_title="Retrieval of Sql query")
st.title("Text to Query Generater")
st.header("Convert your text to SQL Query")

db_file = st.file_uploader("Upload DB", type="db")
question = st.text_input("Ask your question:", key="input")

if db_file:
    db_path = "new-file.db"
    with open(db_path, "wb") as f:
        f.write(db_file.getbuffer())

    schema_df = get_db_schema_df(db_path)
    st.subheader("DB Schema:")
    st.dataframe(schema_df)

    prompt = [
        f"""
        You are an expert in converting English questions to SQL queries!
        Here is the database schema for reference:
        {', '.join([f'Table {row["Table"]} with columns {", ".join(schema_df[schema_df["Table"] == row["Table"]]["Column"].unique())}' for _, row in schema_df.iterrows()])}

        Use this schema information to accurately construct SQL queries based on the user question.
        
        Instructions:
           1. Accurately analyze the user's natural language question and generate a valid SQL query.
           2. Do NOT include any markdown formatting, backticks (```), or the keyword "sql" in your output.
           3. Return only the SQL query as plain text.
           4. Ensure the generated SQL query is syntactically correct and optimized based on the schema provided.
        
        Example questions:
        - How many records are in the employee table?
        - Show all data for students in the 'Computer Science' course.
        - What are the names of customers who made a purchase in the last month?
        """
    ]

    if st.button("Ask the question"):
        sql_command = get_gemini_response(question, prompt)
        response_df = read_sql_query(sql_command, db_path)
        st.subheader("The Response is")
        st.dataframe(response_df)
        os.remove(db_path)
