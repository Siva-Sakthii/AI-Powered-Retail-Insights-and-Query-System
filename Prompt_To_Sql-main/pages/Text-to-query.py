from dotenv import load_dotenv
import streamlit as st
import os
import sqlite3
import pandas as pd
import google.generativeai as genai
import time
import io

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(question, prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt[0], question])
        return response.text
    except Exception as e:
        st.error(f"Error generating SQL query: {e}")
        return None

def read_sql_query(sql, db_path):
    start_time = time.time()
    try:
        con = sqlite3.connect(db_path)
        df = pd.read_sql_query(sql, con)
    except Exception as e:
        st.error(f"Error executing SQL query: {e}")
        return None, None
    finally:
        con.close()
    execution_time = time.time() - start_time
    return df, execution_time

def validate_database(db_path):
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cur.fetchall()
        if not tables:
            raise ValueError("The database does not contain any tables.")
        return True
    except sqlite3.DatabaseError:
        st.error("The uploaded file is not a valid SQLite database.")
        return False
    except Exception as e:
        st.error(f"Unexpected error validating database: {e}")
        return False
    finally:
        con.close()

def validate_sql_syntax(sql, db_path):
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(f"EXPLAIN {sql}")
        return True
    except sqlite3.OperationalError as e:
        st.error(f"SQL syntax error: {e}")
        return False
    except Exception as e:
        st.error(f"Unexpected error validating SQL: {e}")
        return False
    finally:
        con.close()

def get_db_schema_df(db_path):
    try:
        schema_data = []
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cur.fetchall()]
        if not tables:
            raise ValueError("The database does not contain any tables.")
        for table in tables:
            cur.execute(f"PRAGMA table_info({table});")
            columns = cur.fetchall()
            for col in columns:
                schema_data.append({"Table": table, "Column": col[1], "Data Type": col[2]})
        return pd.DataFrame(schema_data)
    except Exception as e:
        st.error(f"Error retrieving database schema: {e}")
        return pd.DataFrame()
    finally:
        con.close()

# Streamlit UI
st.set_page_config(page_title="SQL Query Generator")
st.title("Text to SQL Query Generator")
st.header("Convert your text to SQL Query")

db_file = st.file_uploader("Upload SQLite Database", type="db")
question = st.text_input("Enter your question:")

if db_file:
    db_path = "uploaded_file.db"
    with open(db_path, "wb") as f:
        f.write(db_file.getbuffer())

    if validate_database(db_path):
        schema_df = get_db_schema_df(db_path)
        if not schema_df.empty:
            st.subheader("Database Schema:")
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
                        - How many products are available in the warehouse?
                        - Show all customers who made purchases in the last month.
                        - List the products purchased by a specific customer.
                        - Find employees whose contact number starts with '98'.
                        - Display the details of offers active during a given period.
                """
            ]

            if st.button("Generate SQL Query"):
                with st.spinner("Generating SQL query..."):
                    sql_command = get_gemini_response(question, prompt)
                if sql_command and validate_sql_syntax(sql_command, db_path):
                    st.write("Generated SQL Query:")
                    st.code(sql_command)

                    with st.spinner("Fetching query results..."):
                        response_df, execution_time = read_sql_query(sql_command, db_path)

                    if response_df is not None:
                        # Save DataFrame to session state
                        st.session_state['response_df'] = response_df
                        st.session_state['execution_time'] = execution_time

                        st.subheader("Query Results:")
                        st.dataframe(response_df)

# Sorting and Display
if 'response_df' in st.session_state:
    st.subheader("Filter and Sort Results")
    response_df = st.session_state['response_df']

    # Dropdown to select sort column
    columns = response_df.columns.tolist()
    sort_by = st.selectbox("Sort by", ["None"] + columns, key="sort_by")

    # Checkbox for sort order
    ascending = st.checkbox("Sort in Ascending Order", value=True, key="ascending")

    # Sort DataFrame
    sorted_df = response_df
    if sort_by != "None":
        sorted_df = response_df.sort_values(by=sort_by, ascending=ascending)

    # Display sorted results
    st.write("Sorted DataFrame:")
    st.dataframe(sorted_df)

    # Download sorted results
    st.subheader("Download Sorted Results")
    csv = sorted_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Sorted Results as CSV", 
        data=csv, 
        file_name="sorted_query_results.csv", 
        mime="text/csv"
    )

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        sorted_df.to_excel(writer, index=False)
    excel_buffer.seek(0)
    st.download_button(
        "Download Sorted Results as Excel", 
        data=excel_buffer, 
        file_name="sorted_query_results.xlsx", 
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.write(f"Query Execution Time: {st.session_state['execution_time']:.2f} seconds")
