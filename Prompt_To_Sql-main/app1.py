from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import os
import sqlite3
import pandas as pd
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)

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

@app.route("/")
def index():
    return render_template("index.html")  # HTML template for upload and input

@app.route("/upload", methods=["POST"])
def upload():
    if "db_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    db_file = request.files["db_file"]
    db_path = "uploaded_file.db"
    db_file.save(db_path)

    schema_df = get_db_schema_df(db_path)
    schema_data = schema_df.to_dict(orient="records")
    return jsonify({"schema": schema_data})

@app.route("/generate_query", methods=["POST"])
def generate_query():
    data = request.json
    db_path = data.get("db_path")
    question = data.get("question")
    schema = data.get("schema")

    if not db_path or not question or not schema:
        return jsonify({"error": "Missing required parameters"}), 400

    prompt = [
        f"""
        You are an expert in converting English questions to SQL queries!
        Here is the database schema for reference:
        {', '.join([f'Table {item["Table"]} with columns {", ".join([col["Column"] for col in schema if col["Table"] == item["Table"]])}' for item in schema])}

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

    sql_command = get_gemini_response(question, prompt)
    response_df = read_sql_query(sql_command, db_path)
    return jsonify({"sql_query": sql_command, "response": response_df.to_dict(orient="records")})

if __name__ == "__main__":
    app.run(debug=True)
