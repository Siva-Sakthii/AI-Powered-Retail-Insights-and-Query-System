from flask import Flask, redirect, url_for
import subprocess
import os

app = Flask(__name__)

def run_streamlit():
    script_path = os.path.join(os.getcwd(), "streamlit_app.py")
    subprocess.Popen(
        ["streamlit", "run", script_path, "--server.port=8501", "--server.address=0.0.0.0"]
    )

@app.route("/")
def home():
    
    return redirect("http://localhost:8501")

if __name__ == "__main__":
    
    run_streamlit()
    
    app.run(host="0.0.0.0", port=5000)
