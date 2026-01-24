from flask import Flask, request, jsonify, render_template
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

app = Flask(__name__)

# Updated: Hugging Face Inference API function with Falcon model
def query_huggingface(prompt):
    API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        try:
            return response.json()[0]["generated_text"]
        except (KeyError, IndexError):
            return "❌ Model returned an unexpected response."
    else:
        return f"Error: {response.status_code} - {response.text}"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.form.get("prompt", "")
    try:
        idea = query_huggingface(prompt)
        return jsonify({"idea": idea})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
