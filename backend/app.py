import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import mimetypes
import datetime
from threading import Lock

app = Flask(__name__)

# Configure CORS
cors = CORS(app, resources={
    r"/*": {"origins": "https://misinformation-checker-infinite-ite.vercel.app"}
})

# --- Configuration and Setup ---
TRENDS_LOG_FILE = "trends_log.json"
log_lock = Lock()

if not os.path.exists(TRENDS_LOG_FILE):
    with open(TRENDS_LOG_FILE, 'w') as f:
        json.dump([], f)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Gemini API and Dataset Loading ---
try:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    print("✅ Gemini API configured successfully.")
except Exception as e:
    print(f"🚨 Error configuring Gemini API: {e}")
    api_key = None

df, vectorizer, X = None, None, None
try:
    print("⏳ Loading local datasets...")
    df1 = pd.read_csv("IFND.csv", encoding='latin1', on_bad_lines='skip')
    df2 = pd.read_csv("news_dataset.csv", encoding='latin1', on_bad_lines='skip')
    df1 = df1.rename(columns={"Statement": "text", "Label": "label", "Web": "source"})
    if 'source' not in df2.columns:
        df2['source'] = ''
    df = pd.concat([df1, df2], ignore_index=True)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(df["text"].astype(str))
    print(f"📄 Datasets loaded successfully with {len(df)} statements.")
except Exception as e:
    print(f"❌ Could not load local datasets: {e}")


# --- Core Logic Functions ---
def retrieve_relevant_facts(query: str, top_k: int = 3) -> pd.DataFrame:
    if df is None or vectorizer is None:
        return pd.DataFrame()
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]

def get_claim_category(claim: str) -> str:
    if not api_key or not claim.strip():
        return "Uncategorized"
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"""
        Classify the following claim into one of these categories: 
        Health, Financial Scam, Political, Social, Technology, Other.
        Return only the category name.

        Claim: "{claim}"
        Category:
        """
        response = model.generate_content(prompt)
        category = response.text.strip()
        valid_categories = ["Health", "Financial Scam", "Political", "Social", "Technology", "Other"]
        return category if category in valid_categories else "Other"
    except Exception:
        return "Uncategorized"

def log_trend_data(verdict: str, category: str, source: str):
    with log_lock:
        new_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "verdict": verdict,
            "category": category,
            "source": source
        }
        try:
            with open(TRENDS_LOG_FILE, 'r+') as f:
                data = json.load(f)
                data.append(new_entry)
                f.seek(0)
                json.dump(data, f, indent=4)
        except (IOError, json.JSONDecodeError):
            with open(TRENDS_LOG_FILE, 'w') as f:
                json.dump([new_entry], f, indent=4)

def get_fact_check_from_gemini(claim: str, source_type: str, files: list = None) -> dict:
    if not api_key:
        return {"error": "API Key is not configured on the server."}
    
    facts = retrieve_relevant_facts(claim)
    facts_text = "No relevant facts found in the local dataset."
    if not facts.empty:
        facts_text = "\n".join([f"- {row['text']} (Label: {row['label']})" for _, row in facts.iterrows()])

    prompt = f""" ... (your fact-check prompt here) ... """

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        generation_config = GenerationConfig(response_mime_type="application/json")
        contents = [prompt]
        if files:
            contents.extend(files)
        response = model.generate_content(contents, generation_config=generation_config)
        result = json.loads(response.text)

        if "claim_analysis" in result and "score" in result["claim_analysis"]:
            score = result["claim_analysis"]["score"]
            if isinstance(score, float) and 0.0 <= score <= 1.0:
                result["claim_analysis"]["score"] = int(score * 100)

        if "claim_analysis" in result and "verdict" in result["claim_analysis"]:
            verdict = result["claim_analysis"]["verdict"]
            if verdict:
                category = get_claim_category(claim)
                log_trend_data(verdict, category, source_type)
        
        return result
            
    except Exception as e:
        return {"error": f"An API or other error occurred: {e}"}

# --- API Routes ---
@app.route("/fact-check-text", methods=["POST"])
def fact_check_text():
    data = request.get_json()
    if not data or "claim" not in data:
        return jsonify({"error": "Invalid request. 'claim' key is missing."}), 400
    
    claim = data["claim"]
    result = get_fact_check_from_gemini(claim, source_type="text")
    return jsonify(result)

@app.route("/fact-check-url", methods=["POST"])
def fact_check_url():
    # TODO: implement scraping + Gemini call
    return jsonify({"error": "Not implemented yet."}), 501

@app.route("/fact-check-file", methods=["POST"])
def fact_check_file():
    # TODO: implement file handling + Gemini call
    return jsonify({"error": "Not implemented yet."}), 501

@app.route("/generate-reply", methods=["POST"])
def generate_smart_reply():
    data = request.get_json()
    if not data or "analysis" not in data:
        return jsonify({"error": "Invalid request. 'analysis' key is required."}), 400

    analysis = data["analysis"]
    language = data.get("language", "English")
    analysis_str = json.dumps(analysis, indent=2)

    prompt = f""" ... (your smart reply prompt here) ... """
    
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        generation_config = GenerationConfig(response_mime_type="application/json")
        response = model.generate_content(prompt, generation_config=generation_config)
        result = json.loads(response.text)
        return jsonify(result)
        
    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse JSON response from the model."}), 500
    except Exception as e:
        return jsonify({"error": f"An API or other error occurred: {e}"}), 500

@app.route("/api/trends")
def get_trends():
    try:
        with open(TRENDS_LOG_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception:
        return jsonify([])

if __name__ == "__main__":
    app.run(debug=True, port=5000)
