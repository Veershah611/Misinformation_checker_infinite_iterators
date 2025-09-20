# app.py (Corrected and Complete)

import os
import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer # No longer needed
# from sklearn.metrics.pairwise import cosine_similarity # No longer needed
import google.generativeai as genai
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import mimetypes
import datetime
from threading import Lock

app = Flask(__name__)
CORS(app)

# --- Configuration and Setup ---
TRENDS_LOG_FILE = "trends_log.json"
log_lock = Lock()

if not os.path.exists(TRENDS_LOG_FILE):
    with open(TRENDS_LOG_FILE, 'w') as f:
        json.dump([], f)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
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

# --- Core Logic Functions ---
def get_claim_category(claim: str) -> str:
    if not api_key or not claim.strip():
        return "Uncategorized"
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Classify the following claim into one of these categories: Health, Financial Scam, Political, Social, Technology, Other.
        Return only the category name.

        Claim: "{claim}"
        Category:
        """
        response = model.generate_content(prompt)
        category = response.text.strip()
        # Ensure category is one of the predefined ones
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
                json.dump(data, f, indent=4) # Using indent for readability
        except (IOError, json.JSONDecodeError):
            with open(TRENDS_LOG_FILE, 'w') as f:
                json.dump([new_entry], f, indent=4)

def get_fact_check_from_gemini(claim: str, source_type: str, files: list = None) -> dict:
    if not api_key:
        return {"error": "API Key is not configured on the server."}
    
    # Get the real current date to provide as context
    current_date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    # --- THIS PROMPT HAS BEEN CORRECTED TO HANDLE DATES CORRECTLY ---
    prompt = f"""
### ROLE ###
You are a highly advanced AI fact-checking engine. Your ONLY function is to analyze claims against the most recent, reputable news sources available on the web. You must be objective, fast, and precise.

### CONTEXT ###
Today's actual date is: {current_date_str}. You MUST use this as your frame of reference. Do not rely on your internal knowledge cutoff date.

### PRIMARY DIRECTIVE ###
For the user's claim, you must perform a real-time, deep web search focused on news articles relevant to the claim's timeframe. If the claim is about a recent event, focus on the LAST 72 HOURS.

### CLAIM ###
"{claim}"

### STEP-BY-STEP EXECUTION PLAN ###
1.  **Date Analysis:** Analyze the claim for any specific dates. Your web search and verification MUST be relative to that date. **Crucially, you must ignore your internal knowledge cut-off date (e.g., 2023) and rely solely on real-time web search results relevant to the date in the claim.**
2.  **Language Detection:** Identify the language of the claim. You will provide your final JSON response in this language.
3.  **Search Query Formulation:** Formulate multiple, precise English search queries. If there was a date in the claim, use it in your search terms. Create a list of these queries for your evidence log.
4.  **Source Prioritization & Vetting:** Execute the search. You MUST prioritize results from: Reuters, Associated Press (AP), Press Trust of India (PTI), ANI. If nothing is found, expand to other major outlets.
5.  **Evidence Synthesis (English):** Analyze the search results in English. If top-tier sources have NO mention of a major claimed event, state this explicitly as strong evidence that the claim is false.
6.  **Final JSON Generation (Translated):** Construct the final JSON object. Translate ALL text values to the language detected in Step 2.

### STRICT OUTPUT FORMAT ###
You MUST return ONLY a single, valid JSON object. Do not include markdown.

**JSON Schema:**
{{
  "claim_analysis": {{
    "verdict": "A clear, one-word conclusion: 'True', 'False', 'Misleading', 'Partially True', or 'Unverified'.",
    "score": "An integer from 0 (Completely False/No Evidence) to 100 (Verified by multiple top-tier sources).",
    "explanation": "A detailed explanation. START by listing the reputable sources you checked (e.g., 'Based on a real-time search of Reuters, PTI, and The Hindu...'). Then, summarize your findings. The entire explanation must be translated."
  }},
  "categorized_points": {{
    "points_supporting_truthfulness": ["A translated list of distinct supporting facts."],
    "points_refuting_the_claim": ["A translated list of distinct refuting facts."]
  }},
  "risk_assessment": {{
    "possible_consequences": ["A translated list of potential harms."]
  }},
  "public_guidance_and_resources": {{
    "tips_to_identify_similar_scams": ["A translated list of actionable tips."],
    "official_government_resources": {{
      "relevant_agency_website": "URL to an official site.",
      "national_helpline_number": "Official helpline number."
    }}
  }},
  "evidence_log": {{
    "search_queries_used": ["A list of the exact English search queries you formulated."],
    "external_sources": [{{ "source_name": "Source Name (e.g., Reuters).", "url": "Direct URL to the article." }}],
    "image_analysis": {{
        "reverse_search_summary": "A summary of reverse image search findings.",
        "original_context": "Describe the original context of the image if found."
    }}
  }}
}}
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        contents = [prompt]
        if files:
            contents.extend(files)
        response = model.generate_content(contents, generation_config=generation_config)
        result = json.loads(response.text)

        if "claim_analysis" in result and "verdict" in result["claim_analysis"]:
            verdict = result["claim_analysis"]["verdict"]
            if verdict: # Ensure verdict is not empty
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

# --- FIXED /fact-check-url ENDPOINT ---
@app.route("/fact-check-url", methods=["POST"])
def fact_check_url():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Invalid request. 'url' key is missing."}), 400

    url = data["url"]
    claim_context = data.get("claim", "Analyze the content of the provided URL.")

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        page = requests.get(url, headers=headers, timeout=10)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, 'html.parser')
        
        texts = [p.get_text(strip=True) for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'article', 'title'])]
        content = "\n".join(filter(None, texts))
        
        if not content:
            return jsonify({"error": f"Could not extract meaningful text content from the URL."}), 400
        
        full_claim = f"{claim_context}\n\n**Content from URL ({url}):**\n{content[:4000]}"
        
        result = get_fact_check_from_gemini(full_claim, source_type="url")
        return jsonify(result)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch URL: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred while processing the URL: {e}"}), 500

# --- FIXED /fact-check-file ENDPOINT ---
@app.route("/fact-check-file", methods=["POST"])
def fact_check_file():
    claim_text = request.form.get("claim", "Analyze the content of the attached file(s).")
    files = request.files.getlist("files")
    
    if not files:
        return jsonify({"error": "No files were uploaded."}), 400

    gemini_files = []
    for file in files:
        if file and allowed_file(file.filename):
            mime_type = mimetypes.guess_type(file.filename)[0]
            if mime_type:
                gemini_files.append({"mime_type": mime_type, "data": file.read()})
    
    if not gemini_files:
        return jsonify({"error": "Uploaded file types are not supported. Please use JPG, PNG, or PDF."}), 400

    result = get_fact_check_from_gemini(claim_text, source_type="file", files=gemini_files)
    return jsonify(result)

@app.route("/generate-reply", methods=["POST"])
def generate_smart_reply():
    data = request.get_json()
    if not data or "analysis" not in data:
        return jsonify({"error": "Invalid request. 'analysis' key is required."}), 400

    analysis = data["analysis"]
    language = data.get("language", "English")
    analysis_str = json.dumps(analysis, indent=2)

    prompt = f"""
    You are a helpful communication assistant. Based on the provided fact-check analysis, your task is to generate three distinct, polite, and non-confrontational reply templates. These replies will be used by a person to correct misinformation in a group chat or on social media.

    Guidelines:
    - The tone should be helpful and educational, not accusatory.
    - Each reply should be concise and easy to read.
    - One reply should be direct, one should be more gentle/inquisitive, and one should focus on the risks.
    - Crucially, all replies must be in {language}.

    Fact-Check Analysis:
    ```json
    {analysis_str}
    ```

    Required JSON Output Structure:
    Return a single JSON object with a key "replies", which is a list of three strings. Do not wrap it in markdown.
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
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
        if not data:
            return jsonify({"category_counts": {}, "verdict_counts": {}, "reports_over_time": []})
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        category_counts = df['category'].value_counts().to_dict()
        verdict_counts = df['verdict'].value_counts().to_dict()
        
        # Filter for the last 30 days for the time series chart
        recent_df = df[df['timestamp'] >= (pd.Timestamp.now() - pd.Timedelta(days=30))]
        
        # Resample data by day
        reports_over_time = recent_df.set_index('timestamp').resample('D').size().reset_index(name='count')
        reports_over_time['date'] = reports_over_time['timestamp'].dt.strftime('%Y-%m-%d')
        
        return jsonify({
            "category_counts": category_counts,
            "verdict_counts": verdict_counts,
            "reports_over_time": reports_over_time[['date', 'count']].to_dict('records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)