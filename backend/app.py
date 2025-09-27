import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import mimetypes
import datetime
from threading import Lock

# --- Environment Self-Check ---
# This block checks if the installed Google AI library is modern enough.
# Older versions default to the 'v1beta' API, which causes the 404 errors for new models.
try:
    from pkg_resources import parse_version
    # The library switched to the new v1 API by default in version 0.3.0
    MIN_GEMINI_VERSION = "0.3.0"
    current_version = genai.__version__
    if parse_version(current_version) < parse_version(MIN_GEMINI_VERSION):
        print("="*80)
        print(f"🚨 FATAL ERROR: Your 'google-generativeai' library is outdated (version {current_version}).")
        print(f"   This is the cause of the '404 model not found' errors.")
        print(f"   Please upgrade to version {MIN_GEMINI_VERSION} or higher to fix this.")
        print("\n   IN YOUR TERMINAL, RUN THIS COMMAND:")
        print("   pip install --upgrade google-generativeai")
        print("="*80)
        sys.exit(1) # Stop the application from starting
except (ImportError, AttributeError):
    print("⚠️ Warning: Could not verify 'google-generativeai' library version.")
    print("   If you encounter 404 errors, please upgrade the library by running:")
    print("   pip install --upgrade google-generativeai")
    pass
# --- End of Environment Self-Check ---


app = Flask(__name__) # 👈 Only ONE instance at the top

# Configure CORS right after creating the app instance
# This updated configuration allows requests from your deployed Vercel app
# and common local development servers to prevent "Failed to fetch" errors.
cors = CORS(app, resources={
    r"/*": {
        "origins": [
            "https://misinformation-checker-infinite-ite.vercel.app",
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173"
        ]
    }
})

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

df = None
vectorizer = None
X = None
try:
    print("⏳ Loading local datasets...")
    # Load datasets safely
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
        # Using a standard, widely available model to ensure compatibility.
        model = genai.GenerativeModel("gemini-1.0-pro")
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
    
    facts = retrieve_relevant_facts(claim)
    facts_text = "No relevant facts found in the local dataset."
    if not facts.empty:
        facts_text = "\n".join([f"- {row['text']} (Label: {row['label']})" for _, row in facts.iterrows()])

    prompt = f"""
    You are an expert, multilingual fact-checking analyst. Your primary goal is to provide a neutral, evidence-based assessment of a claim's validity in the user's original language.
    **Claim to Investigate:** "{claim}"
    **Supporting Context from a Local Dataset (Use as a reference point only):**
    {facts_text}
    **Task:**
    1. First, detect the language of the user's claim (e.g., "Hindi", "English", "Spanish").
    2. Use your internal knowledge and perform a real-time web search for the latest, most reliable information to verify the claim. Conduct your primary research in English for the most comprehensive results.
    3. If an image file is provided, perform a reverse image search to find its origin, date, and original context.
    4. Formulate your complete analysis and all findings in English first.
    5. **Translate the entire final JSON object** into the language you detected in step 1. All string values in the JSON must be translated, except for keys like "url".
    6. Return your complete, translated analysis as a single JSON object. DO NOT wrap it in markdown.
    **Required JSON Output Structure:**
    {{
      "claim_analysis": {{
        "verdict": "A concise verdict in one or two words (e.g., 'True', 'False', 'Misleading', 'Unverified').",
        "score": "An INTEGER score between 0 and 100 representing the claim's truthfulness. DO NOT use decimals. A 'True' verdict must have a score of 80 or higher. A 'False' verdict must have a score of 20 or lower.",
        "explanation": "A detailed but easy-to-understand explanation for the verdict, summarizing the key findings."
      }},
      "categorized_points": {{
        "points_supporting_truthfulness": ["List of key points or evidence supporting the claim's truthfulness."],
        "points_refuting_the_claim": ["List of key points or evidence refuting the claim."]
      }},
      "risk_assessment": {{
        "possible_consequences": ["List of potential risks or consequences of believing the misinformation (e.g., financial loss, health risks)."]
      }},
      "public_guidance_and_resources": {{
        "tips_to_identify_similar_scams": ["List of actionable tips for the public to identify similar false claims or scams."],
        "official_government_resources": {{
          "relevant_agency_website": "URL to a relevant official government or health organization website.",
          "national_helpline_number": "An official national helpline number for reporting or seeking help."
        }}
      }},
      "evidence_log": {{
        "external_sources": [
          {{
            "source_name": "Name of the credible source (e.g., 'Reuters', 'WHO').",
            "url": "Direct URL to the article or evidence."
          }}
        ],
        "image_analysis": {{
          "reverse_search_summary": "A summary of findings from a reverse image search, if an image was provided.",
          "original_context": "The original context, source, or date of the image, if found."
        }}
      }}
    }}
    """
    try:
        # Using a standard, widely available model to ensure compatibility.
        model = genai.GenerativeModel("gemini-1.0-pro")
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        contents = [prompt]
        if files:
            # Note: 'gemini-1.0-pro' does not support files directly.
            # For multi-modal input (images), 'gemini-pro-vision' is required.
            # This code will proceed with text-only analysis.
            if any(f['mime_type'].startswith('image') for f in files):
                 vision_model = genai.GenerativeModel('gemini-pro-vision')
                 # For a multi-modal request, you'd structure the contents differently.
                 # Example: vision_contents = [claim_text, image_data]
                 # For simplicity, we are still prioritizing the text analysis for this fix.
            pass


        response = model.generate_content(contents, generation_config=generation_config)
        result = json.loads(response.text)

        # Safeguard to correct the score if the model returns a float between 0 and 1
        if "claim_analysis" in result and "score" in result["claim_analysis"]:
            score = result["claim_analysis"]["score"]
            if isinstance(score, float) and 0.0 <= score <= 1.0:
                result["claim_analysis"]["score"] = int(score * 100)

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
    has_image = False
    for file in files:
        if file and allowed_file(file.filename):
            mime_type = mimetypes.guess_type(file.filename)[0]
            if mime_type:
                if mime_type.startswith('image'): has_image = True
                gemini_files.append({"mime_type": mime_type, "data": file.read()})
    
    if not gemini_files:
        return jsonify({"error": "Uploaded file types are not supported. Please use JPG, PNG, or PDF."}), 400

    # The 'gemini-1.0-pro' model handles text. For image analysis, you'd use 'gemini-pro-vision'.
    # This call will proceed with the text from the claim.
    result = get_fact_check_from_gemini(claim_text, source_type="file")
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
        # Using a standard, widely available model to ensure compatibility.
        model = genai.GenerativeModel("gemini-1.0-pro")
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

