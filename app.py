import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
from googletrans import Translator
from PIL import Image
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
import mimetypes

app = Flask(__name__)
# Enable CORS to allow requests from the React frontend (running on a different port)
CORS(app)

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- 1. API Key Configuration ---
try:
    # Load environment variables if you have a .env file
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

# --- 2. Data Loading and Retriever Setup (Done once on startup) ---
df = None
vectorizer = None
X = None
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

def retrieve_relevant_facts(query: str, top_k: int = 3) -> pd.DataFrame:
    """Retrieves the top_k most relevant facts from the local dataframe."""
    if df is None or vectorizer is None:
        return pd.DataFrame()
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]

# --- 3. Core Fact-Checking & Translation Logic ---
def get_fact_check_from_gemini(claim: str, files: list = None, output_lang='en') -> dict:
    if not api_key:
        return {"error": "API Key is not configured on the server."}
    
    facts = retrieve_relevant_facts(claim)
    facts_text = "No relevant facts found in the local dataset."
    if not facts.empty:
        facts_text = "\n".join([f"- {row['text']} (Label: {row['label']})" for _, row in facts.iterrows()])

    prompt = f"""
    You are a meticulous fact-checking analyst. Your task is to rigorously investigate the claim provided.

    **Claim to Investigate:** "{claim}"

    **Supporting Context from a Local Dataset (Use as a reference point only):**
    {facts_text}

    **Task:**
    1.  Use your internal knowledge and perform a real-time web search for the latest, most reliable information to verify the claim.
    2.  If files are provided, analyze their content (images, text) as primary evidence.
    3.  Return your complete analysis as a single JSON object. DO NOT wrap it in markdown or any other formatting.

    **Required JSON Output Structure:**
    {{
      "claim_analysis": {{
        "verdict": "A clear, one-word conclusion: 'True', 'False', 'Misleading', 'Partially True', or 'Unverifiable'.",
        "score": "A truthfulness confidence score (0-100).",
        "explanation": "A detailed, narrative-style explanation of your findings, citing sources within the text where applicable."
      }},
      "categorized_points": {{
         "points_supporting_truthfulness": ["A list of distinct facts or arguments that support the claim's validity."],
         "points_refuting_the_claim": ["A list of distinct facts or arguments that contradict the claim."]
      }},
      "risk_assessment": {{
        "possible_consequences": ["A list of potential real-world harms if this information is widely believed."]
      }},
      "public_guidance_and_resources": {{
        "tips_to_identify_similar_scams": ["Actionable advice for the public to spot similar false information."],
        "official_government_resources": {{
          "relevant_agency_website": "Link to an official government site (e.g., Ministry of Health, RBI).",
          "national_helpline_number": "An official, relevant helpline number (e.g., 'National Cyber Crime Reporting Portal: 1930')."
        }}
      }},
      "evidence_log": {{
        "external_sources": [{{ "source_name": "Name of the external source (e.g., World Health Organization).", "url": "A direct URL to the source material." }}]
      }}
    }}
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Prepare content for the API call (text + files)
        contents = [prompt]
        if files:
            contents.extend(files)

        response = model.generate_content(contents)
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        result = json.loads(cleaned_text)
        
        # Translate output if the original language was not English
        if output_lang != 'en' and output_lang != 'en-gb':
            translator = Translator()
            # Recursively translate all string values in the JSON object, skipping URLs
            def translate_recursive(data):
                if isinstance(data, str):
                    return translator.translate(data, dest=output_lang).text
                elif isinstance(data, list):
                    return [translate_recursive(item) for item in data]
                elif isinstance(data, dict):
                    return {k: v if k in ['url', 'relevant_agency_website'] else translate_recursive(v) for k, v in data.items()}
                return data
            
            return translate_recursive(result)
            
        return result
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON response from the model.", "raw_response": response.text}
    except Exception as e:
        return {"error": f"An API or other error occurred: {e}"}

# --- 4. Language Translation Utility ---
def translate_text_to_english(text):
    """Detects language and translates to English if necessary."""
    if not text or not text.strip():
        return text, 'en'
    translator = Translator()
    try:
        detected = translator.detect(text)
        input_lang = detected.lang
        if input_lang != 'en':
            translated_text = translator.translate(text, dest='en').text
            return translated_text, input_lang
        return text, 'en'
    except Exception:
        # If detection fails for any reason, assume English
        return text, 'en'

# --- 5. Flask API Routes ---
@app.route("/fact-check-text", methods=["POST"])
def fact_check_text():
    """Receives a text claim, processes it, and returns the result."""
    data = request.get_json()
    if not data or "claim" not in data:
        return jsonify({"error": "Invalid request. 'claim' key is missing."}), 400
    
    claim, input_lang = translate_text_to_english(data["claim"])
    result = get_fact_check_from_gemini(claim, output_lang=input_lang)
    return jsonify(result)

@app.route("/fact-check-url", methods=["POST"])
def fact_check_url():
    """Receives a URL, scrapes its content, and returns the fact-check result."""
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
        
        full_claim = f"{claim_context}\n\n**Content from URL ({url}):**\n{content[:4000]}" # Limit context size
        
        claim_en, input_lang = translate_text_to_english(full_claim)
        result = get_fact_check_from_gemini(claim_en, output_lang=input_lang)
        return jsonify(result)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch URL: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred while processing the URL: {e}"}), 500

@app.route("/fact-check-file", methods=["POST"])
def fact_check_file():
    """Receives files (images/PDFs), sends them to the model, and returns the result."""
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

    claim_en, input_lang = translate_text_to_english(claim_text)
    result = get_fact_check_from_gemini(claim_en, files=gemini_files, output_lang=input_lang)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)