import os
import json
import re
import requests
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import plotly.graph_objects as go

# ─── 1. CONFIGURATION ────────────────────────────────────────
load_dotenv()

# Securely get API keys (works for local .env or Streamlit Cloud Secrets)
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY") or os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
    except:
        model = genai.GenerativeModel('gemini-1.5-pro')

# ─── 2. CORE LOGIC ───────────────────────────────────────────
def get_reviews_data(product_name):
    """Fetches text data using Serper API without needing a browser."""
    url = "https://google.serper.dev/search"
    payload = {
        "q": f"{product_name} user reviews pros cons reddit amazon",
        "num": 10,
        "gl": "us",
        "autocorrect": True
    }
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        results = response.json()
        
        combined_text = ""
        organic = results.get('organic', [])
        
        if not organic:
            return None

        for item in organic:
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            combined_text += f"\nSource: {title}\nReview Snippet: {snippet}\n"
            
        return combined_text
    except Exception as e:
        st.error(f"Search API Error: {e}")
        return None

def analyze_sentiment(review_text, product_name):
    # This tells Gemini to ignore 'harm' filters so it can read spicy reviews
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    prompt = f"""
    You are a product analyst. Analyze these search snippets for '{product_name}':
    {review_text}
    
    Return ONLY a valid JSON:
    {{"score": 85, "vibe": "Great product overall", "pros": ["item1"], "cons": ["item1"]}}
    """
    
    try:
        # We add the safety_settings here
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config={"response_mime_type": "application/json"}
        )
        
        return json.loads(response.text)
    except Exception as e:
        return {
            "score": 0, 
            "vibe": f"Analysis Error: {str(e)[:50]}", 
            "pros": ["N/A"], "cons": ["N/A"]
        }
# ─── 3. UI HELPERS ───────────────────────────────────────────
def score_to_label(score):
    if score >= 80: return "Highly Recommended", "#22c98a"
    if score >= 65: return "Worth Buying",        "#5bc8a8"
    if score >= 45: return "Mixed Signals",        "#f0a500"
    return "Not Recommended", "#e85d5d"

def build_gauge(score):
    _, accent = score_to_label(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"color": accent, "size": 50}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": accent}}
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=250, margin=dict(t=0, b=0))
    return fig

# ─── 4. CSS & LAYOUT ─────────────────────────────────────────
st.set_page_config(page_title="Vibe Check", page_icon="🛡️", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [data-testid="stAppViewContainer"] { background: #0b0e1a !important; font-family: 'IBM Plex Sans', sans-serif; }
    .card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
    .stTextInput input { background: #1a1f35 !important; color: white !important; border: 1px solid #30364d !important; }
    h1, h2, h3, p, span, label { color: white !important; }
    .app-header { font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; border-bottom: 2px solid #f0a500; }
    .app-header em { color: #f0a500; font-style: italic; }
</style>
<div class="app-header">Vibe<em>Check</em></div>
""", unsafe_allow_html=True)

product_query = st.text_input("Product Name", placeholder="e.g. Sony WH-1000XM5")
analyze_btn = st.button("Run Intelligence Check →")

if analyze_btn and product_query:
    with st.spinner("🕵️ Scouring the web..."):
        data = get_reviews_data(product_query)

    if data:
        st.write(f"DEBUG: Found {len(data)} characters of data.")
    
    if data:
        with st.spinner("🤖 Quantifying sentiment..."):
            analysis = analyze_sentiment(data, product_query)
            score = analysis.get('score', 50)
            label, color = score_to_label(score)
            
            st.plotly_chart(build_gauge(score), width='stretch')
            
            st.markdown(f"""
            <div class="card">
                <h2 style="color:{color} !important; margin-top:0;">{label}</h2>
                <p style="font-size: 1.1rem; line-height: 1.5;">{analysis.get('vibe', '')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### ✅ Strengths")
                for p in analysis.get('pros', []): st.write(f"▲ {p}")
            with c2:
                st.markdown("### ❌ Weaknesses")
                for c in analysis.get('cons', []): st.write(f"▼ {c}")
    else:
        st.error("No data found. Check your API key or try a more common product name.")