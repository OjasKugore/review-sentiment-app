import os
import json
import re
import time
import requests
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import plotly.graph_objects as go

# ─── 1. CONFIGURATION ────────────────────────────────────────
load_dotenv()

# Fetch keys from Streamlit Secrets or local .env
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY") or os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

# ─── 2. CORE LOGIC ───────────────────────────────────────────

def get_reviews_data(product_name):
    """Fetches text data using Serper API."""
    if not SERPER_API_KEY:
        st.error("Serper API Key missing!")
        return None

    url = "https://google.serper.dev/search"
    # Specific query to get high-quality review data
    payload = {
        "q": f"{product_name} detailed user reviews pros and cons amazon reddit",
        "num": 10,
        "gl": "us",
        "autocorrect": True
    }
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
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
    """Uses Gemini 1.5 Flash with Retry Logic for Quota Errors."""
    if not GEMINI_API_KEY:
        st.error("Gemini API Key missing!")
        return None

    genai.configure(api_key=GEMINI_API_KEY)
    # Using 1.5-flash as it has a more stable free-tier quota on Streamlit
    model = genai.GenerativeModel('gemini-1.5-flash')

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    prompt = f"""
    Analyze product sentiment for '{product_name}' using these search snippets:
    {review_text}
    
    You MUST return ONLY a valid JSON object:
    {{
        "score": 0-100,
        "vibe": "One sentence summary of the general feeling",
        "pros": ["pro1", "pro2", "pro3"],
        "cons": ["con1", "con2", "con3"]
    }}
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 12 # 12s, 24s...
                    st.warning(f"Quota busy. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    st.error("Quota fully exhausted for now. Please wait a few minutes.")
            else:
                st.error(f"AI Error: {err_str}")
            return None

# ─── 3. UI HELPERS ────────────────────────────────────────────

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
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": accent},
            "bgcolor": "rgba(255,255,255,0.05)"
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=250, margin=dict(t=0, b=0))
    return fig

# ─── 4. STREAMLIT UI ──────────────────────────────────────────

st.set_page_config(page_title="Vibe Check", page_icon="🛡️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [data-testid="stAppViewContainer"] { background: #0b0e1a !important; font-family: 'IBM Plex Sans', sans-serif; }
    .card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
    h1, h2, h3, p, span, label { color: white !important; }
    .app-header { font-size: 2.5rem; font-weight: 700; border-bottom: 2px solid #f0a500; margin-bottom: 2rem; }
    .app-header em { color: #f0a500; font-style: italic; }
</style>
<div class="app-header">Vibe<em>Check</em></div>
""", unsafe_allow_html=True)

product_query = st.text_input("Product Name", placeholder="e.g. Sony WH-1000XM5")
analyze_btn = st.button("Run Intelligence Check →")

if analyze_btn and product_query:
    status_container = st.empty()
    
    status_container.info("🕵️ Scouring the web...")
    data = get_reviews_data(product_query)
    
    if data:
        st.write(f"📊 Signal Strength: {len(data)} characters found.")
        
        status_container.info("🤖 Quantifying sentiment...")
        analysis = analyze_sentiment(data, product_query)
        
        if analysis:
            status_container.empty()
            score = analysis.get('score', 50)
            label, color = score_to_label(score)
            
            st.plotly_chart(build_gauge(score), width='stretch')
            
            st.markdown(f"""
            <div class="card">
                <h2 style="color:{color} !important; margin-top:0;">{label}</h2>
                <p style="font-size: 1.1rem;">{analysis.get('vibe', '')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ✅ Strengths")
                for p in analysis.get('pros', []): st.write(f"▲ {p}")
            with col2:
                st.markdown("### ❌ Weaknesses")
                for c in analysis.get('cons', []): st.write(f"▼ {c}")
    else:
        status_container.error("No search data found. Check your Serper API Key.")