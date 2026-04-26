import os
import json
import time
import requests
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import plotly.graph_objects as go

# ─── 1. CONFIGURATION ────────────────────────────────────────
load_dotenv()

# Securely fetch keys from Streamlit Secrets or Local .env
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY") or os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

# ─── 2. CORE LOGIC ───────────────────────────────────────────

def get_reviews_data(product_name):
    """Fetches text data using Serper API (Browserless)."""
    if not SERPER_API_KEY:
        st.error("Serper API Key missing!")
        return None

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
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        results = response.json()
        
        combined_text = ""
        organic = results.get('organic', [])
        
        if not organic:
            return None

        for item in organic:
            snippet = item.get('snippet', '')
            combined_text += f"{snippet} "
            
        return combined_text.strip()
    except Exception as e:
        st.error(f"Search API Error: {e}")
        return None

def analyze_sentiment(review_text, product_name):
    """Uses Gemini 1.5 Flash with Retry Logic for Quota Errors."""
    if not GEMINI_API_KEY:
        st.error("Gemini API Key missing!")
        return None

    genai.configure(api_key=GEMINI_API_KEY)
    
    try:
        # This is the current workhorse model
        model = genai.GenerativeModel('gemini-2.5-flash')
    except:
        # Fallback to the 'latest' alias which Google auto-updates
        model = genai.GenerativeModel('gemini-flash-latest')

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    prompt = f"""
    Analyze product sentiment for '{product_name}' using this text:
    {review_text}
    
    Return ONLY a valid JSON object:
    {{
        "score": 0-100,
        "vibe": "One sentence summary",
        "pros": ["pro1", "pro2"],
        "cons": ["con1", "con2"]
    }}
    """
    
    # Retry Loop for 429 Quota Errors
    for attempt in range(3):
        try:
            response = model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config={"response_mime_type": "application/json"}
            )
            if response and response.text:
                return json.loads(response.text)
        except Exception as e:
            if "429" in str(e):
                wait = (attempt + 1) * 15 # Wait 15s, then 30s
                st.warning(f"Quota busy. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            st.error(f"AI Error: {e}")
            break
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

# Custom Dark Mode Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap');
    html, body, [data-testid="stAppViewContainer"] { background: #0b0e1a !important; font-family: 'IBM Plex Sans', sans-serif; }
    .card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
    h1, h2, h3, p, span, label { color: white !important; }
    .app-header { font-size: 2.5rem; font-weight: 700; border-bottom: 2px solid #f0a500; margin-bottom: 2rem; }
    .app-header em { color: #f0a500; font-style: italic; }
</style>
<div class="app-header">Vibe<em>Check</em></div>
""", unsafe_allow_html=True)

product_query = st.text_input("What product are we checking?", placeholder="e.g. iPhone 17 Pro")
analyze_btn = st.button("Run Intelligence Check →")

if analyze_btn and product_query:
    with st.spinner("🕵️ Scouring the web..."):
        data = get_reviews_data(product_query)
    
    if data:
        st.write(f"📊 Signal Strength: {len(data)} characters found.")
        
        with st.spinner("🤖 Quantifying sentiment..."):
            analysis = analyze_sentiment(data, product_query)
        
        if analysis:
            score = analysis.get('score', 0)
            label, color = score_to_label(score)
            
            # Display Gauge
            st.plotly_chart(build_gauge(score), width='stretch')
            
            # Summary Card
            st.markdown(f"""
            <div class="card">
                <h2 style="color:{color} !important; margin-top:0;">{label}</h2>
                <p style="font-size: 1.1rem;">{analysis.get('vibe', '')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Pros/Cons
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ✅ Strengths")
                for p in analysis.get('pros', []): st.write(f"▲ {p}")
            with col2:
                st.markdown("### ❌ Weaknesses")
                for c in analysis.get('cons', []): st.write(f"▼ {c}")
        else:
            st.warning("⚠️ AI analysis failed. This is usually due to API limits. Try again in 60 seconds.")
    else:
        st.error("No search data found. Check your Serper API Key.")