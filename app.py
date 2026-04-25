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
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
try:
    model = genai.GenerativeModel('gemini-2.0-flash')
except:
    model = genai.GenerativeModel('gemini-1.5-flash')

# ─── 2. CORE LOGIC (NO BROWSER NEEDED) ───────────────────────
def get_reviews_data(product_name):
    """Fetches review data using Serper's built-in scraping."""
    url = "https://google.serper.dev/search"
    payload = {
        "q": f"{product_name} user reviews amazon reddit",
        "num": 4,
        "autocorrect": True
    }
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        results = response.json()
        
        # We combine snippets and 'Sitelinks' to feed the AI
        combined_text = ""
        for item in results.get('organic', []):
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            combined_text += f"\nSource: {title}\nContent: {snippet}\n"
            
        return combined_text
    except Exception as e:
        st.error(f"Search failed: {e}")
        return ""

def analyze_sentiment(review_text, product_name):
    prompt = f"""
    You are an expert product analyst. Based on the following review data for '{product_name}', 
    provide a detailed sentiment analysis.
    
    Data: {review_text}
    
    Return ONLY a valid JSON object:
    {{
        "score": 72,
        "vibe": "One sentence summary of the general feeling",
        "pros": ["pro1", "pro2", "pro3"],
        "cons": ["con1", "con2", "con3"]
    }}
    """
    response = model.generate_content(prompt)
    raw = response.text
    raw = raw.replace("```json", "").replace("```", "").strip()
    
    # Clean up JSON formatting
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group(0)
    raw = re.sub(r',\s*([\]}])', r'\1', raw)

    try:
        return json.loads(raw)
    except:
        # Emergency fix if JSON is malformed
        fix_response = model.generate_content(f"Fix this JSON: {raw}")
        return json.loads(fix_response.text.replace("```json", "").replace("```", "").strip())

# ─── 3. CHART HELPERS ────────────────────────────────────────
def score_to_label(score):
    if score >= 80: return "Highly Recommended", "#22c98a"
    if score >= 65: return "Worth Buying",        "#5bc8a8"
    if score >= 45: return "Mixed Signals",        "#f0a500"
    if score >= 25: return "Proceed with Caution", "#e07b39"
    return "Not Recommended", "#e85d5d"

def build_gauge(score):
    _, accent = score_to_label(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 48, "color": accent}, "suffix": ""},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": accent, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 50], "color": "rgba(255,255,255,0.05)"},
                {"range": [50, 100], "color": "rgba(255,255,255,0.05)"}
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(t=30, b=0))
    return fig

def build_radar(pros, cons):
    cats = pros + cons
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[90, 80, 70, 0, 0, 0], theta=cats, fill='toself', name='Pros', line_color='#22c98a'))
    fig.add_trace(go.Scatterpolar(r=[0, 0, 0, 70, 80, 60], theta=cats, fill='toself', name='Cons', line_color='#e85d5d'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 100])), paper_bgcolor='rgba(0,0,0,0)', showlegend=True)
    return fig

# ─── 4. CSS STYLING ──────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [data-testid="stAppViewContainer"] { background: #0b0e1a !important; font-family: 'IBM Plex Sans', sans-serif; }
.card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; }
.app-header { border-bottom: 1px solid #f0a500; padding-bottom: 1rem; margin-bottom: 2rem; }
.app-wordmark { font-size: 2.5rem; font-weight: 700; color: #fff; }
.app-wordmark em { color: #f0a500; font-style: italic; }
h1, h2, h3, p, span { color: white !important; }
</style>
"""

# ─── 5. APP INTERFACE ────────────────────────────────────────
st.set_page_config(page_title="Vibe Check", page_icon="🛡️")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown('<div class="app-header"><div class="app-wordmark">Vibe<em>Check</em></div></div>', unsafe_allow_html=True)

product_query = st.text_input("Enter product name:")
analyze_btn = st.button("Analyze Vibe →")

if analyze_btn and product_query:
    with st.spinner("🔍 Scanning web signals..."):
        text_data = get_reviews_data(product_query)
    
    if text_data:
        with st.spinner("🤖 AI Sentiment Analysis..."):
            analysis = analyze_sentiment(text_data, product_query)
            
            score = analysis['score']
            vibe = analysis['vibe']
            label, color = score_to_label(score)
            
            col1, col2 = st.columns([1, 1.5])
            with col1:
                st.markdown(f'<div class="card"><h3 style="color:{color} !important;">{score}/100</h3><p>{label}</p></div>', unsafe_allow_html=True)
                st.plotly_chart(build_gauge(score), use_container_width=True)
            with col2:
                st.markdown(f'<div class="card"><i>"{vibe}"</i></div>', unsafe_allow_html=True)
            
            st.plotly_chart(build_radar(analysis['pros'], analysis['cons']), use_container_width=True)
    else:
        st.error("Could not find enough data for this product.")