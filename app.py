import os
import subprocess
import streamlit as st
import asyncio
import json
import re
import requests
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler
import google.generativeai as genai
import plotly.graph_objects as go

# ─── 1. SETUP LOGIC (Must come first) ────────────────────────
@st.cache_resource
def power_on_browser():
    # 1. Check if we've already done this to save time
    if not os.path.exists("browser_installed.txt"):
        try:
            # 2. Force install the playwright library first
            subprocess.run(["pip", "install", "playwright"], check=True)
            
            # 3. Install the actual Chromium browser
            subprocess.run(["python", "-m", "playwright", "install", "chromium"], check=True)
            
            with open("browser_installed.txt", "w") as f:
                f.write("done")
        except subprocess.CalledProcessError as e:
            st.error(f"Installation failed: {e}")
            # If it fails, we try a fallback command
            os.system("python -m playwright install chromium")

power_on_browser()

# ─── 2. CONFIGURATION ────────────────────────────────────────
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
try:
    model = genai.GenerativeModel('gemini-2.0-flash')
except:
    model = genai.GenerativeModel('gemini-1.5-flash')

# ─── 3. CORE FUNCTIONS (Defining before use) ─────────────────
def search_product_reviews(product_name):
    url = "https://google.serper.dev/search"
    payload = {"q": f"{product_name} user reviews amazon reddit", "num": 2}
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, json=payload)
    return [item['link'] for item in response.json().get('organic', [])]

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

async def get_clean_text(urls):
    combined_text = ""
    
    # 1. Configure the browser for low-memory cloud environments
    browser_cfg = BrowserConfig(
        headless=True,
        browser_type="chromium",
        extra_args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",  # Crucial: uses RAM instead of shared memory
            "--disable-gpu",            # No GPU on Streamlit servers
            "--single-process",         # Uses significantly less RAM
        ]
    )
    
    # 2. Configure the run settings
    run_cfg = CrawlerRunConfig(cache_mode="bypass")

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for url in urls:
            result = await crawler.arun(url=url, config=run_cfg)
            if result.success:
                combined_text += result.markdown[:4000]
                
    return combined_text

def analyze_sentiment(review_text, product_name):
    prompt = f"""
    You are an expert product analyst. Based on the following scraped review data for '{product_name}', 
    provide a detailed sentiment analysis.
    
    Data: {review_text}
    
    Return ONLY a valid JSON object:
    {{
        "score": 72,
        "vibe": "One sentence summary",
        "pros": ["pro1", "pro2", "pro3"],
        "cons": ["con1", "con2", "con3"]
    }}
    """
    response = model.generate_content(prompt)
    raw = response.text
    raw = raw.replace("```json", "").replace("```", "").strip()
    raw = (raw.replace("\u201c", '"').replace("\u201d", '"')
              .replace("\u2018", "'").replace("\u2019", "'"))
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match: raw = match.group(0)
    raw = re.sub(r',\s*([\]}])', r'\1', raw)

    try:
        return json.loads(raw)
    except:
        fix_response = model.generate_content(f"Fix this JSON: {raw}")
        fixed = fix_response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(fixed)

# ─── 4. CHART HELPERS (Defining your visual logic) ──────────
def score_to_label(score):
    if score >= 80: return "Highly Recommended", "#22c98a"
    if score >= 65: return "Worth Buying",        "#5bc8a8"
    if score >= 45: return "Mixed Signals",        "#f0a500"
    if score >= 25: return "Proceed with Caution", "#e07b39"
    return "Not Recommended", "#e85d5d"

def build_gauge(score):
    _, accent = score_to_label(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': accent}}
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=210)
    return fig

def build_radar(pros, cons):
    def clip(s, n=28): return s[:n] + "…" if len(s) > n else s
    cats = [clip(p) for p in pros] + [clip(c) for c in cons]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[88, 76, 68, 0, 0, 0, 88], theta=cats + [cats[0]], fill='toself', name='Strengths'))
    fig.add_trace(go.Scatterpolar(r=[0, 0, 0, 66, 72, 55, 0], theta=cats + [cats[0]], fill='toself', name='Weaknesses'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def build_bar(pros, cons):
    def clip(s, n=34): return s[:n] + "…" if len(s) > n else s
    labels = [clip(p) for p in pros] + [clip(c) for c in cons]
    values = [88, 76, 68, -66, -72, -55]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation='h'))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# ─── 5. UI & CSS (Your Layout) ───────────────────────────────
CSS = """
<style>
/* ... (All your CSS exactly as you had it) ... */
</style>
"""

st.set_page_config(page_title="Vibe Check", page_icon="🛡️", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""<div class="app-header"><div class="app-wordmark">Vibe<em>Check</em></div></div>""", unsafe_allow_html=True)

product_query = st.text_input("Product name")
analyze = st.button("Analyze Vibe →")

if analyze and product_query:
    with st.spinner("Analyzing..."):
        links = search_product_reviews(product_query)
        text_data = asyncio.run(get_clean_text(links))
        
        if text_data:
            analysis = analyze_sentiment(text_data, product_query)
            score, vibe, pros, cons = analysis['score'], analysis['vibe'], analysis['pros'], analysis['cons']
            label, color = score_to_label(score)

            # RENDER UI
            st.plotly_chart(build_gauge(score))
            st.write(f"Verdict: {vibe}")
            st.plotly_chart(build_radar(pros, cons))
            st.plotly_chart(build_bar(pros, cons))