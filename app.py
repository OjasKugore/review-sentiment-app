import os
import subprocess
import streamlit as st

# --- CLOUD SETUP FOR PLAYWRIGHT ---
@st.cache_resource
def install_playwright_browsers():
    try:
        # Check if chromium is already installed to save time
        subprocess.run(["python", "-m", "playwright", "install", "chromium"], check=True)
        subprocess.run(["python", "-m", "playwright", "install-deps"], check=True)
    except Exception as e:
        st.error(f"Error installing browser: {e}")

# Only run this once per session
install_playwright_browsers()
import requests

import asyncio
import json
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler
import google.generativeai as genai
import plotly.graph_objects as go

load_dotenv()

# ─── API KEYS ────────────────────────────────────────────────
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
try:
    model = genai.GenerativeModel('gemini-2.5-flash')
except:
    model = genai.GenerativeModel('gemini-1.5-flash')

# ─── CORE LOGIC (unchanged) ──────────────────────────────────
def search_product_reviews(product_name):
    url = "https://google.serper.dev/search"
    payload = {"q": f"{product_name} user reviews amazon reddit", "num": 2}
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, json=payload)
    return [item['link'] for item in response.json().get('organic', [])]

async def get_clean_text(urls):
    combined_text = ""
    async with AsyncWebCrawler(verbose=False) as crawler:
        for url in urls:
            result = await crawler.arun(url=url)
            combined_text += result.markdown[:4000]
    return combined_text

def analyze_sentiment(review_text, product_name):
    import re
    prompt = f"""
    You are an expert product analyst. Based on the following scraped review data for '{product_name}', 
    provide a detailed sentiment analysis.
    
    Data: {review_text}
    
    Return ONLY a valid JSON object — no trailing commas, no comments, no extra text outside the JSON:
    {{
        "score": 72,
        "vibe": "One sentence summary of the general feeling",
        "pros": ["pro1", "pro2", "pro3"],
        "cons": ["con1", "con2", "con3"]
    }}
    """
    response = model.generate_content(prompt)
    raw = response.text

    # Strip markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Replace curly/smart quotes with straight quotes
    raw = (raw.replace("\u201c", '"').replace("\u201d", '"')
              .replace("\u2018", "'").replace("\u2019", "'"))

    # Extract first JSON object if there is surrounding text
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group(0)

    # Remove trailing commas before ] or } (common LLM mistake)
    raw = re.sub(r',\s*([\]}])', r'\1', raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Last resort: ask the model to fix its own output
        fix_response = model.generate_content(
            f"Fix this malformed JSON and return ONLY the corrected JSON, nothing else:\n{raw}"
        )
        fixed = fix_response.text.replace("```json", "").replace("```", "").strip()
        fixed = re.sub(r',\s*([\]}])', r'\1', fixed)
        return json.loads(fixed)

# ─── CHART HELPERS ───────────────────────────────────────────
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
        number={"font": {"size": 48, "color": accent, "family": "Fraunces, Georgia, serif"}, "suffix": ""},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickvals": [0, 25, 50, 75, 100],
                "ticktext": ["0", "25", "50", "75", "100"],
                "tickcolor": "rgba(255,255,255,0.15)",
                "tickfont": {"color": "rgba(255,255,255,0.28)", "size": 10, "family": "IBM Plex Sans"},
            },
            "bar": {"color": accent, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  25],  "color": "rgba(232,93,93,0.09)"},
                {"range": [25, 50],  "color": "rgba(224,123,57,0.09)"},
                {"range": [50, 75],  "color": "rgba(240,165,0,0.09)"},
                {"range": [75, 100], "color": "rgba(34,201,138,0.09)"},
            ],
            "threshold": {"line": {"color": accent, "width": 2}, "thickness": 0.8, "value": score},
        },
        title={"text": "CONFIDENCE METER",
               "font": {"color": "rgba(255,255,255,0.28)", "size": 10, "family": "IBM Plex Sans"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=36, b=0, l=28, r=28), height=210,
    )
    return fig


def build_radar(pros, cons):
    def clip(s, n=28): return s[:n] + "…" if len(s) > n else s
    cats     = [clip(p) for p in pros] + [clip(c) for c in cons]
    pro_vals = [88, 76, 68, 0, 0, 0]
    con_vals = [0, 0, 0, 66, 72, 55]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=pro_vals + [pro_vals[0]], theta=cats + [cats[0]],
        fill='toself', name='Strengths',
        line=dict(color='#22c98a', width=2.5),
        fillcolor='rgba(34,201,138,0.12)',
        marker=dict(size=6, color='#22c98a'),
    ))
    fig.add_trace(go.Scatterpolar(
        r=con_vals + [con_vals[0]], theta=cats + [cats[0]],
        fill='toself', name='Weaknesses',
        line=dict(color='#e85d5d', width=2.5),
        fillcolor='rgba(232,93,93,0.12)',
        marker=dict(size=6, color='#e85d5d'),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(255,255,255,0.02)',
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor='rgba(255,255,255,0.07)',
                tickvals=[25, 50, 75],
                tickfont=dict(color='rgba(255,255,255,0.2)', size=9, family='IBM Plex Sans'),
                linecolor='rgba(255,255,255,0.05)',
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.07)',
                tickfont=dict(color='rgba(255,255,255,0.62)', size=12, family='IBM Plex Sans'),
                linecolor='rgba(255,255,255,0.07)',
            ),
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            font=dict(color='rgba(255,255,255,0.52)', size=12, family='IBM Plex Sans'),
            bgcolor='rgba(255,255,255,0.04)',
            bordercolor='rgba(255,255,255,0.08)', borderwidth=1,
            x=0.5, xanchor='center', y=-0.1, orientation='h',
        ),
        margin=dict(t=24, b=70, l=70, r=70),
        height=450,
    )
    return fig


def build_bar(pros, cons):
    def clip(s, n=34): return s[:n] + "…" if len(s) > n else s
    labels = [clip(p) for p in pros] + [clip(c) for c in cons]
    values = [88, 76, 68, -66, -72, -55]
    colors = ['rgba(34,201,138,0.78)' if v > 0 else 'rgba(232,93,93,0.78)' for v in values]
    border = ['rgba(34,201,138,0.4)'  if v > 0 else 'rgba(232,93,93,0.4)'  for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation='h',
        marker=dict(color=colors, line=dict(color=border, width=1)),
        text=[f" +{v} " if v > 0 else f" {v} " for v in values],
        textposition='outside',
        textfont=dict(color='rgba(255,255,255,0.42)', size=11, family='IBM Plex Sans'),
        hovertemplate='%{y}<extra></extra>',
        width=0.52,
    ))
    fig.add_vline(x=0, line_color='rgba(255,255,255,0.18)', line_width=1.5)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-135, 135]),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.04)',
            tickfont=dict(color='rgba(255,255,255,0.68)', size=12, family='IBM Plex Sans'),
            automargin=True,
        ),
        margin=dict(t=16, b=24, l=16, r=80),
        height=360,
        bargap=0.44,
    )
    return fig


# ─── CSS ─────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;1,9..144,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] { background: #0b0e1a !important; }

/* Dot-grid texture */
[data-testid="stAppViewContainer"]::after {
    content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image: radial-gradient(rgba(255,255,255,0.04) 1px, transparent 1px);
    background-size: 28px 28px;
}
/* Ambient blobs */
[data-testid="stAppViewContainer"]::before {
    content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 65% 50% at 4%  12%,  rgba(240,165,0,0.13)  0%, transparent 58%),
        radial-gradient(ellipse 50% 42% at 93% 82%,  rgba(34,201,138,0.11) 0%, transparent 55%),
        radial-gradient(ellipse 38% 32% at 50% 48%,  rgba(72,52,140,0.09)  0%, transparent 55%);
}

[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }

section.main > div { padding-top: 2.5rem; max-width: 880px; margin: auto; }

*, p, div, span, label { font-family: 'IBM Plex Sans', sans-serif !important; color: rgba(255,255,255,0.82); }

/* ── Header ── */
.app-header {
    display: flex; align-items: baseline; gap: 1.1rem;
    border-bottom: 1px solid rgba(240,165,0,0.22);
    padding-bottom: 1.1rem; margin-bottom: 0.2rem;
}
.app-wordmark {
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: 2.6rem; font-weight: 700;
    letter-spacing: -0.025em; line-height: 1;
    color: #fff !important;
}
.app-wordmark em { font-style: italic; color: #f0a500 !important; }
.app-tagline {
    font-size: 0.7rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: rgba(255,255,255,0.28) !important;
    padding-bottom: 0.1rem;
}

/* ── Section markers ── */
.section-row {
    display: flex; align-items: center; gap: 0.85rem;
    margin: 2rem 0 0.9rem;
}
.section-num {
    font-family: 'Fraunces', serif !important; font-size: 0.7rem;
    font-weight: 600; letter-spacing: 0.1em; color: #f0a500 !important;
    border: 1px solid rgba(240,165,0,0.32); border-radius: 4px;
    padding: 0.15rem 0.42rem; white-space: nowrap;
}
.section-title {
    font-family: 'Fraunces', serif !important; font-size: 0.76rem;
    font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase;
    color: rgba(255,255,255,0.35) !important;
}
.section-rule { flex: 1; height: 1px; background: rgba(255,255,255,0.07); }

/* ── Glass card ── */
.card {
    background: rgba(255,255,255,0.038);
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08); border-radius: 16px;
    padding: 1.6rem 1.8rem; margin-bottom: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.055);
    position: relative;
}
.card-amber { border-left: 3px solid rgba(240,165,0,0.55); }
.card-green  { border-left: 3px solid rgba(34,201,138,0.55); }
.card-red    { border-left: 3px solid rgba(232,93,93,0.55); }

/* ── Score ── */
.score-super {
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: 5.8rem; font-weight: 700; line-height: 1; letter-spacing: -0.04em;
}
.score-denom {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 1rem; color: rgba(255,255,255,0.22) !important;
    vertical-align: super; margin-left: 3px;
}
.verdict-badge {
    display: inline-block;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; border-radius: 20px;
    padding: 0.26rem 0.78rem; margin-top: 0.55rem; border: 1px solid;
}

/* ── Verdict quote ── */
.verdict-quote {
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: 1.12rem; font-weight: 300; font-style: italic;
    line-height: 1.65; color: rgba(255,255,255,0.76) !important;
    border-left: 2px solid rgba(240,165,0,0.45);
    padding-left: 1rem; margin: 0.9rem 0 0.7rem;
}

/* ── Product pill ── */
.product-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(240,165,0,0.1); border: 1px solid rgba(240,165,0,0.25);
    border-radius: 20px; padding: 0.24rem 0.82rem;
    font-size: 0.76rem; color: rgba(255,215,120,0.85) !important;
}

/* ── Pro/Con items ── */
.pcon-item {
    display: flex; align-items: flex-start; gap: 0.7rem;
    padding: 0.72rem 0.88rem; border-radius: 10px;
    margin-bottom: 0.52rem; font-size: 0.875rem; line-height: 1.45;
}
.pro-item { background: rgba(34,201,138,0.07); border: 1px solid rgba(34,201,138,0.15); }
.con-item { background: rgba(232,93,93,0.07);  border: 1px solid rgba(232,93,93,0.15); }
.pro-item .dot { color: #22c98a !important; font-size: 0.9rem; margin-top: 0.1rem; }
.con-item .dot { color: #e85d5d !important; font-size: 0.9rem; margin-top: 0.1rem; }

/* ── Chart caption ── */
.chart-caption {
    font-size: 0.74rem; color: rgba(255,255,255,0.28) !important;
    font-style: italic; letter-spacing: 0.03em; margin-bottom: 0.2rem;
}

/* ── Input ── */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important; color: white !important;
    padding: 0.82rem 1rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stTextInput"] input::placeholder { color: rgba(255,255,255,0.2) !important; }
[data-testid="stTextInput"] input:focus {
    border-color: rgba(240,165,0,0.48) !important;
    box-shadow: 0 0 0 3px rgba(240,165,0,0.1) !important;
    outline: none !important;
}
[data-testid="stTextInput"] label {
    font-size: 0.7rem !important; letter-spacing: 0.13em;
    text-transform: uppercase; color: rgba(255,255,255,0.36) !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #f0a500 0%, #e07b39 100%) !important;
    color: #0b0e1a !important; border: none !important;
    border-radius: 12px !important; padding: 0.78rem 2rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 0.88rem !important;
    letter-spacing: 0.04em !important; transition: all 0.2s ease !important;
    box-shadow: 0 4px 18px rgba(240,165,0,0.28) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 26px rgba(240,165,0,0.42) !important;
}

[data-testid="stSpinner"] p { color: rgba(255,255,255,0.38) !important; }
.js-plotly-plot .plotly, .js-plotly-plot .plotly .plot-container { background: transparent !important; }

/* ── Style plotly chart blocks as cards (avoids broken open/close div hack) ── */
[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.038) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 1rem 0.5rem 0.2rem !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.055);
}

/* Pull caption flush against the chart card below it */
p.chart-caption {
    margin-bottom: -0.5rem !important;
    padding: 0.7rem 1.2rem 0 !important;
    background: rgba(255,255,255,0.038);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    border-bottom: none;
    border-radius: 16px 16px 0 0;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.055);
    display: block;
}
/* Chart card gets square top corners when caption is above it */
p.chart-caption + div[data-testid="stPlotlyChart"],
p.chart-caption ~ div > [data-testid="stPlotlyChart"] {
    border-radius: 0 0 16px 16px !important;
    border-top: none !important;
}
</style>
"""

# ─── PAGE SETUP ──────────────────────────────────────────────
st.set_page_config(page_title="Vibe Check", page_icon="🛡️", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-wordmark">Vibe<em>Check</em></div>
    <div class="app-tagline">AI-powered product sentiment intelligence</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="height:0.15rem"></div>', unsafe_allow_html=True)

product_query = st.text_input(
    "Product name",
    placeholder="e.g. Sony WH-1000XM5, Dyson V15, MacBook Air M3…",
    label_visibility="visible"
)
analyze = st.button("Analyze Vibe →")

# ─── ANALYSIS ────────────────────────────────────────────────
if analyze:
    if product_query:
        with st.spinner("Hunting and reading reviews across the web…"):
            links     = search_product_reviews(product_query)
            text_data = asyncio.run(get_clean_text(links))

        if text_data:
            with st.spinner("Calculating sentiment…"):
                analysis = analyze_sentiment(text_data, product_query)

            score  = analysis['score']
            vibe   = analysis['vibe']
            pros   = analysis['pros']
            cons   = analysis['cons']
            label, color = score_to_label(score)

            # ══ 01  OVERVIEW ════════════════════════════════
            st.markdown("""
            <div class="section-row">
                <span class="section-num">01</span>
                <span class="section-title">Overview</span>
                <div class="section-rule"></div>
            </div>""", unsafe_allow_html=True)

            col_l, col_r = st.columns([1, 1.65], gap="large")

            with col_l:
                st.markdown(f"""
                <div class="card card-amber">
                    <div style="font-size:0.66rem; letter-spacing:0.15em; text-transform:uppercase;
                                color:rgba(255,255,255,0.26); margin-bottom:0.35rem;">Score</div>
                    <div>
                        <span class="score-super" style="color:{color};">{score}</span>
                        <span class="score-denom">/100</span>
                    </div>
                    <div class="verdict-badge"
                         style="color:{color}; border-color:{color}40; background:{color}18;">
                        {label}
                    </div>
                </div>""", unsafe_allow_html=True)
                st.plotly_chart(build_gauge(score), use_container_width=True,
                                config={"displayModeBar": False})

            with col_r:
                st.markdown(f"""
                <div class="card" style="height:100%;">
                    <div style="font-size:0.66rem; letter-spacing:0.15em; text-transform:uppercase;
                                color:rgba(255,255,255,0.26); margin-bottom:0.5rem;">Product</div>
                    <div class="product-pill">📦 {product_query}</div>
                    <div class="verdict-quote">"{vibe}"</div>
                    <div style="font-size:0.7rem; color:rgba(255,255,255,0.22);
                                margin-top:0.8rem; letter-spacing:0.04em; font-style:italic;">
                        Synthesised from live web reviews &amp; community discussion threads
                    </div>
                </div>""", unsafe_allow_html=True)

            # ══ 02  STRENGTHS & WEAKNESSES ══════════════════
            st.markdown("""
            <div class="section-row">
                <span class="section-num">02</span>
                <span class="section-title">Strengths &amp; Weaknesses</span>
                <div class="section-rule"></div>
            </div>""", unsafe_allow_html=True)

            pc, cc = st.columns(2, gap="medium")
            with pc:
                pros_html = "".join([
                    f'<div class="pcon-item pro-item"><span class="dot">▲</span><span>{p}</span></div>'
                    for p in pros
                ])
                st.markdown(f"""
                <div class="card card-green">
                    <div style="font-size:0.66rem; letter-spacing:0.14em; text-transform:uppercase;
                                color:#22c98a; margin-bottom:0.85rem; font-weight:600;">
                        What people love
                    </div>{pros_html}
                </div>""", unsafe_allow_html=True)

            with cc:
                cons_html = "".join([
                    f'<div class="pcon-item con-item"><span class="dot">▼</span><span>{c}</span></div>'
                    for c in cons
                ])
                st.markdown(f"""
                <div class="card card-red">
                    <div style="font-size:0.66rem; letter-spacing:0.14em; text-transform:uppercase;
                                color:#e85d5d; margin-bottom:0.85rem; font-weight:600;">
                        Common complaints
                    </div>{cons_html}
                </div>""", unsafe_allow_html=True)

            # ══ 03  ATTRIBUTE RADAR ═════════════════════════
            st.markdown("""
            <div class="section-row">
                <span class="section-num">03</span>
                <span class="section-title">Attribute Radar</span>
                <div class="section-rule"></div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<p class="chart-caption" style="margin-bottom:0.3rem;">Relative attribute mapping across reported strengths and weaknesses</p>', unsafe_allow_html=True)
            st.plotly_chart(build_radar(pros, cons), use_container_width=True,
                            config={"displayModeBar": False})

            # ══ 04  SIGNAL STRENGTH ═════════════════════════
            st.markdown("""
            <div class="section-row">
                <span class="section-num">04</span>
                <span class="section-title">Signal Strength</span>
                <div class="section-rule"></div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<p class="chart-caption" style="margin-bottom:0.3rem;">Diverging bar view — green bars are positive signals, red are negative</p>', unsafe_allow_html=True)
            st.plotly_chart(build_bar(pros, cons), use_container_width=True,
                            config={"displayModeBar": False})

            # ══ FOOTER ══════════════════════════════════════
            st.markdown("""
            <div style="text-align:center; padding:1.6rem 0 1rem;
                        font-size:0.68rem; color:rgba(255,255,255,0.16);
                        letter-spacing:0.08em; font-style:italic;">
                Analysis generated from live review data · Powered by Gemini AI
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="card card-red" style="text-align:center; padding:2.5rem;">
                <div style="font-size:2rem; margin-bottom:0.6rem;">⚠</div>
                <div style="font-weight:600; font-size:1rem;">Not enough data found.</div>
                <div style="color:rgba(255,255,255,0.32); font-size:0.85rem; margin-top:0.4rem;">
                    Try a more specific product name or model number.
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem;
                    color:rgba(255,255,255,0.22); font-size:0.88rem;
                    letter-spacing:0.05em; font-style:italic;">
            Enter a product name above to begin your analysis.
        </div>""", unsafe_allow_html=True)