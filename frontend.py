import streamlit as st
import requests
import json
import time
import os
import tempfile
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Neural Gateway",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# PREMIUM CSS — Full production-grade dark UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #c9d1d9;
    -webkit-font-smoothing: antialiased;
}

/* ── Animated background ── */
@keyframes gradBG {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stApp {
    background: radial-gradient(ellipse at 20% 20%, #0d1117 0%, #010409 100%);
    background-size: 400% 400%;
}

/* Particle canvas overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(circle at 15% 80%, rgba(88,166,255,0.06) 0%, transparent 50%),
        radial-gradient(circle at 85% 20%, rgba(149,128,255,0.06) 0%, transparent 50%),
        radial-gradient(circle at 50% 50%, rgba(56,189,248,0.03) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

header, footer { display: none !important; }
#MainMenu { visibility: hidden; }

/* ── Typography ── */
h1 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.6rem !important;
    letter-spacing: -1.5px !important;
    background: linear-gradient(135deg, #58a6ff 0%, #8b5cf6 50%, #38bdf8 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: 0 !important;
    line-height: 1.1 !important;
}

h2, h3 {
    font-family: 'Outfit', sans-serif !important;
    color: #e6edf3 !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(13, 17, 23, 0.97) !important;
    border-right: 1px solid rgba(88,166,255,0.12) !important;
}

[data-testid="stSidebar"] > div {
    padding: 24px 18px !important;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.sidebar-logo-icon {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #58a6ff, #8b5cf6);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    box-shadow: 0 0 20px rgba(88,166,255,0.35);
}

.sidebar-logo-text {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: #e6edf3;
    letter-spacing: -0.3px;
}

.sidebar-logo-version {
    font-size: 0.65rem;
    color: #58a6ff;
    background: rgba(88,166,255,0.1);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 6px;
    padding: 1px 6px;
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
}

.section-header {
    font-family: 'Outfit', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #8b949e;
    margin: 20px 0 12px 0;
}

/* ── Inputs ── */
.stTextArea textarea {
    background: rgba(22, 27, 34, 0.95) !important;
    border: 1px solid rgba(48, 54, 61, 0.8) !important;
    color: #e6edf3 !important;
    border-radius: 14px !important;
    font-size: 1rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 18px !important;
    line-height: 1.7 !important;
    transition: all 0.25s ease !important;
    resize: vertical !important;
}

.stTextArea textarea:focus {
    border-color: rgba(88,166,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.12), 0 0 40px rgba(88,166,255,0.08) !important;
    background: rgba(22, 27, 34, 1) !important;
}

.stTextArea textarea::placeholder {
    color: #484f58 !important;
}

.stTextInput input {
    background: rgba(22, 27, 34, 0.95) !important;
    border: 1px solid rgba(48, 54, 61, 0.8) !important;
    color: #e6edf3 !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s ease !important;
}

.stTextInput input:focus {
    border-color: rgba(88,166,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.12) !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] {
    padding: 6px 0 !important;
}

.stSlider [data-testid="stThumbValue"] {
    background: rgba(88,166,255,0.15) !important;
    color: #58a6ff !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Toggle ── */
.stToggle label {
    color: #c9d1d9 !important;
    font-size: 0.88rem !important;
}

/* ── Multiselect ── */
[data-baseweb="select"] > div {
    background: rgba(22, 27, 34, 0.95) !important;
    border: 1px solid rgba(48, 54, 61, 0.8) !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
}

/* ── Primary CTA Button ── */
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}

@keyframes glow-pulse {
    0%, 100% { box-shadow: 0 0 20px rgba(88,166,255,0.3), 0 4px 20px rgba(88,166,255,0.2); }
    50%       { box-shadow: 0 0 35px rgba(88,166,255,0.5), 0 4px 30px rgba(139,92,246,0.3); }
}

.stButton > button[kind="primary"],
.stButton > button:first-child {
    background: linear-gradient(135deg, #1d4ed8 0%, #4338ca 40%, #6d28d9 100%) !important;
    background-size: 200% auto !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: 0.3px !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    animation: glow-pulse 3s ease-in-out infinite !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button:first-child:hover {
    background: linear-gradient(135deg, #2563eb 0%, #4f46e5 40%, #7c3aed 100%) !important;
    transform: translateY(-2px) !important;
    animation: none !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.5) !important;
}

.stButton > button:first-child:active {
    transform: translateY(0px) !important;
}

/* ── Cards ── */
.ng-card {
    background: linear-gradient(145deg, rgba(22,27,34,0.9) 0%, rgba(13,17,23,0.95) 100%);
    border: 1px solid rgba(48,54,61,0.7);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    backdrop-filter: blur(20px);
}

.ng-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(88,166,255,0.3), transparent);
}

.ng-card:hover {
    border-color: rgba(88,166,255,0.25);
    box-shadow: 0 8px 40px rgba(88,166,255,0.08);
}

.ng-card-accent {
    border-color: rgba(88,166,255,0.3);
    box-shadow: 0 0 0 1px rgba(88,166,255,0.08) inset, 0 8px 40px rgba(88,166,255,0.06);
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(145deg, rgba(22,27,34,0.95) 0%, rgba(13,17,23,0.98) 100%);
    border: 1px solid rgba(48,54,61,0.8);
    border-radius: 16px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #58a6ff, #8b5cf6);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.3s ease;
}

.metric-card:hover::after {
    transform: scaleX(1);
}

.metric-card:hover {
    border-color: rgba(88,166,255,0.3);
    box-shadow: 0 4px 24px rgba(88,166,255,0.1);
    transform: translateY(-2px);
}

.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    color: #8b949e;
    margin-bottom: 10px;
    font-family: 'Inter', sans-serif;
}

.metric-value {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.65rem;
    color: #e6edf3;
    letter-spacing: -0.5px;
    line-height: 1.1;
    white-space: normal;
    word-break: break-word;
    overflow: visible;
}

.metric-value.highlight {
    background: linear-gradient(135deg, #58a6ff 0%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-icon {
    font-size: 1.5rem;
    margin-bottom: 10px;
    display: block;
}

/* ── Status Banner ── */
.status-banner {
    border-radius: 14px;
    padding: 16px 22px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 20px;
    font-weight: 500;
    font-size: 0.95rem;
    border: 1px solid transparent;
}

.status-success {
    background: linear-gradient(135deg, rgba(22,101,52,0.25), rgba(21,128,61,0.15));
    border-color: rgba(34,197,94,0.3);
    color: #4ade80;
}

.status-error {
    background: linear-gradient(135deg, rgba(127,29,29,0.25), rgba(153,27,27,0.15));
    border-color: rgba(239,68,68,0.3);
    color: #f87171;
}

.status-warning {
    background: linear-gradient(135deg, rgba(120,53,15,0.25), rgba(146,64,14,0.15));
    border-color: rgba(251,146,60,0.3);
    color: #fb923c;
}

/* ── Model Tag Pill ── */
.model-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    background: rgba(88,166,255,0.1);
    border: 1px solid rgba(88,166,255,0.25);
    border-radius: 100px;
    color: #58a6ff;
    font-size: 0.82rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
    margin: 4px;
    transition: all 0.2s ease;
}

.model-pill:hover {
    background: rgba(88,166,255,0.18);
    border-color: rgba(88,166,255,0.4);
}

.model-pill-fallback {
    background: rgba(48,54,61,0.5);
    border-color: rgba(48,54,61,0.8);
    color: #8b949e;
}

/* ── Progress Bars ── */
.progress-row {
    margin-bottom: 14px;
}

.progress-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.progress-label {
    font-size: 0.83rem;
    color: #c9d1d9;
    font-weight: 500;
}

.progress-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #58a6ff;
    font-weight: 600;
}

.progress-track {
    height: 6px;
    background: rgba(48,54,61,0.8);
    border-radius: 100px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #1d4ed8, #58a6ff);
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}

/* ── Tag chips ── */
.tag-chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.3px;
    margin: 3px;
}

.tag-blue   { background: rgba(88,166,255,0.12); border: 1px solid rgba(88,166,255,0.3); color: #58a6ff; }
.tag-purple { background: rgba(139,92,246,0.12); border: 1px solid rgba(139,92,246,0.3); color: #a78bfa; }
.tag-green  { background: rgba(34,197,94,0.1);   border: 1px solid rgba(34,197,94,0.25); color: #4ade80; }
.tag-orange { background: rgba(251,146,60,0.1);  border: 1px solid rgba(251,146,60,0.25); color: #fb923c; }
.tag-red    { background: rgba(239,68,68,0.1);   border: 1px solid rgba(239,68,68,0.25); color: #f87171; }

/* ── Divider ── */
.ng-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(48,54,61,0.8), transparent);
    margin: 24px 0;
}

/* ── Streamlit native overrides ── */
[data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #e6edf3 !important;
    white-space: normal !important;
    word-break: break-word !important;
    overflow: visible !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.8px !important;
    color: #8b949e !important;
    font-weight: 600 !important;
}

[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

[data-testid="stExpander"] {
    background: rgba(22,27,34,0.6) !important;
    border: 1px solid rgba(48,54,61,0.6) !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #c9d1d9 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #58a6ff !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(22,27,34,0.6) !important;
    border: 2px dashed rgba(48,54,61,0.8) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s ease !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(88,166,255,0.4) !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(22,27,34,0.95) !important;
    border: 1px solid rgba(48,54,61,0.8) !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
}

/* ── Toast ── */
[data-testid="stToast"] {
    background: rgba(22,27,34,0.98) !important;
    border: 1px solid rgba(34,197,94,0.3) !important;
    border-radius: 12px !important;
}

/* ── Code blocks ── */
.stCode {
    background: rgba(13,17,23,0.9) !important;
    border: 1px solid rgba(48,54,61,0.6) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(88,166,255,0.2); border-radius: 100px; }
::-webkit-scrollbar-thumb:hover { background: rgba(88,166,255,0.35); }

/* ── Sidebar fix for select truncation ── */
div[data-baseweb="select"] > div {
    overflow-x: auto !important;
    overflow-y: hidden !important;
    flex-wrap: nowrap !important;
}

div[data-baseweb="select"] > div::-webkit-scrollbar { display: none; }

.stMultiSelect div[data-baseweb="select"] span[data-baseweb="tag"] {
    white-space: nowrap !important;
}

/* ── LLM streaming output ── */
.llm-output {
    background: rgba(13,17,23,0.8);
    border: 1px solid rgba(48,54,61,0.7);
    border-radius: 14px;
    padding: 22px 26px;
    font-size: 0.98rem;
    line-height: 1.75;
    color: #e6edf3;
    white-space: pre-wrap;
    font-family: 'Inter', sans-serif;
}

/* ── Hero area ── */
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: rgba(88,166,255,0.08);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 14px;
    font-family: 'Inter', sans-serif;
}

/* ── Feedback panel ── */
.feedback-note {
    background: rgba(88,166,255,0.06);
    border: 1px solid rgba(88,166,255,0.15);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 0.875rem;
    color: #8b949e;
    line-height: 1.6;
}

/* ── Animated dots ── */
@keyframes dotPulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.3; }
}

.live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #4ade80;
    animation: dotPulse 1.5s ease-in-out infinite;
    margin-right: 6px;
    vertical-align: middle;
}

/* Sidebar powered-by footer */
.sidebar-footer {
    position: fixed;
    bottom: 20px;
    padding: 12px 16px;
    background: rgba(22,27,34,0.8);
    border: 1px solid rgba(48,54,61,0.6);
    border-radius: 12px;
    font-size: 0.75rem;
    color: #484f58;
    text-align: center;
    line-height: 1.5;
}

.sidebar-footer span {
    color: #58a6ff;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
API_URL = os.getenv("NEURAL_GATEWAY_API_URL", "http://127.0.0.1:8000").rstrip("/")

RISK_COLOR = {"low": "tag-green", "medium": "tag-orange", "high": "tag-red", "extreme": "tag-red"}
COMPLEXITY_COLOR = {"low": "tag-green", "medium": "tag-orange", "high": "tag-purple"}

FAMILIES = ["chat", "coding", "reasoning", "mathematics", "vision", "ocr",
            "document_qa", "audio", "agent", "translation", "summarization",
            "image_generation", "video_generation"]
DOMAINS = ["general", "software", "medical", "legal", "finance",
           "security", "crm", "hrm", "project", "accounts"]
RISKS = ["low", "medium", "high", "extreme"]
COMPLEXITIES = ["low", "medium", "high"]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_available_models():
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        if resp.status_code == 200:
            models = resp.json()
            models.sort(key=lambda m: (m.get("provider", ""), m.get("name", "")))
            names = [m["name"] for m in models]
            provider_map = {m["name"]: m.get("provider", "Unknown") for m in models}
            return names, provider_map
    except Exception:
        pass
    return [], {}


def get_friendly_reason(raw_data):
    if not raw_data:
        return "Unknown error"
    status = raw_data.get("decision_record", {}).get("status", "")
    if status == "plan_exceeds_hard_constraints":
        constraints = raw_data.get("decision_record", {}).get("plan_constraints", {})
        est_cost = constraints.get("expected_cost_usd", 0)
        max_cost_v = constraints.get("max_cost_usd")
        est_lat = constraints.get("expected_latency_ms", 0)
        max_lat_v = constraints.get("max_latency_ms")
        reason = "The best available model exceeds your configured limits.\n\n"
        if max_cost_v is not None and est_cost > max_cost_v:
            reason += f"- **Cost:** Estimated **${est_cost:.4f}** exceeds your cap of **${max_cost_v:.2f}**.\n"
        if max_lat_v is not None and est_lat > max_lat_v:
            reason += f"- **Latency:** Estimated **{est_lat/1000:.1f}s** exceeds your cap of **{max_lat_v/1000:.1f}s**.\n"
        return reason + "\n*Increase the sliders in the sidebar to allow more capable models.*"
    elif status == "no_feasible_models":
        msg = "No models in the registry can satisfy this request. The document may be too large, or no model supports the required features (e.g., Web Search, OCR).\n\n"
        dr = raw_data.get("decision_record", {})
        feas = dr.get("feasibility_reasons", {})
        if feas:
            msg += "**Why models were excluded:**\n"
            for model, reason in list(feas.items())[:5]:
                msg += f"- `{model}`: {reason}\n"
        return msg
    elif status == "no_models_after_policy":
        msg = "Models are available, but all were blocked by your active Tenant Policy constraints.\n\n"
        dr = raw_data.get("decision_record", {})
        notes = dr.get("policy_notes", [])
        if notes:
            msg += "**Policy Notes:**\n" + "\n".join([f"- {n}" for n in notes]) + "\n\n"
        
        feas = dr.get("feasibility_reasons", {})
        policy_reasons = dr.get("policy_reasons", {})
        
        combined_reasons = {**feas, **policy_reasons}
        if combined_reasons:
            msg += "**Why models were excluded:**\n"
            for model, reason in list(combined_reasons.items())[:5]:
                msg += f"- `{model}`: {reason}\n"
            if len(combined_reasons) > 5:
                msg += f"- ...and {len(combined_reasons)-5} more.\n"
        return msg
    elif status == "abstained":
        return "The routing engine could not reach a confident decision. The winning candidate's confidence score was below the minimum threshold."
    return f"Status code: `{status}`"


def render_progress_bar(label: str, value: float):
    pct = min(max(float(value), 0.0), 1.0)
    color = "#58a6ff" if pct > 0.65 else ("#fb923c" if pct > 0.4 else "#f87171")
    st.markdown(f"""
    <div class="progress-row">
        <div class="progress-label-row">
            <span class="progress-label">{label}</span>
            <span class="progress-value">{pct*100:.1f}%</span>
        </div>
        <div class="progress-track">
            <div class="progress-fill" style="width:{pct*100:.1f}%; background: linear-gradient(90deg, {color}cc, {color});"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def tag(text, style="tag-blue"):
    return f'<span class="tag-chip {style}">{text}</span>'


def model_pill(name, fallback=False):
    cls = "model-pill model-pill-fallback" if fallback else "model-pill"
    short = name.split("/")[-1]
    return f'<span class="{cls}">{"↓" if fallback else "⚡"} {short}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "decision": None, "abstained": False, "status": "",
    "task_summary": None, "trace": None, "prompt": "",
    "raw_data": None, "elapsed_ms": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
available_models, provider_map = fetch_available_models()

with st.sidebar:
    # Logo
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">⚡</div>
        <div>
            <div class="sidebar-logo-text">Neural Gateway</div>
            <div class="sidebar-logo-version">v1.0 · Production</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status indicator
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        is_live = health.status_code == 200
    except Exception:
        is_live = False

    if is_live:
        st.markdown('<p style="font-size:0.83rem; color:#4ade80;"><span class="live-dot"></span>API Online</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:0.83rem; color:#f87171;">● API Offline — Start the FastAPI server</p>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Request Capabilities</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        req_json   = st.toggle("JSON Mode",   value=False)
        req_search = st.toggle("Web Search",   value=False)
    with col2:
        req_ocr    = st.toggle("OCR",          value=False)
        req_cite   = st.toggle("Citations",    value=False)

    st.markdown('<div class="section-header">Performance Limits</div>', unsafe_allow_html=True)
    max_latency = st.slider("Max Latency (ms)", 1000, 120000, 30000, 1000, format="%d ms")
    max_cost    = st.slider("Max Cost ($/1M tokens)", 0.1, 50.0, 10.0, 0.5, format="$%.1f")

    st.markdown('<div class="section-header">Tenant & Model Filter</div>', unsafe_allow_html=True)
    tenant_id = st.text_input("Tenant ID", value="tenant-ui-demo")
    allowed_models = st.multiselect(
        "Allowed Models (empty = all)",
        options=available_models,
        format_func=lambda x: f"[{provider_map.get(x, '?')}] {x}"
    )

    st.markdown('<div class="section-header">LLM Execution</div>', unsafe_allow_html=True)
    openrouter_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="sk-or-…",
        help="Paste your OpenRouter key to run Stage 2 LLM generation with the selected model."
    )

    st.markdown("""
    <div style="margin-top: 28px; padding: 14px; background: rgba(22,27,34,0.6); border: 1px solid rgba(48,54,61,0.5); border-radius: 12px; font-size: 0.75rem; color: #484f58; text-align: center; line-height: 1.7;">
        Powered by <span style="color:#58a6ff">Bayesian Inference</span><br>
        &amp; <span style="color:#8b5cf6">Thompson Sampling</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-badge">⚡ Enterprise AI Routing</div>
""", unsafe_allow_html=True)
st.title("Neural Gateway")
st.markdown("""
<p style="color:#8b949e; font-size:1.05rem; margin-top:-8px; margin-bottom:28px; line-height:1.6;">
    Intelligent, multi-objective model selection powered by live benchmarks,
    Bayesian scoring, and Thompson Sampling.
</p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT INPUT AREA
# ─────────────────────────────────────────────────────────────────────────────
with st.container():
    st.markdown("""
    <p style="font-size:0.82rem; font-weight:600; text-transform:uppercase; letter-spacing:1.8px;
    color:#8b949e; margin:0 0 8px 0; font-family:'Inter',sans-serif;">📝 Your Prompt</p>
    """, unsafe_allow_html=True)
    prompt = st.text_area(
        label="Prompt",
        label_visibility="collapsed",
        height=160,
        placeholder="Describe your task — e.g. 'Implement a concurrent web crawler in Python with rate limiting, exponential backoff, and structured JSON output…'"
    )
    col_upload, col_btn = st.columns([3, 1])
    with col_upload:
        uploaded_files = st.file_uploader(
            "Attach files",
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
    with col_btn:
        st.markdown('<div style="margin-top: 8px;"></div>', unsafe_allow_html=True)
        route_clicked = st.button("⚡ Route Request", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTING LOGIC
# ─────────────────────────────────────────────────────────────────────────────
if route_clicked:
    if not prompt.strip():
        st.markdown('<div class="status-banner status-warning">⚠️ Please enter a prompt before routing.</div>', unsafe_allow_html=True)
    else:
        tc = {"tenant_id": tenant_id}
        if allowed_models:
            tc["allowed_models"] = allowed_models

        payload = {
            "prompt": prompt,
            "request_constraints": {
                "require_json":       req_json,
                "require_ocr":        req_ocr,
                "require_web_search": req_search,
                "require_citations":  req_cite,
                "max_latency_ms":     float(max_latency),
                "max_cost_usd":       float(max_cost),
            },
            "tenant_context": tc,
        }

        saved_file_paths = []
        if uploaded_files:
            temp_dir = tempfile.mkdtemp()
            for uf in uploaded_files:
                fp = os.path.join(temp_dir, uf.name)
                with open(fp, "wb") as f:
                    f.write(uf.getbuffer())
                saved_file_paths.append(fp)
            payload["files"] = saved_file_paths

        with st.spinner("🧠 Analyzing task & scoring models across the registry…"):
            start_time = time.time()
            try:
                resp = requests.post(f"{API_URL}/route", json=payload, timeout=60.0)
                elapsed = (time.time() - start_time) * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.decision     = data.get("selected_plan", {})
                    st.session_state.abstained    = data.get("abstain", False)
                    st.session_state.status       = data.get("decision_record", {}).get("status", "")
                    st.session_state.task_summary = data.get("decision_record", {}).get("task_summary", {}) or {}
                    st.session_state.trace        = data.get("decision_record", {}).get("pipeline_trace", {}) or {}
                    st.session_state.raw_data     = data
                    st.session_state.prompt       = prompt
                    st.session_state.elapsed_ms   = elapsed
                else:
                    st.session_state.decision  = None
                    st.session_state.abstained = True
                    st.session_state.raw_data  = {"decision_record": {"status": f"http_{resp.status_code}"}}
                    st.markdown(f'<div class="status-banner status-error">❌ Backend error {resp.status_code}: {resp.text[:200]}</div>', unsafe_allow_html=True)

            except requests.exceptions.ConnectionError:
                st.markdown(f'<div class="status-banner status-error">❌ Cannot connect to API at <code>{API_URL}</code>. Is the FastAPI server running?</div>', unsafe_allow_html=True)
                st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS — ABSTAINED
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.abstained:
    friendly_msg = get_friendly_reason(st.session_state.raw_data)
    st.markdown(f"""
    <div class="status-banner status-error">
        🛑 <strong>Router Abstained</strong> — The request could not be routed.
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="ng-card"><p style="color:#f87171; font-weight:600; margin-bottom:10px;">Why was this rejected?</p><p style="color:#c9d1d9; line-height:1.7;">{friendly_msg}</p></div>', unsafe_allow_html=True)
    with st.expander("🔍 Raw JSON Response"):
        st.json(st.session_state.raw_data)

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS — SUCCESS
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.decision:
    decision     = st.session_state.decision
    task_summary = st.session_state.task_summary or {}
    trace        = st.session_state.trace or {}
    elapsed_ms   = st.session_state.elapsed_ms

    # Success banner
    selected_short = decision.get("selected_model", "N/A").split("/")[-1]
    st.markdown(f"""
    <div class="status-banner status-success">
        ✅ <strong>Routing Successful</strong> — Decision made in <strong>{elapsed_ms:.0f}ms</strong>
        &nbsp;·&nbsp; Routing to <strong>{selected_short}</strong>
    </div>
    """, unsafe_allow_html=True)

    # ── Top Metrics ──
    mc1, mc2, mc3, mc4 = st.columns(4)

    with mc1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-label">Selected Model</div>
            <div class="metric-value highlight" style="font-size:1.1rem;">{decision.get("selected_model","N/A").split("/")[-1]}</div>
        </div>
        """, unsafe_allow_html=True)

    with mc2:
        confidence_val = decision.get("confidence", 0)
        conf_pct = f"{confidence_val*100:.1f}%"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📊</div>
            <div class="metric-label">Confidence</div>
            <div class="metric-value highlight">{conf_pct}</div>
        </div>
        """, unsafe_allow_html=True)

    with mc3:
        lat = decision.get("expected_latency_ms", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-label">Est. Latency</div>
            <div class="metric-value">{lat:.0f} <span style="font-size:1rem;color:#8b949e;">ms</span></div>
        </div>
        """, unsafe_allow_html=True)

    with mc4:
        cost = decision.get("expected_cost_usd", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">💰</div>
            <div class="metric-label">Est. Cost</div>
            <div class="metric-value">${cost:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column layout: Task Analysis + Pipeline Trace ──
    left, right = st.columns([1.1, 1])

    with left:
        family     = task_summary.get("primary_family", "N/A")
        domain     = task_summary.get("domain", "N/A")
        complexity = task_summary.get("complexity", "N/A")
        risk       = task_summary.get("risk_tier", "N/A")
        complexity_style = COMPLEXITY_COLOR.get(complexity, "tag-blue")
        risk_style = RISK_COLOR.get(risk, "tag-blue")
        fallbacks  = decision.get("fallback_models", [])
        fallback_html = ""
        if fallbacks:
            pills = "".join(model_pill(f, fallback=True) for f in fallbacks)
            fallback_html = f'<p style="color:#8b949e;font-size:0.78rem;text-transform:uppercase;letter-spacing:1.5px;margin-top:16px;margin-bottom:8px;font-weight:600;">Fallback Chain</p><div>{pills}</div>'
        complexity_alert = ""
        if complexity == "high":
            complexity_alert = '<div style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.25);border-radius:10px;padding:12px 16px;font-size:0.85rem;color:#a78bfa;margin-bottom:14px;">🧩 <strong>High-Complexity Task Detected</strong> — Router prioritized frontier reasoning models.</div>'
        st.markdown(f"""
        <div class="ng-card">
            <p style="font-weight:700;font-size:1rem;color:#e6edf3;margin:0 0 10px 0;">🧠 Semantic Task Analysis</p>
            <div class="ng-divider"></div>
            <div style="margin-bottom:18px;">
                <p style="color:#8b949e;font-size:0.78rem;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;font-weight:600;">Task Classification</p>
                {tag(family, "tag-blue")} {tag(domain, "tag-purple")}
            </div>
            <div style="margin-bottom:18px;">
                <p style="color:#8b949e;font-size:0.78rem;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;font-weight:600;">Risk & Complexity</p>
                {tag(f"Risk: {risk}", risk_style)} {tag(f"Complexity: {complexity}", complexity_style)}
            </div>
            {complexity_alert}
            {fallback_html}
        </div>
        """, unsafe_allow_html=True)

    with right:
        registry = trace.get("registry_models", 0)
        feasible = trace.get("feasible_after_filter", 0)
        policy   = trace.get("after_policy", 0)
        pareto   = trace.get("after_pareto", 0)
        st.markdown(f"""
        <div class="ng-card">
            <p style="font-weight:700;font-size:1rem;color:#e6edf3;margin:0 0 10px 0;">🔬 Pipeline Trace</p>
            <div class="ng-divider"></div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                <div style="background:rgba(22,27,34,0.8);border:1px solid rgba(48,54,61,0.6);border-radius:12px;padding:14px;">
                    <div style="font-size:0.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;font-weight:600;">Registry</div>
                    <div style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:700;color:#e6edf3;">{registry}</div>
                </div>
                <div style="background:rgba(22,27,34,0.8);border:1px solid rgba(48,54,61,0.6);border-radius:12px;padding:14px;">
                    <div style="font-size:0.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;font-weight:600;">Feasible</div>
                    <div style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:700;color:#58a6ff;">{feasible}</div>
                </div>
                <div style="background:rgba(22,27,34,0.8);border:1px solid rgba(48,54,61,0.6);border-radius:12px;padding:14px;">
                    <div style="font-size:0.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;font-weight:600;">After Policy</div>
                    <div style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:700;color:#a78bfa;">{policy}</div>
                </div>
                <div style="background:rgba(22,27,34,0.8);border:1px solid rgba(48,54,61,0.6);border-radius:12px;padding:14px;">
                    <div style="font-size:0.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;font-weight:600;">Pareto Frontier</div>
                    <div style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:700;color:#4ade80;">{pareto}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Quality Breakdown ──
    q = decision.get("explanation", {}).get("quality_breakdown", {})
    if q:
        def _pbar(label, val):
            pct = min(max(float(val), 0.0), 1.0)
            color = "#58a6ff" if pct > 0.65 else ("#fb923c" if pct > 0.4 else "#f87171")
            return f"""
            <div class="progress-row">
                <div class="progress-label-row">
                    <span class="progress-label">{label}</span>
                    <span class="progress-value">{pct*100:.1f}%</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width:{pct*100:.1f}%;background:linear-gradient(90deg,{color}cc,{color});"></div>
                </div>
            </div>"""
        items = [
            ("Contextual Fit",        q.get("contextual_mean", 0)),
            ("Workflow Fit",          q.get("workflow_fit", 0)),
            ("Domain Fit",            q.get("domain_fit", 0)),
            ("Runtime Adjusted Mean", q.get("runtime_adjusted_mean", 0)),
        ]
        left_bars  = _pbar(items[0][0], items[0][1]) + _pbar(items[2][0], items[2][1])
        right_bars = _pbar(items[1][0], items[1][1]) + _pbar(items[3][0], items[3][1])
        st.markdown(f"""
        <div class="ng-card">
            <p style="font-weight:700;font-size:1rem;color:#e6edf3;margin:0 0 10px 0;">📈 Quality Score Breakdown</p>
            <div class="ng-divider"></div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
                <div>{left_bars}</div>
                <div>{right_bars}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Raw JSON ──
    with st.expander("🔍 View Raw JSON Response"):
        st.json(st.session_state.raw_data)

    # ── Train the Engine ──
    st.markdown("""
    <div class="ng-card">
        <p style="font-weight:700;font-size:1rem;color:#e6edf3;margin:0 0 10px 0;">🎓 Correct the Semantic Parser</p>
        <div class="ng-divider"></div>
        <div class="feedback-note">If the router misclassified your task, submit the correct labels here. The KNN memory matrix will be instantly rebuilt with your correction.</div>
    </div>
    """, unsafe_allow_html=True)
    with st.container():
        fb1, fb2, fb3, fb4, fb5 = st.columns([2, 2, 2, 2, 1.5])
        with fb1:
            correct_family = st.selectbox("Primary Family", options=FAMILIES, key="feedback_family")
        with fb2:
            correct_domain = st.selectbox("Domain", options=DOMAINS, key="feedback_domain")
        with fb3:
            correct_risk = st.selectbox("Risk Tier", options=RISKS, key="feedback_risk")
        with fb4:
            correct_complexity = st.selectbox("Complexity", options=COMPLEXITIES, key="feedback_complexity")
        with fb5:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Submit Feedback", use_container_width=True):
                try:
                    r = requests.post(f"{API_URL}/train_parser", json={
                        "prompt":         st.session_state.prompt,
                        "primary_family": correct_family,
                        "domain":         correct_domain,
                        "risk_tier":      correct_risk,
                        "complexity":     correct_complexity,
                    })
                    if r.status_code == 200:
                        st.toast("✅ Training example submitted — engine updated!", icon="🎉")
                        st.balloons()
                    else:
                        st.error("Failed to submit feedback.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Stage 2: LLM Execution ──
    selected_model_id = decision.get("selected_model", "")
    model_pill_html = f'<span class="model-pill" style="display:inline-flex;margin-left:4px;">⚡ {selected_model_id.split("/")[-1]}</span>' if selected_model_id else ""
    st.markdown(f"""
    <div class="ng-card">
        <p style="font-weight:700;font-size:1rem;color:#e6edf3;margin:0 0 10px 0;">⚡ Stage 2 — Generate Response</p>
        <div class="ng-divider"></div>
        <div class="feedback-note">Execute the routed request directly via OpenRouter using the selected model. Enter your API key in the sidebar to enable this.</div>
        <p style="color:#8b949e;font-size:0.9rem;margin:14px 0 0 0;">Will execute using: {model_pill_html}</p>
    </div>
    """, unsafe_allow_html=True)

    gen_clicked = st.button("🚀 Generate with Selected Model", use_container_width=True)

    if gen_clicked:
        if not openrouter_key:
            st.markdown('<div class="status-banner status-warning">⚠️ Enter your OpenRouter API Key in the sidebar first.</div>', unsafe_allow_html=True)
        else:
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
            response_placeholder = st.empty()
            full_response = ""

            with st.spinner(f"Generating with {selected_model_id}…"):
                try:
                    completion = client.chat.completions.create(
                        model=selected_model_id,
                        messages=[{"role": "user", "content": st.session_state.prompt}],
                        stream=True,
                    )
                    for chunk in completion:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            full_response += delta
                            response_placeholder.markdown(
                                f'<div class="llm-output">{full_response}▌</div>',
                                unsafe_allow_html=True
                            )

                    if re.match(r"^https?://[^\s]+$", full_response.strip()):
                        full_response = f"![Generated Image]({full_response.strip()})"
                        response_placeholder.markdown(full_response)
                    else:
                        response_placeholder.markdown(
                            f'<div class="llm-output">{full_response}</div>',
                            unsafe_allow_html=True
                        )
                    st.markdown('<div class="status-banner status-success">✅ Generation complete.</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(f'<div class="status-banner status-error">❌ Execution failed: {str(e)}</div>', unsafe_allow_html=True)
