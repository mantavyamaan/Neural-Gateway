import streamlit as st
import requests
import json
import time
import os
import tempfile
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Page Config must be the first Streamlit command
st.set_page_config(
    page_title="Neural Gateway | Intelligent Gateway",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Custom CSS ---
st.markdown("""
<style>
/* Reset and base styles */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    color: #e0e0e0;
}

/* Animated Dark Premium Background */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stApp {
    background: linear-gradient(-45deg, #050508, #0a0a14, #050508, #0c0c16);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

/* Hide default streamlit elements */
header, footer {
    display: none !important;
}

/* Title Styling */
h1 {
    font-family: 'Outfit', sans-serif !important;
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    margin-bottom: 0.5rem;
}

h2, h3 {
    font-family: 'Outfit', sans-serif !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* High-End Glassmorphism Cards */
.glass-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 24px;
    box-shadow: 0 10px 40px -10px rgba(0,0,0,0.5), inset 0 0 0 1px rgba(255,255,255,0.02);
    transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease, border-color 0.4s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 242, 254, 0.3);
    box-shadow: 0 15px 50px -15px rgba(0,242,254,0.15), inset 0 0 0 1px rgba(0,242,254,0.1);
}

/* Premium Metric Display */
[data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif !important;
    background: linear-gradient(90deg, #ffffff 0%, #a0a0a0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    letter-spacing: -1px;
}

[data-testid="stMetricLabel"] {
    color: #8a8d93 !important;
    font-size: 0.95rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}

/* Premium Text Area with Glow */
.stTextArea textarea {
    background: rgba(10, 10, 15, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #fff !important;
    border-radius: 16px !important;
    font-size: 1.05rem !important;
    padding: 20px !important;
    line-height: 1.6 !important;
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}

.stTextArea textarea:focus {
    border-color: #00f2fe !important;
    box-shadow: 0 0 0 1px #00f2fe, 0 0 30px rgba(0, 242, 254, 0.2) !important;
    background: rgba(15, 15, 25, 0.8) !important;
}

/* Premium Button with Pulse Animation */
@keyframes pulseGlow {
    0% { box-shadow: 0 0 0 0 rgba(0, 242, 254, 0.4); }
    70% { box-shadow: 0 0 0 15px rgba(0, 242, 254, 0); }
    100% { box-shadow: 0 0 0 0 rgba(0, 242, 254, 0); }
}

.stButton>button {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 32px !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: 0.5px;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    width: 100% !important;
    text-transform: uppercase;
    animation: pulseGlow 3s infinite;
}

.stButton>button:hover {
    box-shadow: 0 10px 30px rgba(0, 242, 254, 0.4) !important;
    transform: translateY(-2px) !important;
    filter: brightness(1.1);
}

/* Elegant Tags/Pills */
.model-tag {
    display: inline-block;
    padding: 8px 16px;
    background: linear-gradient(90deg, rgba(0, 242, 254, 0.1), rgba(0, 242, 254, 0.05));
    border: 1px solid rgba(0, 242, 254, 0.3);
    border-radius: 30px;
    color: #00f2fe;
    font-size: 0.9rem;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 8px;
    backdrop-filter: blur(4px);
    box-shadow: 0 2px 10px rgba(0,242,254,0.1);
}
.fallback-tag {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #b0b0b0;
}

/* Progress bars */
.stProgress > div > div > div > div {
/* Fix Streamlit Multiselect input cutoff WITHOUT breaking popover positioning */
div[data-baseweb="select"] > div {
    overflow-x: auto !important;
    overflow-y: hidden !important;
    flex-wrap: nowrap !important;
}
div[data-baseweb="select"] > div::-webkit-scrollbar {
    display: none; /* Hide scrollbar for clean look */
}
.stMultiSelect div[data-baseweb="select"] span[data-baseweb="tag"] {
    white-space: nowrap !important;
}

/* Fix st.metric truncation (e.g. claude-haiku-...) */
[data-testid="stMetricValue"] > div {
    white-space: normal !important;
    word-break: break-word !important;
    line-height: 1.1 !important;
    overflow: visible !important;
    text-overflow: unset !important;
}
[data-testid="stMetricValue"] {
    white-space: normal !important;
    word-break: break-word !important;
    overflow: visible !important;
    text-overflow: unset !important;
}

/* Style Segmented Control (Switch Button) */
[data-testid="stSegmentedControl"] {
    background: rgba(10, 10, 15, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 4px;
}
[data-testid="stSegmentedControl"] label {
    color: #fff !important;
}


</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("NEURAL_GATEWAY_API_URL", "http://127.0.0.1:8000").rstrip("/")

@st.cache_data(ttl=300)
def fetch_available_models():
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        if resp.status_code == 200:
            models = resp.json()
            # Sort by provider first, then by name
            models.sort(key=lambda m: (m.get("provider", ""), m.get("name", "")))
            
            names = [m["name"] for m in models]
            provider_map = {m["name"]: m.get("provider", "Unknown") for m in models}
            return names, provider_map
    except Exception:
        pass
    return [], {}

available_models, provider_map = fetch_available_models()

st.title("🌌 Neural Gateway")
st.markdown("<p style='color: #a0a0a0; font-size: 1.1rem; margin-top: -10px; margin-bottom: 2rem;'>Enterprise-grade AI Routing Gateway</p>", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### ⚙️ Request Constraints")
    
    col1, col2 = st.columns(2)
    with col1:
        req_json = st.toggle("Require JSON", value=False)
        req_search = st.toggle("Web Search", value=False)
    with col2:
        req_ocr = st.toggle("Require OCR", value=False)
        req_cite = st.toggle("Citations", value=False)
        
    st.markdown("---")
    max_latency = st.slider("Max Latency (ms)", min_value=500, max_value=20000, value=5000, step=500)
    max_cost = st.slider("Max Cost ($/1M)", min_value=0.1, max_value=50.0, value=10.0, step=0.5)
    
    tenant_id = st.text_input("Tenant ID", value="tenant-ui-demo")
    allowed_models = st.multiselect(
        "Allowed Models (Leave empty for all)", 
        options=available_models,
        format_func=lambda x: f"[{provider_map.get(x, 'Unknown')}] {x}"
    )
    
    st.markdown("---")
    st.markdown("---")
    st.markdown("### 🔑 API Credentials")
    openrouter_key = st.text_input("OpenRouter API Key", type="password", help="Required for Stage 2 LLM Generation")
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #555; font-size: 0.8rem;'>
            Powered by Bayesian Inference<br/>& Thompson Sampling
        </div>
    """, unsafe_allow_html=True)

# --- Main Interface ---
def get_friendly_reason(raw_data):
    if not raw_data:
        return "Unknown"
    status = raw_data.get("decision_record", {}).get("status", "")
    if status == "plan_exceeds_hard_constraints":
        constraints = raw_data.get("decision_record", {}).get("plan_constraints", {})
        est_cost = constraints.get("expected_cost_usd", 0)
        max_cost = constraints.get("max_cost_usd")
        est_lat = constraints.get("expected_latency_ms", 0)
        max_lat = constraints.get("max_latency_ms")
        
        reason = "The best available AI model exceeds your sidebar limits.\n\n"
        if max_cost is not None and est_cost > max_cost:
            reason += f"* **Cost:** The task would cost **${est_cost:.4f}**, which is above your limit of **${max_cost:.2f}**.\n"
        if max_lat is not None and est_lat > max_lat:
            reason += f"* **Latency:** The task would take **{est_lat/1000:.1f}s**, which is above your limit of **{max_lat/1000:.1f}s**.\n"
        return reason + "\n*Tip: Try increasing the sliders in the sidebar.*"
    elif status == "no_feasible_models":
        return "No AI models in our registry are capable of handling this request. (e.g. The document is too large for any model's context window, or no model supports the required features like Web Search)."
    elif status == "no_models_after_policy":
        return "Models are available, but they were blocked by your active Tenant Policies."
    return f"Raw Status Code: `{status}`"

if "decision" not in st.session_state:
    st.session_state.decision = None
if "abstained" not in st.session_state:
    st.session_state.abstained = False
if "status" not in st.session_state:
    st.session_state.status = ""
if "task_summary" not in st.session_state:
    st.session_state.task_summary = None
if "trace" not in st.session_state:
    st.session_state.trace = None
if "prompt" not in st.session_state:
    st.session_state.prompt = ""
prompt = st.text_area("Enter your prompt for the AI to process:", height=150, placeholder="e.g. Write a complex python script to analyze financial data...")
uploaded_files = st.file_uploader("Upload Artifacts (PDF, Images, Excel, etc.)", accept_multiple_files=True)

if st.button("🚀 Route Request"):
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        tc = {"tenant_id": tenant_id}
        if allowed_models:
            tc["allowed_models"] = allowed_models

        payload = {
            "prompt": prompt,
            "request_constraints": {
                "require_json": req_json,
                "require_ocr": req_ocr,
                "require_web_search": req_search,
                "require_citations": req_cite,
                "max_latency_ms": float(max_latency),
                "max_cost_usd": float(max_cost)
            },
            "tenant_context": tc
        }
        
        saved_file_paths = []
        temp_dir = None
        if uploaded_files:
            temp_dir = tempfile.mkdtemp()
            for uf in uploaded_files:
                file_path = os.path.join(temp_dir, uf.name)
                with open(file_path, "wb") as f:
                    f.write(uf.getbuffer())
                saved_file_paths.append(file_path)
            payload["files"] = saved_file_paths
        
        with st.spinner("Neural Gateway is analyzing complexity and scoring models..."):
            start_time = time.time()
            try:
                resp = requests.post(f"{API_URL}/route", json=payload, timeout=60.0)
                elapsed = time.time() - start_time
                
                if resp.status_code == 200:
                    data = resp.json()
                    
                    if data.get("abstain"):
                        st.warning("Routing Abstained! See details below.")
                    else:
                        st.success(f"Routing successful! Decided in {elapsed*1000:.1f}ms")
                    
                    st.session_state.decision = data.get("selected_plan", {})
                    st.session_state.abstained = data.get("abstain", False)
                    st.session_state.status = data.get("decision_record", {}).get("status", "")
                    st.session_state.task_summary = data.get("decision_record", {}).get("task_summary", {})
                    st.session_state.trace = data.get("decision_record", {}).get("pipeline_trace", {})
                    st.session_state.raw_data = data
                    st.session_state.prompt = prompt
                else:
                    st.error(f"Backend Error: {resp.status_code}")
                    st.write(resp.text)
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Connection Refused. Is the FastAPI server running on {API_URL}?")

if st.session_state.abstained:
    friendly_msg = get_friendly_reason(st.session_state.raw_data)
    st.error(f"🛑 **Router Abstained:** The request was rejected. \n\n{friendly_msg}")
    with st.expander("View Raw JSON Payload"):
        st.json(st.session_state.raw_data)

elif st.session_state.decision:
    decision = st.session_state.decision
    task_summary = st.session_state.task_summary
    trace = st.session_state.trace
    
    # Top Metrics Row
    m1, m2, m3, m4 = st.columns([1.7, 1, 1, 1])
    m1.metric("Selected Model", decision.get("selected_model", "N/A").split('/')[-1])
    m2.metric("Est. Latency", f"{decision.get('expected_latency_ms', 0):.0f} ms")
    m3.metric("Est. Cost", f"${decision.get('expected_cost_usd', 0):.4f}")
    m4.metric("Confidence", f"{decision.get('confidence', 0)*100:.1f}%")
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Routing Analysis", unsafe_allow_html=True)
    
    if task_summary.get('complexity') == 'high':
        st.info("🧠 **Complex task detected!** Risk tier automatically escalated to prioritize stronger reasoning models.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Semantic Understanding**")
        st.markdown(f"**Family:** `{task_summary.get('primary_family', 'N/A')}`")
        st.markdown(f"**Domain:** `{task_summary.get('domain', 'N/A')}`")
        st.markdown(f"**Complexity:** `{task_summary.get('complexity', 'N/A')}`")
        st.markdown(f"**Risk Tier:** `{task_summary.get('risk_tier', 'N/A')}`")
    
    with c2:
        st.markdown("**Fallback Models Prepared**")
        fallbacks = decision.get("fallback_models", [])
        if fallbacks:
            tags = "".join([f"<span class='model-tag fallback-tag'>{f.split('/')[-1]}</span>" for f in fallbacks])
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.markdown("*None*")
    st.markdown("</div>", unsafe_allow_html=True)
    
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Quality & Optimization Trace", unsafe_allow_html=True)
    
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Registry Size", trace.get("registry_models", 0))
    t2.metric("After Feasibility", trace.get("feasible_after_filter", 0))
    t3.metric("After Policy", trace.get("after_policy", 0))
    t4.metric("Pareto Frontier", trace.get("after_pareto", 0))
    
    st.markdown("#### Quality Breakdown")
    q = decision.get("explanation", {}).get("quality_breakdown", {})
    
    if q:
        st.caption("Contextual Fit")
        st.progress(float(q.get("contextual_mean", 0)))
        st.caption("Workflow Fit")
        st.progress(float(q.get("workflow_fit", 0)))
        st.caption("Domain Fit")
        st.progress(float(q.get("domain_fit", 0)))
        st.caption("Runtime Adjusted Mean")
        st.progress(float(q.get("runtime_adjusted_mean", 0)))
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    with st.expander("View Raw JSON Payload"):
        st.json(st.session_state.raw_data)
        
    st.markdown("---")
    st.markdown("### 🧠 Train the Engine (Dynamic KNN Memory)")
    st.info("Did the semantic parser route this incorrectly? Submit the correct mapping to instantly rebuild the vector matrix!")
    
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1.5])
    with col1:
        correct_family = st.selectbox(
            "Primary Family",
            options=["chat", "coding", "reasoning", "mathematics", "vision", "ocr", "document_qa", "audio", "agent", "translation", "summarization", "image_generation", "video_generation"],
            index=0,
            key="feedback_family"
        )
    with col2:
        correct_domain = st.selectbox(
            "Domain",
            options=["general", "software", "medical", "legal", "finance", "security", "crm", "hrm", "project", "accounts"],
            index=0,
            key="feedback_domain"
        )
    with col3:
        correct_risk = st.selectbox(
            "Risk Tier",
            options=["low", "medium", "high", "extreme"],
            index=0,
            key="feedback_risk"
        )
    with col4:
        correct_complexity = st.selectbox(
            "Complexity",
            options=["low", "medium", "high"],
            index=0,
            key="feedback_complexity"
        )
    with col5:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Submit", type="primary", use_container_width=True):
            try:
                resp = requests.post(f"{API_URL}/train_parser", json={
                    "prompt": st.session_state.prompt,
                    "primary_family": correct_family,
                    "domain": correct_domain,
                    "risk_tier": correct_risk,
                    "complexity": correct_complexity
                })
                if resp.status_code == 200:
                    st.toast("✅ Training example added successfully!", icon="🎉")
                    st.balloons()
                    st.success("✅ Matrix Rebuilt! The routing engine has learned your preference.")
                else:
                    st.error("Failed to update dataset.")
            except Exception as e:
                st.error(f"Error: {e}")


    st.markdown("---")
    st.markdown("### ⚡ Stage 2: LLM Execution")
    
    if st.button("Generate Response with Selected Model"):
        if not openrouter_key:
            st.error("Please enter your OpenRouter API Key in the sidebar to generate a response.")
        else:
            selected_model_id = decision.get("selected_model")
            st.info(f"Executing request using `{selected_model_id}` via OpenRouter...")
            
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key
            )
            
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # Stream the response
                completion = client.chat.completions.create(
                    model=selected_model_id,
                    messages=[{"role": "user", "content": st.session_state.prompt}],
                    stream=True,
                )
                
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")
                
                import re
                
                # Auto-format raw image URLs returned by Image Generation models
                if re.match(r"^https?://[^\s]+$", full_response.strip()):
                    full_response = f"![Generated Image]({full_response.strip()})"
                
                response_placeholder.markdown(full_response)
                st.success("Generation Complete!")
                
            except Exception as e:
                st.error(f"Execution Failed: {str(e)}")
