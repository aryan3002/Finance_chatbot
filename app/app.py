# app/app.py
import os
import pickle
from pathlib import Path
from typing import List, Dict
from html import escape

import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from prompts import SYSTEM_PROMPT, QA_PROMPT

# ----------------------------
# Env & constants
# ----------------------------
load_dotenv()
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()
INDEX_PATH = Path("artifacts/index.faiss")
DOCSTORE_PATH = Path("artifacts/docstore.pkl")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ----------------------------
# LLM provider wrapper
# ----------------------------
def llm_complete(prompt: str, provider: str, temperature: float = 0.2) -> str:
    provider = (provider or DEFAULT_PROVIDER).lower()

    if provider == "groq":
        try:
            from groq import Groq
        except Exception:
            return "⚠️ Missing dependency: `pip install groq`"
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "⚠️ GROQ_API_KEY not set in your .env"
        client = Groq(api_key=api_key)
        try:
            resp = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=float(temperature),
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"❌ Groq error: {e}"

    elif provider == "openai":
        try:
            from openai import OpenAI
        except Exception:
            return "⚠️ Missing dependency: `pip install openai`"
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "⚠️ OPENAI_API_KEY not set in your .env"
        client = OpenAI(api_key=api_key)
        try:
            resp = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=float(temperature),
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"❌ OpenAI error: {e}"

    return "Local mode placeholder. Set LLM_PROVIDER=groq and add GROQ_API_KEY in .env for real answers."

# ----------------------------
# Index / embedder loaders
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMB_MODEL)

@st.cache_resource(show_spinner=False)
def load_index_and_store():
    if not INDEX_PATH.exists() or not DOCSTORE_PATH.exists():
        raise FileNotFoundError(
            "Index or docstore not found. Run:\n"
            "  python scripts/ingest.py\n"
            "  python scripts/chunk_and_index.py"
        )
    index = faiss.read_index(str(INDEX_PATH))
    with open(DOCSTORE_PATH, "rb") as f:
        docstore = pickle.load(f)
    return index, docstore

def retrieve(query: str, k: int = 5) -> List[Dict]:
    embedder = load_embedder()
    index, ds = load_index_and_store()
    q_vec = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(q_vec, k)
    hits = []
    for rank, idx in enumerate(I[0]):
        if idx < 0:
            continue
        hits.append({
            "rank": rank + 1,
            "text": ds["texts"][idx],
            "meta": ds["metadatas"][idx],
            "score": float(D[0][rank]),
        })
    return hits

def format_context(hits: List[Dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        src = h["meta"].get("source", "unknown")
        blocks.append(f"[Source {i}] ({src})\n{h['text']}\n")
    return "\n\n".join(blocks)

# ----------------------------
# UI Setup
# ----------------------------
st.set_page_config(page_title="Compliance Q&A", page_icon="⚖️", layout="wide")

# Custom styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --primary-color: #4f46e5;
        --secondary-color: #1d4ed8;
        --accent-color: #14b8a6;
        --surface-color: #111827;
        --surface-muted: #0b1220;
        --surface-elevated: #16213d;
        --text-primary: #f4f6ff;
        --text-muted: #a0b4d6;
        --border-subtle: rgba(116, 139, 255, 0.18);
    }

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stMarkdown td, .stMarkdown th {
        color: var(--text-primary) !important;
    }

    .stMarkdown table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.4rem;
        margin-bottom: 0.6rem;
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(79, 70, 229, 0.28);
        border-radius: 12px;
        overflow: hidden;
    }

    .stMarkdown th {
        background: rgba(59, 130, 246, 0.12);
        font-weight: 600;
    }

    .stMarkdown td, .stMarkdown th {
        border: 1px solid rgba(79, 70, 229, 0.18);
        padding: 0.55rem 0.75rem;
    }

    .stMarkdown tr:nth-child(even) td {
        background: rgba(15, 23, 42, 0.75);
    }

    #root > div:has(> div[data-testid="stDecoration"]){
        display: none;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #040711 0%, #101a34 100%);
    }

    div.block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1120px;
    }

    #MainMenu, footer,
    header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0;
    }

    .stButton button {
        border-radius: 999px;
        border: none;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: #ffffff;
        font-weight: 600;
        padding: 0.55rem 1.5rem;
        box-shadow: 0 16px 40px rgba(31, 60, 136, 0.34);
        transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 22px 48px rgba(30, 64, 175, 0.5);
        filter: brightness(1.05);
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 16px;
        border: 1px solid rgba(79, 70, 229, 0.35);
        background: rgba(15, 23, 42, 0.9);
        color: var(--text-primary);
        box-shadow: 0 12px 34px rgba(11, 17, 33, 0.55);
    }

    [data-testid="stChatInput"] label {
        font-weight: 600;
        color: rgba(228, 233, 255, 0.85);
    }

    section[data-testid="stSidebar"] {
        background: #050915;
        border-right: 1px solid rgba(79, 70, 229, 0.18);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2.5rem;
    }

    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label {
        color: rgba(244, 246, 255, 0.92);
    }

    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSelectbox label {
        font-weight: 500;
    }

    section[data-testid="stSidebar"] .stSlider,
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stAlert {
        background: rgba(36, 56, 112, 0.25);
        border-radius: 14px;
        padding: 0.5rem 0.75rem;
        border: 1px solid rgba(147, 197, 253, 0.18);
    }

    .hero {
        background: linear-gradient(120deg, rgba(27, 38, 69, 0.85), rgba(42, 62, 118, 0.85));
        padding: 2.4rem 2.8rem;
        border-radius: 24px;
        color: #ffffff;
        box-shadow: 0 30px 60px rgba(5, 12, 32, 0.65);
        margin-bottom: 2.1rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(79, 70, 229, 0.25);
    }

    .hero::after {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 20% 20%, rgba(94, 234, 212, 0.2), transparent 60%),
                    radial-gradient(circle at 80% 10%, rgba(96, 165, 250, 0.32), transparent 55%);
        opacity: 0.65;
        pointer-events: none;
    }

    .hero__eyebrow {
        text-transform: uppercase;
        font-size: 0.83rem;
        letter-spacing: 0.25rem;
        margin-bottom: 0.9rem;
        opacity: 0.72;
    }

    .hero h1 {
        font-size: 2.6rem;
        line-height: 1.2;
        margin: 0;
        position: relative;
        z-index: 1;
    }

    .hero p {
        margin-top: 0.9rem;
        font-size: 1.1rem;
        max-width: 620px;
        position: relative;
        z-index: 1;
        opacity: 0.88;
    }

    .info-card,
    .about-card,
    .tips-card,
    .source-card {
        background: var(--surface-color);
        border-radius: 20px;
        padding: 1.4rem;
        box-shadow: 0 22px 45px rgba(3, 7, 18, 0.45);
        border: 1px solid rgba(56, 97, 177, 0.28);
    }

    .info-card {
        padding: 1.6rem 1.4rem 1.5rem;
    }

    .info-card__icon {
        width: 46px;
        height: 46px;
        border-radius: 12px;
        display: grid;
        place-items: center;
        background: rgba(31, 60, 136, 0.12);
        color: var(--primary-color);
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }

    .info-card__title {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.5rem;
        color: var(--text-primary);
    }

    .info-card__body {
        color: var(--text-muted);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .about-card strong {
        font-size: 1rem;
        color: var(--text-primary);
    }

    .section-title {
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: rgba(160, 180, 214, 0.9);
        margin-bottom: 0.9rem;
    }

    .tips-card ul {
        padding-left: 1.2rem;
        margin: 0.4rem 0 0;
        color: var(--text-muted);
    }

    .tips-card li {
        margin-bottom: 0.35rem;
    }

    .chat-bubble {
        border-radius: 18px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 12px 36px rgba(3, 7, 18, 0.55);
        border: 1px solid rgba(62, 86, 179, 0.42);
        line-height: 1.58;
    }

    .chat-bubble.assistant-bubble {
        background: var(--surface-elevated);
        color: var(--text-primary);
    }

    .chat-bubble.user-bubble {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.92), rgba(59, 130, 246, 0.82));
        color: #ffffff;
        border: none;
    }

    .source-title {
        font-weight: 600;
        color: rgba(148, 163, 208, 0.95);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.75rem;
        margin: 1.2rem 0 0.6rem;
    }

    .source-card {
        margin-bottom: 0.75rem;
    }

    .source-card__title {
        font-weight: 600;
        margin-bottom: 0.45rem;
        color: var(--text-primary);
    }

    .source-card__body {
        color: var(--text-muted);
        font-size: 0.92rem;
        line-height: 1.5;
        white-space: pre-line;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="hero">
        <div class="hero__eyebrow">Financial Compliance Intelligence</div>
        <h1>Precision answers for your regulatory questions</h1>
        <p>Navigate AML, KYC, GDPR, PCI DSS, and reporting standards with grounded, source-backed responses tailored to your workflow.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Feature highlights
feature_cards = [
    ("🔍", "Targeted Retrieval", "Surface the most relevant obligations, controls, and citations in seconds."),
    ("🧭", "Audit-Ready", "Trace every answer back to its source to support documentation and reviews."),
    ("⚙️", "Configurable", "Tune model, temperature, and search depth to match your compliance posture."),
]

feature_cols = st.columns(3, gap="large")
for (icon, title, body), col in zip(feature_cards, feature_cols):
    with col:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-card__icon">{icon}</div>
                <div class="info-card__title">{title}</div>
                <div class="info-card__body">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

about_col, tips_col = st.columns([1.8, 1.2], gap="large")

with about_col:
    st.markdown(
        """
        <div class="about-card">
            <strong>ℹ️ About this assistant</strong>
            <p style="margin-top: 0.6rem; color: var(--text-muted);">
                The assistant continuously scans your indexed regulatory corpus to deliver contextual answers. Each response is supported by source excerpts so you can validate obligations and streamline reporting workflows.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tips_col:
    st.markdown(
        """
        <div class="tips-card">
            <div class="section-title">Pro tips</div>
            <ul>
                <li>Ask one obligation or scenario per question for precise citations.</li>
                <li>Adjust sources and temperature to balance breadth with specificity.</li>
                <li>Use the sources section to copy text directly into audit notes.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("Model Selection")
    provider = st.selectbox(
        "AI Model",
        options=["groq", "openai", "local"],
        index=["groq", "openai", "local"].index(DEFAULT_PROVIDER)
        if DEFAULT_PROVIDER in ["groq", "openai", "local"] else 0,
        help="Configure via .env (LLM_PROVIDER, GROQ_API_KEY/OPENAI_API_KEY)"
    )
    
    st.subheader("Search Configuration")
    top_k = st.slider(
        "Sources to Consider",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of regulatory documents to reference"
    )
    
    temperature = st.slider(
        "Response Style",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        help="Lower = more precise, Higher = more creative"
    )
    
    st.markdown(
        """
        <div style="margin-top: 1.1rem; background: rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 1rem 1.1rem; color: rgba(255, 255, 255, 0.82); font-size: 0.9rem; line-height: 1.5;">
            💡 <strong style="color: #ffffff;">Usage Tip</strong><br>
            Keep the response style low when you need precise regulatory language.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin-top: 1.5rem; color: rgba(255, 255, 255, 0.55); font-size: 0.78rem; line-height: 1.6;">
            Update the knowledge base by running <code>python scripts/ingest.py</code> followed by <code>python scripts/chunk_and_index.py</code> when new regulations are added.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": """Welcome! 👋 I'm your compliance assistant. I can help with:

• AML/KYC regulations
• Banking secrecy laws
• Data protection requirements
• Financial reporting guidelines
• Cross-border regulations

Ask me anything about financial compliance."""
    }]

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        role = message["role"]
        bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
        content_md = message["content"].replace("<br>", "  \n")

        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(
                f"<div class=\"chat-bubble {bubble_class}\">",
                unsafe_allow_html=True,
            )
            st.markdown(content_md)
            st.markdown("</div>", unsafe_allow_html=True)

            if role != "user" and message.get("sources"):
                st.markdown("<div class=\"source-title\">Sources consulted</div>", unsafe_allow_html=True)
                for src in message["sources"]:
                    src_title = escape(src["title"])
                    src_text = escape(src["text"]).replace("<br>", "\n").replace("\n", "<br>")
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-card__title">{src_title}</div>
                            <div class="source-card__body">{src_text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Chat input
user_input = st.chat_input("Ask your compliance question...")

# Handle user input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    try:
        hits = retrieve(user_input, k=top_k)
        ctx = format_context(hits) if hits else ""
        
        prompt = QA_PROMPT.format(question=user_input, context=ctx)
        answer = llm_complete(prompt, provider=provider, temperature=temperature)
        
        # Clean up the answer by replacing <br> with proper line breaks
        answer = answer.replace("<br>", "\n")
        
        sources = []
        if hits:
            for h in hits:
                text = h['text'][:600] + ("..." if len(h['text']) > 600 else "")
                # Clean up source text as well
                text = text.replace("<br>", "\n")
                sources.append({
                    "title": f"Source: {h['meta'].get('source', 'unknown')}",
                    "text": text
                })
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        
        st.rerun()
        
    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Error: {str(e)}")
