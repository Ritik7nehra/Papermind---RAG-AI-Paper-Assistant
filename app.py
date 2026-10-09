"""Streamlit Web UI: AI Research Paper Assistant.

Chat with any research paper (PDF) using LangChain, Chroma, and RAG.
Features dual execution modes:
  - Simulated / Mock Mode (Free, offline, local synthesis)
  - OpenAI API Mode (gpt-4o, gpt-4o-mini, text-embedding-3-small)

Run with:
    streamlit run app.py
"""

import hashlib
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
import rag

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling (Rich Aesthetics & Premium UI)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Polish */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 16px;
    }

    /* Metric Badges */
    .pill-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 10px;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        color: #e2e8f0;
        font-weight: 500;
    }
    .pill-primary {
        background: rgba(56, 189, 248, 0.1);
        border-color: rgba(56, 189, 248, 0.3);
        color: #38bdf8;
    }
    .pill-success {
        background: rgba(52, 211, 153, 0.1);
        border-color: rgba(52, 211, 153, 0.3);
        color: #34d399;
    }

    /* Suggestions Grid */
    .suggestions-header {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 10px;
    }

    /* Chunk Source Card */
    .chunk-card {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #38bdf8;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .chunk-meta {
        font-size: 0.75rem;
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .chunk-text {
        color: #cbd5e1;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        white-space: pre-wrap;
    }

    /* Sidebar Clean styling */
    section[data-testid="stSidebar"] {
        background-color: #0b1120;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SUGGESTIONS = [
    "What is the main idea of this paper?",
    "Summarize the methodology in 5 points.",
    "What are the limitations?",
    "What were the results and benchmark scores?",
    "What optimizer and hardware were used?",
]


# -----------------------------------------------------------------------------
# Document Upload Helper
# -----------------------------------------------------------------------------
def get_or_save_pdf(uploaded_file) -> Tuple[Path, str, int]:
    """Persist uploaded PDF in data/uploads/ to prevent temporary file deletion issues."""
    if uploaded_file is None:
        if not rag.SAMPLE_PDF.exists():
            raise FileNotFoundError("Default sample paper not found at data/paper.pdf")
        return rag.SAMPLE_PDF, "sample paper (Attention-Based Sequence Models)", len(rag.SAMPLE_PDF.read_bytes())

    uploads_dir = rag.UPLOADS_DIR
    uploads_dir.mkdir(parents=True, exist_ok=True)

    bytes_data = uploaded_file.getvalue()
    fp = hashlib.sha256(bytes_data).hexdigest()[:12]
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", uploaded_file.name)
    target_path = uploads_dir / f"{fp}_{safe_name}"

    if not target_path.exists():
        target_path.write_bytes(bytes_data)

    return target_path, uploaded_file.name, len(bytes_data)


# -----------------------------------------------------------------------------
# Assistant Factory with Cache
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing document into Chroma vector store...")
def get_chat_assistant(
    pdf_path_str: str,
    chunk_size: int,
    chunk_overlap: int,
    k: int,
    use_mock: bool,
    api_key: str = "",
    model_name: str = "",
    temperature: float = 0.0,
):
    """Build or reopen the index for this PDF and return a PaperChat assistant."""
    path = Path(pdf_path_str)
    return rag.create_assistant(
        pdf_path=path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        k=k,
        is_mock=use_mock,
        api_key=api_key or None,
        model_name=model_name or None,
        temperature=temperature,
    )


def render_sources_expander(searched: Optional[str], docs: list):
    """Render a clean, formatted sources breakdown panel."""
    pages = sorted({d.metadata.get("page", 0) + 1 for d in docs})
    page_str = ", ".join(map(str, pages)) if pages else "N/A"

    with st.expander(f"📑 Sources & Retrieved Chunks (Pages {page_str} • {len(docs)} chunks)"):
        if searched:
            st.markdown(f"🔍 **Retriever Search Query:** `{searched}`")

        for idx, doc in enumerate(docs, 1):
            p_num = doc.metadata.get("page", 0) + 1
            st.markdown(
                f"""
                <div class="chunk-card">
                    <div class="chunk-meta">Chunk #{idx} &nbsp;•&nbsp; Page {p_num}</div>
                    <div class="chunk-text">{doc.page_content.strip()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📄 Paper Assistant")
    st.caption("Chat with research papers using LangChain & Chroma RAG")

    st.markdown("---")
    st.subheader("⚙️ Execution Mode")

    default_mock = rag.is_mock_mode()
    mode_selection = st.radio(
        "Mode",
        ["🧪 Simulated / Mock Mode (Free)", "🌐 OpenAI API"],
        index=0 if default_mock else 1,
        help="Simulated mode runs offline locally without API keys or costs. OpenAI mode connects to production GPT models.",
    )
    use_mock = "Simulated" in mode_selection

    api_key_input = ""
    selected_model = rag.CHAT_MODEL
    temperature = 0.0

    if not use_mock:
        saved_key = os.getenv("OPENAI_API_KEY", "")
        if saved_key in ("mock-mode", "mock", "your_api_key_here"):
            saved_key = ""

        api_key_input = st.text_input(
            "OpenAI API Key",
            value=saved_key,
            type="password",
            placeholder="sk-...",
            help="Your API key is used strictly for this session.",
        )
        selected_model = st.selectbox(
            "OpenAI Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0,
            help="gpt-4o-mini is optimized for speed and cost; gpt-4o offers flagship reasoning.",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Lower values (0.0) ensure strictly grounded factual answers.",
        )
        if not api_key_input:
            st.warning("⚠️ Enter an OpenAI API key or toggle to Simulated Mode.")
    else:
        st.info("🧪 **Offline Simulated Mode Active**: Embeddings and synthesis execute locally without external API calls.")

    st.markdown("---")
    st.subheader("📁 Upload Research Paper")
    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload any research paper in PDF format. Defaults to included sample paper.",
    )

    st.markdown("---")
    st.subheader("🔧 RAG Hyperparameters")
    chunk_size = st.slider("Chunk Size (characters)", 200, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 500, 150, step=25)
    k_chunks = st.slider("Retrieved Chunks per Query (k)", 1, 10, 4)

    st.markdown("---")
    st.subheader("🧹 Actions")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Chat Export
    if "messages" in st.session_state and st.session_state.messages:
        export_text = "# AI Research Paper Assistant - Conversation Export\n\n"
        export_text += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        export_text += f"**Mode:** {'Simulated/Mock' if use_mock else f'OpenAI ({selected_model})'}\n\n---\n\n"
        for msg in st.session_state.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            export_text += f"### {role}\n{msg['content']}\n\n"
            if msg.get("sources"):
                pages = sorted({d.metadata.get("page", 0) + 1 for d in msg["sources"]})
                export_text += f"*Sources: Pages {', '.join(map(str, pages))}*\n\n"
            export_text += "---\n\n"

        st.download_button(
            "📥 Export Chat (Markdown)",
            data=export_text,
            file_name="paper_chat_history.md",
            mime="text/markdown",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
if not use_mock and not api_key_input:
    st.error(
        "OpenAI API Mode is enabled, but no API key was provided. "
        "Please enter your OpenAI API key in the sidebar, or switch to **🧪 Simulated / Mock Mode**."
    )
    st.stop()

if chunk_overlap >= chunk_size:
    st.error("Chunk overlap must be strictly smaller than the chunk size.")
    st.stop()

# Resolve Document
try:
    active_pdf_path, display_name, file_size_bytes = get_or_save_pdf(uploaded_pdf)
except Exception as e:
    st.error(f"Error loading document: {e}")
    st.stop()

# Initialize / Load Assistant
try:
    chat_assistant = get_chat_assistant(
        str(active_pdf_path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        k=k_chunks,
        use_mock=use_mock,
        api_key=api_key_input,
        model_name=selected_model,
        temperature=temperature,
    )
    doc_stats = chat_assistant.get_stats()
except Exception as exc:
    st.error(f"Failed to index paper into Chroma vector store: {exc}")
    st.stop()

# Reset conversation if document changes
if st.session_state.get("current_paper") != display_name or "messages" not in st.session_state:
    st.session_state.current_paper = display_name
    st.session_state.messages = []

# Hero Banner & Document Details
size_kb = file_size_bytes / 1024
size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"
mode_badge = "🧪 Simulated Mode (Offline)" if use_mock else f"🤖 OpenAI ({selected_model})"

st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-title">AI Research Paper Assistant</div>
        <div class="hero-subtitle">Interactive literature exploration with retrieval-augmented generation & source verification.</div>
        <div class="pill-container">
            <span class="pill pill-primary">📄 {display_name}</span>
            <span class="pill pill-success">{mode_badge}</span>
            <span class="pill">📦 {doc_stats['chunk_count']} Chunks Indexed</span>
            <span class="pill">⚖️ {size_str}</span>
            <span class="pill">🔍 Top-k: {k_chunks}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources_expander(message.get("searched"), message["sources"])

# Suggested Questions (Rendered if conversation is clean)
selected_prompt = None
if not st.session_state.messages:
    st.markdown('<div class="suggestions-header">💡 Suggested Explorations</div>', unsafe_allow_html=True)
    s_cols = st.columns(len(SUGGESTIONS))
    for i, suggestion in enumerate(SUGGESTIONS):
        if s_cols[i].button(suggestion, key=f"sugg_btn_{i}", use_container_width=True):
            selected_prompt = suggestion

# Chat Input Bar
user_query = st.chat_input("Ask a question about the research paper...") or selected_prompt

if user_query:
    history = list(st.session_state.messages)

    # Render User Turn
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate Assistant Turn
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving relevant passages and synthesizing response..."):
                standalone_q, docs = chat_assistant.retrieve(user_query, history)

            # Stream response
            response_text = st.write_stream(chat_assistant.stream_answer(user_query, history, docs))
        except Exception as exc:
            st.error(f"Error generating answer: {exc}")
        else:
            render_sources_expander(standalone_q if standalone_q != user_query else None, docs)

            # Record in conversation memory
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "sources": docs,
                    "searched": standalone_q if standalone_q != user_query else None,
                }
            )
