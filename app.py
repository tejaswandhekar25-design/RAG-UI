# 🌾 Agriculture RAG Assistant — Streamlit UI
# Choice: Streamlit — fastest path to integrate with an existing Python answer() function (one-line swap).

import streamlit as st

# ============================================================================
# 🔌 BACKEND INTEGRATION — Connected to rag_backend.py (ChromaDB + Gemini)
# ============================================================================
from rag_backend import answer as _rag_answer

def get_answer(query: str) -> str:
    """Calls the real RAG pipeline: retrieval (ChromaDB) → prompt → Gemini LLM."""
    return _rag_answer(query)
# ============================================================================


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agriculture RAG Assistant",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for agriculture theme & chat bubbles ──────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --green-900: #1a3a1a;
    --green-800: #2d5a27;
    --green-700: #3a7d32;
    --green-600: #4a9e3f;
    --green-500: #5cb85c;
    --green-400: #7ecf7e;
    --green-300: #a8dda8;
    --green-200: #c8eac8;
    --green-100: #e8f5e8;
    --green-50:  #f0faf0;
    --brown-700: #5d4037;
    --brown-600: #6d4c41;
    --brown-500: #795548;
    --brown-100: #efebe9;
    --warm-white: #fefdfb;
    --warm-gray:  #f5f3f0;
    --text-dark:  #2c2c2c;
    --text-muted: #6b6b6b;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: linear-gradient(170deg, var(--green-50) 0%, var(--warm-white) 40%, var(--warm-gray) 100%);
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 1.5rem 1rem 1rem;
    margin-bottom: 0.5rem;
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: var(--green-800);
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.app-header p {
    font-size: 0.95rem;
    color: var(--text-muted);
    margin: 0;
    max-width: 520px;
    margin-inline: auto;
    line-height: 1.5;
}

/* ── Chat container ── */
.chat-container {
    max-width: 720px;
    margin: 0 auto;
    padding: 0.5rem 0;
}

/* ── Chat bubbles ── */
.chat-bubble-row {
    display: flex;
    margin-bottom: 1rem;
    animation: fadeSlideIn 0.35s ease-out;
}
.chat-bubble-row.user {
    justify-content: flex-end;
}
.chat-bubble-row.assistant {
    justify-content: flex-start;
}

.chat-bubble {
    max-width: 78%;
    padding: 0.85rem 1.1rem;
    border-radius: 1.1rem;
    font-size: 0.92rem;
    line-height: 1.6;
    word-wrap: break-word;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

.chat-bubble.user {
    background: linear-gradient(135deg, var(--green-700), var(--green-600));
    color: #fff;
    border-bottom-right-radius: 0.3rem;
}
.chat-bubble.assistant {
    background: #fff;
    color: var(--text-dark);
    border: 1px solid var(--green-200);
    border-bottom-left-radius: 0.3rem;
}

.bubble-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.25rem;
    opacity: 0.7;
}
.chat-bubble.user .bubble-label { color: var(--green-200); }
.chat-bubble.assistant .bubble-label { color: var(--green-700); }

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Sample question chips ── */
.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin: 1rem 0 1.5rem;
}

/* ── Welcome empty state ── */
.welcome-card {
    text-align: center;
    padding: 2.5rem 1.5rem;
    background: rgba(255,255,255,0.7);
    border: 1px solid var(--green-200);
    border-radius: 1.2rem;
    margin: 1rem auto 0.5rem;
    max-width: 560px;
    backdrop-filter: blur(6px);
}
.welcome-card h3 {
    color: var(--green-800);
    font-weight: 600;
    margin: 0 0 0.5rem;
}
.welcome-card p {
    color: var(--text-muted);
    font-size: 0.9rem;
    line-height: 1.6;
    margin: 0;
}

/* ── Thinking indicator ── */
.thinking-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.8rem 1.1rem;
    background: #fff;
    border: 1px solid var(--green-200);
    border-radius: 1.1rem;
    border-bottom-left-radius: 0.3rem;
    max-width: 200px;
    font-size: 0.88rem;
    color: var(--green-700);
    font-weight: 500;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    animation: fadeSlideIn 0.3s ease-out;
}
.thinking-dots span {
    display: inline-block;
    width: 6px; height: 6px;
    background: var(--green-500);
    border-radius: 50%;
    animation: dotPulse 1.4s infinite ease-in-out;
    margin: 0 2px;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes dotPulse {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
}

/* ── Sidebar styling ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--green-900) 0%, #1e3320 100%);
}
section[data-testid="stSidebar"] * {
    color: #e0e0d0 !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--green-300) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15);
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 1.5rem 1rem 1rem;
    font-size: 0.78rem;
    color: var(--text-muted);
    opacity: 0.7;
}

/* ── Streamlit overrides ── */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
}

/* Sample Q buttons */
div[data-testid="stHorizontalBlock"] .stButton > button {
    border-radius: 2rem;
}

/* ── Mobile responsive tweaks ── */
@media (max-width: 640px) {
    .app-header h1 { font-size: 1.5rem; }
    .chat-bubble { max-width: 88%; font-size: 0.88rem; }
    .welcome-card { padding: 1.5rem 1rem; }
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# ── Sample questions ─────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "🌾 What is the yield of wheat in Punjab?",
    "📊 Compare rice and wheat production across states",
    "🧪 Which crops use the most fertilizer?",
    "🌿 Top 5 crops by area in Maharashtra",
    "📈 How has sugarcane production changed over the years?",
    "🌱 What is the average pesticide usage for cotton?",
]

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 About This Project")
    st.markdown(
        "This **Agriculture RAG Assistant** uses a Retrieval-Augmented "
        "Generation pipeline to answer questions about Indian agriculture data."
    )
    st.markdown(
        "**How it works:**\n"
        "1. Your question is converted to an embedding\n"
        "2. Relevant data chunks are retrieved from **ChromaDB**\n"
        "3. An **LLM** generates an answer grounded in the retrieved data"
    )

    st.divider()

    st.markdown("### 📋 Dataset Coverage")
    st.markdown("**Crops:** Arhar/Tur, Bajra, Castor Seed, Cotton, Gram, "
                "Groundnut, Jowar, Jute, Linseed, Maize, Mesta, Moong, "
                "Niger Seed, Onion, Potato, Rapeseed & Mustard, Rice, "
                "Ragi, Safflower, Sesamum, Soybean, Sugarcane, Sunflower, "
                "Sweet Potato, Tobacco, Turmeric, Urad, Wheat")
    st.markdown("**States:** Andhra Pradesh, Assam, Bihar, Chhattisgarh, "
                "Gujarat, Haryana, Jharkhand, Karnataka, Kerala, "
                "Madhya Pradesh, Maharashtra, Odisha, Punjab, Rajasthan, "
                "Tamil Nadu, Telangana, Uttar Pradesh, West Bengal & more")
    st.markdown("**Features:** Crop, Year, Season, State, Area, Production, "
                "Fertilizer, Pesticide, Yield")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.processing = False
        st.rerun()

    st.divider()
    st.caption("Built with Streamlit • ChromaDB • LLM")


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🌾 Agriculture RAG Assistant</h1>
    <p>Ask questions about Indian crop production, yield, fertilizer usage, 
    pesticide data, and more — powered by Retrieval-Augmented Generation.</p>
</div>
""", unsafe_allow_html=True)


# ── Helper: render a single chat bubble ──────────────────────────────────────
def render_bubble(role: str, content: str):
    """Render a styled chat bubble as HTML."""
    alignment = "user" if role == "user" else "assistant"
    label = "You" if role == "user" else "🌾 Assistant"
    st.markdown(f"""
    <div class="chat-bubble-row {alignment}">
        <div class="chat-bubble {alignment}">
            <div class="bubble-label">{label}</div>
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Helper: render sample question chips ─────────────────────────────────────
def render_sample_questions():
    """Show clickable sample question buttons."""
    cols = st.columns(2)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                submit_question(q)


# ── Submit a question ────────────────────────────────────────────────────────
def submit_question(query: str):
    """Add user message, call backend, add response."""
    query = query.strip()
    if not query:
        return

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.processing = True
    st.rerun()


# ── Process pending question (runs after rerun) ─────────────────────────────
def process_pending():
    """If we're in processing state, call the backend and store the answer."""
    if not st.session_state.processing:
        return False
    if not st.session_state.messages:
        st.session_state.processing = False
        return False

    last_msg = st.session_state.messages[-1]
    if last_msg["role"] != "user":
        st.session_state.processing = False
        return False

    return True


# ── Main chat area ───────────────────────────────────────────────────────────

# Empty state
if not st.session_state.messages and not st.session_state.processing:
    st.markdown("""
    <div class="welcome-card">
        <h3>👋 Welcome!</h3>
        <p>I can help you explore Indian agriculture data — ask about crop yields, 
        production trends, fertilizer and pesticide usage, state-wise comparisons, 
        and more. Try one of the questions below to get started!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    render_sample_questions()

else:
    # Render chat history
    for msg in st.session_state.messages:
        render_bubble(msg["role"], msg["content"])

    # If we're waiting on a response, show thinking indicator & call backend
    if process_pending():
        # Show thinking animation
        st.markdown("""
        <div class="chat-bubble-row assistant">
            <div class="thinking-indicator">
                Thinking 🌱
                <span class="thinking-dots">
                    <span></span><span></span><span></span>
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        user_query = st.session_state.messages[-1]["content"]

        # ── Call the backend ─────────────────────────────────────────────
        try:
            answer = get_answer(user_query)
            if not answer or not answer.strip():
                answer = ("I couldn't find relevant data for that query. "
                          "Try rephrasing your question or asking about a "
                          "specific crop, state, or metric. 🌾")
        except Exception:
            answer = ("Sorry, something went wrong while processing your "
                      "question. Please try again or rephrase your query. 🌱")
        # ─────────────────────────────────────────────────────────────────

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.processing = False
        st.rerun()

    # Show sample questions below chat
    st.markdown("")
    st.markdown("##### 💡 Try asking:")
    render_sample_questions()


# ── Chat input (always at the bottom) ────────────────────────────────────────
st.markdown("")  # spacing

user_input = st.chat_input(
    placeholder="Ask about crops, yields, fertilizer usage, states...",
    disabled=st.session_state.processing,
)

if user_input:
    submit_question(user_input)


