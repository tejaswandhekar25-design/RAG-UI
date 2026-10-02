# rag_backend.py — LLM integration layer
# This file connects the retrieval pipeline (ChromaDB) to the Gemini LLM.
# The UI calls answer(query) — that's it.

import os
from google import genai

# ── Import the retrieval function from the pipeline ──────────────────────────
# It uses: ChromaDB + sentence-transformers/all-MiniLM-L6-v2 embeddings
# Function signature: retrieve(query, k=3, score_threshold=0.3) -> list[str]
from retrieval_pipeline import retrieve


# ── Gemini API configuration ────────────────────────────────────────────────
def _get_api_key() -> str:
    """
    Read Gemini API key from Streamlit secrets first (for Streamlit Cloud),
    then fall back to environment variable (for local .env / terminal usage).
    """
    # Try Streamlit secrets first
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key and key != "PASTE_YOUR_GEMINI_API_KEY_HERE":
            return key
    except Exception:
        pass

    # Fall back to environment variable
    key = os.environ.get("GEMINI_API_KEY", "")
    if key and key != "PASTE_YOUR_GEMINI_API_KEY_HERE":
        return key

    raise ValueError(
        "GEMINI_API_KEY not found. Set it in .streamlit/secrets.toml "
        "or as an environment variable."
    )


# ── Prompt template ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an agriculture data assistant for India. "
    "Answer the user's question using the context below, which is "
    "retrieved from Indian agriculture datasets containing information about "
    "crops, years, seasons, states, districts, area, production, fertilizer usage, "
    "pesticide usage, yield, temperature, rainfall, soil data, and more.\n\n"
    "Rules:\n"
    "1. ALWAYS try to answer the question using the provided context. "
    "Extract relevant numbers, trends, and comparisons from the data chunks.\n"
    "2. If the context contains partial information (e.g., data for some "
    "crops/states but not all), answer based on what IS available and note "
    "that the answer covers the data found in the retrieved context.\n"
    "3. When citing numbers, include the units and source fields "
    "(e.g., yield in tonnes/hectare, area in hectares).\n"
    "4. Be concise but informative. Use bullet points or tables when "
    "comparing multiple items.\n"
    "5. For reasoning or analytical questions (e.g., 'why do yields differ'), "
    "use the data in context (soil, weather, fertilizer) to provide a "
    "data-driven answer. You may combine data insights with general "
    "agricultural knowledge for such questions.\n"
    "6. Only say you cannot answer if the context is completely irrelevant "
    "to the question. If there is ANY related data, use it to form an answer.\n"
    "7. Answer in the same language the question was asked in."
)


def _build_prompt(query: str, chunks: list[str]) -> str:
    """Build the full prompt with retrieved context."""
    context_block = "\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- Retrieved Context ({len(chunks)} chunks) ---\n"
        f"{context_block}\n"
        f"--- End of Context ---\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


# ── Main answer function (the ONLY function the UI needs) ───────────────────
def answer(query: str) -> str:
    """
    End-to-end RAG pipeline:
      1. Retrieve relevant chunks from ChromaDB
      2. Build a grounded prompt
      3. Call Gemini LLM
      4. Return the answer string

    This is the single function imported by app.py's get_answer().
    """
    query = query.strip()
    if not query:
        return "Please enter a question to get started! 🌾"

    # Quick friendly greeting check
    greetings = {"hi", "hello", "hey", "namaste", "hola", "good morning", "good evening", "good afternoon"}
    if query.lower().strip("!?., ") in greetings:
        return (
            "Namaste! 🙏 I'm your **Agriculture RAG Assistant**.\n\n"
            "I can help you explore Indian agriculture data across states, crops, yields, "
            "fertilizer/pesticide usage, and historical trends.\n\n"
            "💡 **Try asking questions like:**\n"
            "- *What is the yield of wheat in Punjab?*\n"
            "- *Compare rice and wheat production across states*\n"
            "- *Which crops use the most fertilizer?*\n"
            "- *Top 5 crops by area in Maharashtra*"
        )

    # ── Step 1: Retrieve ─────────────────────────────────────────────────
    try:
        chunks = retrieve(query, k=8)
    except Exception as e:
        print(f"[RAG Backend] Retrieval error: {e}")
        chunks = []

    # ── Step 2: Handle empty retrieval (skip LLM call — saves cost) ──────
    if not chunks:
        return (
            "I couldn't find relevant data for that question in the agriculture "
            "dataset. This could mean the specific crop, state, or metric you "
            "asked about isn't in the indexed data.\n\n"
            "💡 **Try rephrasing**, or ask about specific crops (e.g., Rice, "
            "Wheat, Cotton, Sugarcane) and states (e.g., Punjab, Maharashtra, "
            "Uttar Pradesh) that are well-covered in the dataset."
        )

    # ── Step 3: Build prompt ─────────────────────────────────────────────
    prompt = _build_prompt(query, chunks)

    # ── Step 4: Call Gemini LLM with resilient model fallback ───────────
    try:
        api_key = _get_api_key()
        client = genai.Client(api_key=api_key)
    except ValueError as e:
        return f"⚠️ {str(e)}"
    except Exception as e:
        return f"⚠️ API setup error: {str(e)}"

    candidate_models = [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-flash-latest",
        "gemini-3.8-flash",
    ]

    last_error = None
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            answer_text = response.text.strip() if response and response.text else ""
            if answer_text:
                return answer_text
        except Exception as e:
            last_error = e
            print(f"[RAG Backend] Model {model_name} failed: {e}. Trying fallback...")
            continue

    print(f"[RAG Backend] All models failed. Last error: {last_error}")
    return (
        "Sorry, the AI service is currently experiencing high demand or temporary network issues. "
        "Please try again in a few moments. 🌱"
    )


# ── Quick test (run this file directly to verify) ───────────────────────────
if __name__ == "__main__":
    import sys, io
    # Fix Windows cp1252 console encoding for emoji output
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    test_queries = [
        "What is the yield of wheat in Punjab?",
        "Compare rice and cotton production",
        "Which state has the highest sugarcane yield?",
    ]
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("=" * 60)
        result = answer(q)
        print(f"A: {result}")
