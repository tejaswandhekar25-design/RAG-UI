# rag_backend.py — Member 3: LLM integration layer
# This file connects the retrieval pipeline (ChromaDB) to the Gemini LLM.
# The UI calls answer(query) — that's it.

import os
from google import genai

# ── Import the actual retrieval function from the existing pipeline ──────────
# This is the real retrieval_pipeline.py (Member 2's work), copied into this project.
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
    "Answer the user's question using ONLY the context below, which is "
    "retrieved from an Indian agriculture dataset containing information about "
    "crops, years, seasons, states, area, production, fertilizer usage, "
    "pesticide usage, and yield.\n\n"
    "Rules:\n"
    "1. If the context does not contain enough information to answer "
    "confidently, say so honestly instead of guessing or hallucinating.\n"
    "2. If the question requires comparing or ranking across multiple "
    "crops/states and the context only has a few rows, explicitly note "
    "that your answer is based on limited retrieved data, not the full dataset.\n"
    "3. When citing numbers, include the units and source fields "
    "(e.g., yield in tonnes/hectare, area in hectares).\n"
    "4. Be concise but informative. Use bullet points or tables when "
    "comparing multiple items.\n"
    "5. Answer in the same language the question was asked in."
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

    # ── Step 1: Retrieve ─────────────────────────────────────────────────
    try:
        chunks = retrieve(query, k=5, score_threshold=0.3)
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

    # ── Step 4: Call Gemini LLM (new google-genai SDK) ───────────────────
    try:
        api_key = _get_api_key()
        client = genai.Client(api_key=api_key)

        # Using gemini-3.5-flash — fast, cheap, great for RAG grounding
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )

        answer_text = response.text.strip()
        if not answer_text:
            return (
                "I retrieved some data but couldn't generate a clear answer. "
                "Try rephrasing your question. 🌱"
            )
        return answer_text

    except ValueError as e:
        # API key not configured
        return f"⚠️ {str(e)}"

    except Exception as e:
        print(f"[RAG Backend] LLM error: {e}")
        return (
            "Sorry, I encountered an error while generating the answer. "
            "This might be a temporary API issue. Please try again in a moment. 🌱"
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
