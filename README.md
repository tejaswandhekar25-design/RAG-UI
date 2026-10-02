# 🌾 Agriculture RAG Assistant

An AI-powered chatbot that answers questions about Indian agriculture data using **Retrieval-Augmented Generation (RAG)**.

![Agriculture RAG Assistant Screenshot](https://files.catbox.moe/pc6y3n.png)

## What It Does

Ask natural-language questions about Indian crop production, yield, fertilizer/pesticide usage, soil data, and weather patterns. The system:

1. **Retrieves** relevant data chunks from a ChromaDB vector store (embedded with `sentence-transformers/all-MiniLM-L6-v2`)
2. **Generates** grounded answers using Google Gemini 2.0 Flash LLM
3. **Displays** results in a polished Streamlit chat interface

### Dataset Coverage
- **Crops**: Rice, Wheat, Cotton, Sugarcane, Maize, Soybean, Arhar/Tur, Groundnut, and 20+ more
- **States**: All major Indian states (Punjab, Maharashtra, UP, Karnataka, etc.)
- **Metrics**: Area, Production, Yield, Fertilizer usage, Pesticide usage, Season, Year
- **Additional**: State-level soil data (N, P, K, pH) and weather data (1997–2020)


---

## 🚀 Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Gemini API key
Edit `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-actual-gemini-api-key"
```

Get a free API key at: https://aistudio.google.com/apikey

### 3. Run the app
```bash
streamlit run app.py
```
> **Windows Note**: If PowerShell says `'streamlit' is not recognized`, run:
> ```bash
> python -m streamlit run app.py
> ```

The app will open at `http://localhost:8501`.

---

## ☁️ Deploy on Streamlit Community Cloud

1. **Push to GitHub** (make sure `.streamlit/secrets.toml` is in `.gitignore` — it already is)
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app"
3. Connect your repo, set main file to `app.py`
4. Under **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-actual-gemini-api-key"
   ```
5. Deploy!

> **Note**: The `db/chroma_db/` folder (~22MB) must be included in the repo so the vector store is available on the server.

---

## 📁 Project Structure

```
├── app.py                    # Streamlit UI (chat interface)
├── rag_backend.py            # LLM integration (prompt + Gemini API call)
├── retrieval_pipeline.py     # ChromaDB retrieval (embeddings + similarity search)
├── db/chroma_db/             # Persisted ChromaDB vector store
├── assets/                   # App screenshots & UI assets
├── requirements.txt          # Python dependencies
├── .streamlit/secrets.toml   # API keys (git-ignored)
├── .gitignore
└── README.md
```

## Architecture

```
User Question
     │
     ▼
┌──────────┐    retrieve()     ┌──────────────────┐
│  app.py  │ ───────────────▶  │ retrieval_pipeline│
│ (Streamlit│                   │ (ChromaDB +       │
│   UI)    │                   │  MiniLM-L6-v2)   │
└──────────┘                   └──────────────────┘
     │                                │
     │  answer()                      │ chunks[]
     ▼                                ▼
┌──────────────┐         ┌──────────────────┐
│ rag_backend  │ ◀────── │ Retrieved context │
│ (Gemini LLM) │         └──────────────────┘
└──────────────┘
     │
     ▼
  Final Answer
```

