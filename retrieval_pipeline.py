from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIRECTORY = "db/chroma_db"

# ---- हे फक्त एकदाच, file import झाल्यावर चालतं (परत परत नाही -- speed साठी) ----
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)
# --------------------------------------------------------------


def retrieve(query, k=3, score_threshold=0.3):
    """
    Member 3 हे function import करून वापरेल:
        from retrieval_pipeline import retrieve
        chunks = retrieve("cotton साठी pH किती?")

    Dataset मध्ये संबंधित माहिती नसेल, तर रिकामी list ([]) परत येते --
    error येत नाही. Member 3 ने त्याच्या prompt मध्ये अशी सूचना द्यावी:
    "context रिकामा असल्यास, माहिती उपलब्ध नाही असं स्पष्ट सांग."
    """
    try:
        local_retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold}
        )
        results = local_retriever.invoke(query)
        return [doc.page_content for doc in results]
    except Exception as e:
        print(f"Retrieval error for query '{query}': {e}")
        return []


# ---- स्वतःच्या टेस्टिंगसाठी (hidden-question style प्रश्नांसह) ----
if __name__ == "__main__":
    test_questions = [
        # Direct
        "What is the yield of rice in Chhattisgarh?",
        "Andhra Pradesh मधली soil ची N, P, K value किती आहे?",
        # Paraphrased
        "छत्तीसगडमध्ये भाताचं उत्पादन किती होतं?",
        "आंध्र प्रदेशच्या मातीत नत्र किती आहे?",
        # Indirect / reasoning
        "कोणत्या राज्यात rice चं yield सर्वात जास्त आहे?",
        "Maize आणि rice पैकी कोणाचं production जास्त आहे?",
        # Domain-specific, वेगळं phrasing
        "कापसासाठी कोणता season योग्य आहे?",
        # Edge case - dataset मध्ये नसलेली माहिती
        "Punjab मध्ये cotton चं production किती आहे?",
        "2025 सालचा data आहे का?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print("=" * 60)
        results = retrieve(q)
        if not results:
            print("⚠️ कोणतेही chunks सापडले नाहीत (edge-case प्रश्नांसाठी हे ठीक आहे)")
        for i, chunk in enumerate(results, 1):
            print(f"\nChunk {i}:\n{chunk[:300]}")
