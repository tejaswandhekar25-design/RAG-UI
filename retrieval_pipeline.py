import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIRECTORY = "db/chroma_db"

# ── Load embedding model and ChromaDB once at import time (for speed) ──
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)
# -----------------------------------------------------------------------


def _decompose_query(query):
    """
    Break comparative / multi-topic queries into sub-queries so the
    retriever can find data for each part individually.
    E.g. "Compare rice and wheat production across states"
      -> ["rice production across states", "wheat production across states"]
    """
    q_lower = query.lower().strip()

    # Detect comparison patterns like "compare X and Y", "X vs Y", "X and Y production"
    compare_patterns = [
        r'compare\s+(.+?)\s+and\s+(.+?)(?:\s+production|\s+yield|\s+area|\s+across|\s+in\b|$)',
        r'(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\s+production|\s+yield|\s+area|\s+across|\s+in\b|$)',
        r'difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\s+production|\s+yield|\s+area|\s+across|\s+in\b|$)',
    ]

    for pattern in compare_patterns:
        match = re.search(pattern, q_lower)
        if match:
            item1 = match.group(1).strip()
            item2 = match.group(2).strip()
            # Build sub-queries with the remaining context
            context = q_lower
            for word in ['compare', 'vs', 'versus', 'difference between', 'and']:
                context = context.replace(word, ' ')
            context = ' '.join(context.split())  # clean whitespace

            sub_queries = [
                f"{item1} production yield area data",
                f"{item2} production yield area data",
                query,  # also include the original query
            ]
            return sub_queries

    # For "top N crops" or "which crops" type questions, broaden search
    if any(kw in q_lower for kw in ['top', 'which crops', 'all crops', 'most', 'highest', 'lowest', 'best', 'worst', 'ranking']):
        return [
            query,
            q_lower.replace('top ', '').replace('which ', ''),
            "crop production yield area fertilizer data",
        ]

    # Default: just return the original query
    return [query]


def retrieve(query, k=5, score_threshold=0.3):
    """
    Retrieve relevant chunks from ChromaDB for the given query.
    Uses multi-query decomposition for comparative questions.

    Returns an empty list if no relevant data is found (no error raised).
    The caller should handle the empty case in its prompt.
    """
    sub_queries = _decompose_query(query)

    all_chunks = []
    seen_content = set()

    for sub_q in sub_queries:
        try:
            # Use plain similarity search (no threshold filtering) to ensure
            # we always get results. The LLM can judge relevance from context.
            results = db.similarity_search(sub_q, k=k)
            for doc in results:
                content = doc.page_content
                # Deduplicate by first 200 chars (same chunk from different sub-queries)
                content_key = content[:200]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    all_chunks.append(content)
        except Exception as e:
            print(f"Retrieval error for sub-query '{sub_q}': {e}")

    # Cap total chunks to avoid overwhelming the LLM context
    return all_chunks[:12]


# ── Testing (run this file directly to verify retrieval) ──
if __name__ == "__main__":
    test_questions = [
        "What is the yield of rice in Chhattisgarh?",
        "Compare rice and wheat production across states",
        "Which crops use the most fertilizer?",
        "Top 5 crops by area in Maharashtra",
        "What is the soil pH in Andhra Pradesh?",
        "If two states have identical weather, what explains different yields?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print("=" * 60)
        results = retrieve(q)
        if not results:
            print("No chunks found for this query.")
        else:
            print(f"Found {len(results)} chunks.")
        for i, chunk in enumerate(results, 1):
            print(f"\nChunk {i}:\n{chunk[:300]}")
