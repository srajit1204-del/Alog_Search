"""
Day 4 — AlgoSearch Min Heap Ranker
Combines keyword match score + vector similarity score into a single
ranked list using Python's heapq (min heap).

Flow:
  1. User query → extract keywords
  2. Trie finds which keywords exist in our index
  3. Inverted Index returns candidate algorithm IDs
  4. ChromaDB returns vector similarity scores for those candidates
  5. Combine scores → push into Min Heap → pop top-N
"""

import json
import heapq
from sentence_transformers import SentenceTransformer
import chromadb
from dsa import Trie, InvertedIndex


# ═══════════════════════════════════════════════════════
# STEP 1: Load everything we built in Days 1-3
# ═══════════════════════════════════════════════════════

print("Loading data ...")

# Load algorithms
with open("algorithms.json", "r") as f:
    algorithms = json.load(f)

# Load inverted index
with open("inverted_index.json", "r") as f:
    raw_index = json.load(f)

# Rebuild Trie (in-memory, fast to rebuild from algorithms.json)
trie = Trie()
for algo in algorithms:
    for word in algo["name"].lower().split():
        trie.insert(word)
    for kw in algo["keywords"]:
        trie.insert(kw)

# Rebuild Inverted Index from saved JSON
inv_index = InvertedIndex()
inv_index.index = {k: v for k, v in raw_index.items()}

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("algorithms")

print("  ✓ All components loaded")


# ═══════════════════════════════════════════════════════
# STEP 2: Keyword extractor
#
# Splits the user query into words and checks each one
# against the Trie. Only words that exist in the Trie
# are meaningful — the rest are stop words ("the", "a", "in").
# ═══════════════════════════════════════════════════════

def extract_query_keywords(query):
    """
    Extract meaningful DSA keywords from the user's plain-English query.
    Uses Trie prefix search — if a word or its prefix matches, include it.
    """
    words = query.lower().split()
    keywords = []

    for word in words:
        # Exact match first
        if trie.search_exact(word):
            keywords.append(word)
        else:
            # Try prefix match — "dijkst" should still match "dijkstra"
            matches = trie.search_prefix(word)
            if matches:
                keywords.extend(matches)

    return list(set(keywords))  # deduplicate


# ═══════════════════════════════════════════════════════
# STEP 3: Keyword score
#
# For each candidate algorithm, count how many of the
# query keywords appear in its keyword list.
# Normalize by dividing by total query keywords → score between 0 and 1.
#
# Example:
#   query keywords: ["shortest", "path", "graph"]
#   algo keywords:  ["shortest", "graph", "directed", "weight"]
#   overlap: 2 out of 3 → keyword_score = 2/3 = 0.667
# ═══════════════════════════════════════════════════════

def keyword_score(algo_id, query_keywords):
    """Fraction of query keywords that appear in this algorithm's keywords."""
    if not query_keywords:
        return 0.0
    algo_keywords = set(algorithms[algo_id]["keywords"])
    matches = sum(1 for kw in query_keywords if kw in algo_keywords)
    return matches / len(query_keywords)


# ═══════════════════════════════════════════════════════
# STEP 4: MIN HEAP RANKER
#
# A Min Heap always keeps the SMALLEST element at the top.
# We use it to efficiently track the top-N highest scoring algorithms.
#
# Trick: push NEGATIVE score into the heap.
# → Python's heapq is a min heap by default
# → pushing -score means the "smallest" (most negative) = highest actual score
# → this lets us use heapq as a max heap
#
# Why heap over just sorting?
# → If we had 10,000 algorithms, sorting is O(n log n)
# → Heap gives top-N in O(n log k) where k = N (e.g. 3)
# → Much faster when N << total algorithms
#
# Interview answer: "I use a min heap of size k. For each new score,
# if it's better than the current minimum in the heap, I replace it.
# This gives O(n log k) instead of O(n log n) for sorting."
# ═══════════════════════════════════════════════════════

def rank(query, top_n=3):
    """
    Full ranking pipeline for a user query.
    Returns top_n algorithms with their combined scores.
    """

    # ── 2a: Extract keywords from query ──
    query_keywords = extract_query_keywords(query)
    print(f"\n  Keywords found: {query_keywords}")

    # ── 2b: Get candidate IDs from Inverted Index ──
    if query_keywords:
        candidate_ids = inv_index.query(query_keywords)
    else:
        # No keywords matched — search all algorithms
        candidate_ids = list(range(len(algorithms)))

    print(f"  Candidates from Inverted Index: {len(candidate_ids)} algorithms")

    # ── 2c: Get vector similarity scores from ChromaDB ──
    # Only query ChromaDB for the candidate IDs (pre-filtered)
    query_vector = model.encode(query).tolist()

    # ChromaDB where filter: only look at our candidate IDs
    if candidate_ids:
        chroma_results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(len(candidate_ids), top_n * 3),  # get a few extra
            where={"$or": [{"id": str(i)} for i in candidate_ids]} if len(candidate_ids) < len(algorithms) else None,
        )
    else:
        chroma_results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_n * 3,
        )

    # Build a dict: algo_name → vector_similarity_score
    vec_scores = {}
    for meta, dist in zip(chroma_results["metadatas"][0], chroma_results["distances"][0]):
        vec_scores[meta["name"]] = 1 - dist  # convert distance → similarity

    # ── 2d: Combine scores + push to Min Heap ──
    # Combined score = 40% keyword + 60% vector
    # We weight vector higher because semantic meaning matters more than exact words
    KEYWORD_WEIGHT = 0.4
    VECTOR_WEIGHT = 0.6

    heap = []  # min heap

    for algo_id in candidate_ids:
        algo = algorithms[algo_id]
        kw_score = keyword_score(algo_id, query_keywords)
        vec_score = vec_scores.get(algo["name"], 0.0)
        combined = KEYWORD_WEIGHT * kw_score + VECTOR_WEIGHT * vec_score

        # Push (-combined, algo_id) — negative so min heap acts as max heap
        heapq.heappush(heap, (-combined, algo_id))

    # ── 2e: Pop top-N from heap ──
    top_results = []
    for _ in range(min(top_n, len(heap))):
        neg_score, algo_id = heapq.heappop(heap)
        top_results.append({
            "name": algorithms[algo_id]["name"],
            "complexity": algorithms[algo_id]["complexity"],
            "use_case": algorithms[algo_id]["use_case"],
            "url": algorithms[algo_id]["url"],
            "score": round(-neg_score, 3),
        })

    return top_results


# ═══════════════════════════════════════════════════════
# STEP 5: TEST
# ═══════════════════════════════════════════════════════

def test():
    test_queries = [
        "find shortest path in a weighted graph",
        "sort a list of numbers efficiently",
        "detect cycle in a directed graph",
        "find pattern in a string",
        "range sum queries on an array",
    ]

    for query in test_queries:
        print(f"\n{'='*55}")
        print(f"Query: '{query}'")
        results = rank(query, top_n=3)
        print(f"\n  Top 3 results:")
        for i, r in enumerate(results):
            print(f"  {i+1}. {r['name']} (score: {r['score']}) — {r['complexity']}")


if __name__ == "__main__":
    test()
    print("\n✅ Day 4 complete — Min Heap Ranker working")
