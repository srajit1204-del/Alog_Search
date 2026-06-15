"""
Day 2 — AlgoSearch Embeddings + ChromaDB
Loads algorithms.json, converts each description to a vector using
sentence-transformers, and stores everything in ChromaDB for vector search.
"""

import json
from sentence_transformers import SentenceTransformer
import chromadb

# ─────────────────────────────────────────────
# STEP 1: Load the scraped algorithm data
# This is the algorithms.json we created in Day 1.
# ─────────────────────────────────────────────

print("Loading algorithms.json ...")
with open("algorithms.json", "r", encoding="utf-8") as f:
    algorithms = json.load(f)

print(f"  ✓ Loaded {len(algorithms)} algorithms")


# ─────────────────────────────────────────────
# STEP 2: Load the embedding model
#
# We use "all-MiniLM-L6-v2" — a small but powerful model that:
# - Runs fully locally (no API needed)
# - Converts text → 384-dimensional vector
# - Is fast enough for our use case
#
# Interview Q: "Why this model?"
# A: It's the best free local embedding model for semantic similarity.
#    384 dimensions is enough to capture meaning without being too slow.
# ─────────────────────────────────────────────

print("\nLoading sentence-transformers model (all-MiniLM-L6-v2) ...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("  ✓ Model loaded")


# ─────────────────────────────────────────────
# STEP 3: Set up ChromaDB (local, persistent)
#
# ChromaDB stores vectors on disk in a folder called "chroma_db".
# "Persistent" means the data survives after you close the program.
# We create a "collection" — think of it like a table in a database,
# but instead of rows, it stores vectors + metadata.
#
# Interview Q: "Why ChromaDB over FAISS or Pinecone?"
# A: ChromaDB is free, local, persistent, and has a simple Python API.
#    FAISS is faster but has no built-in metadata storage.
#    Pinecone requires an account and has rate limits.
# ─────────────────────────────────────────────

print("\nSetting up ChromaDB ...")
client = chromadb.PersistentClient(path="./chroma_db")

# Delete existing collection if re-running (so we don't get duplicates)
try:
    client.delete_collection("algorithms")
    print("  ℹ Deleted old collection")
except:
    pass

collection = client.create_collection(
    name="algorithms",
    metadata={"hnsw:space": "cosine"}  # use cosine similarity for text
)
print("  ✓ Collection created")


# ─────────────────────────────────────────────
# STEP 4: Generate embeddings and store in ChromaDB
#
# For each algorithm, we build a "rich text" combining name + description
# + use_case. This gives the embedding more context than just description alone.
#
# Then we call model.encode() which returns a list of 384 numbers.
# That list IS the vector — it represents the semantic meaning of the text.
#
# ChromaDB stores: the vector, the original text, and metadata (name, url, etc.)
# ─────────────────────────────────────────────

print("\nGenerating embeddings and storing in ChromaDB ...")

ids = []
embeddings = []
documents = []
metadatas = []

for i, algo in enumerate(algorithms):
    # Build rich text for embedding — more context = better vector
    rich_text = f"{algo['name']}. {algo['description']} Use case: {algo['use_case']}"

    # Generate embedding (384-dim vector)
    embedding = model.encode(rich_text).tolist()  # .tolist() converts numpy → plain list

    ids.append(str(i))
    embeddings.append(embedding)
    documents.append(rich_text)
    metadatas.append({
        "name": algo["name"],
        "complexity": algo["complexity"],
        "keywords": ", ".join(algo["keywords"]),  # ChromaDB metadata must be strings
        "url": algo["url"],
        "use_case": algo["use_case"][:200],  # trim to avoid metadata size limit
    })

    print(f"  [{i+1}/{len(algorithms)}] Embedded: {algo['name']}")

# Add all to ChromaDB in one batch (faster than adding one by one)
collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas,
)

print(f"\n✅ Done! {len(algorithms)} algorithms embedded and stored in ChromaDB (./chroma_db)")


# ─────────────────────────────────────────────
# STEP 5: Quick test — verify vector search works
#
# We query ChromaDB with a plain English sentence and see if it
# returns the right algorithm. This proves the pipeline works before Day 3.
# ─────────────────────────────────────────────

print("\n--- Quick Test ---")
test_query = "find shortest path in a weighted graph"
print(f"Query: '{test_query}'")

query_vector = model.encode(test_query).tolist()

results = collection.query(
    query_embeddings=[query_vector],
    n_results=3,  # return top 3
)

print("Top 3 results:")
for i, (name_meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
    print(f"  {i+1}. {name_meta['name']} (similarity score: {1 - dist:.3f})")
