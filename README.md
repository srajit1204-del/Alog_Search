# AlgoSearch — Algorithm Finder Engine

> Describe your problem in plain English. Get the right algorithm.

Most of the time when you're solving a problem, you roughly know what you need — something about shortest paths, or sorting, or finding patterns in a string. But you don't always remember the exact algorithm name or which one fits your situation best. AlgoSearch is built for that gap.

---

## Demo

Type: *"find shortest path between two cities avoiding toll roads"*

Get back:
- **Dijkstra's Algorithm**
- Time: O((V+E) log V) — Space: O(V+E)
- Explanation of how it works
- When to use it vs Bellman-Ford vs A*

---

## Why this project exists

Almost every RAG project out there uses it on documents, PDFs, or knowledge bases. This one uses it on DSA — which means every design decision (why a Trie, why a heap, why vector search) has a clean technical reason behind it, not just "because that's what the tutorial said."

---

## How it works

The search pipeline has 5 stages:

**Stage 1 — Trie (keyword matching)**
Your query gets split into words. Each word is looked up in a custom-built Trie. This runs in O(k) where k is the length of the word — much faster than scanning all algorithm names linearly. Words like "shortest", "path", "sort" get matched here.

**Stage 2 — Inverted Index (pre-filtering)**
A Python dict maps each keyword to a list of algorithm IDs. So "shortest" maps to [Dijkstra, Bellman-Ford, BFS]. This cuts down the candidates before vector search runs — no point searching all 28 algorithms if only 8 are relevant.

**Stage 3 — ChromaDB vector search (semantic matching)**
The query gets converted to a 384-dimensional vector using sentence-transformers. ChromaDB compares it against pre-stored vectors of algorithm descriptions using cosine similarity. This catches semantic matches — even if the user didn't use the exact keyword.

**Stage 4 — Min Heap ranker (scoring)**
Each candidate gets two scores: keyword match score and vector similarity score. These are combined (40% keyword, 60% vector) and pushed into a min heap. The top 3 results are popped in O(n log k) time.

**Stage 5 — Groq LLM (explanation)**
The top result is sent to Groq's Llama 3.3 70B with a structured RAG prompt. The LLM generates: algorithm name, explanation, time complexity, space complexity, when to use it, and alternatives. Because it's grounded in the retrieved algorithm data, it doesn't hallucinate.

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Data Collection | BeautifulSoup, requests | Scrape CP-Algorithms.com |
| DSA — Trie | Pure Python | O(k) keyword prefix lookup |
| DSA — Inverted Index | Pure Python dict | Pre-filter by keyword overlap |
| DSA — Min Heap | Python heapq | Rank results by combined score |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Convert text to vectors locally |
| Vector DB | ChromaDB | Store and retrieve embeddings |
| LLM | Groq API (Llama 3.3 70B) | Generate structured explanation |
| Backend | FastAPI + Uvicorn | REST API connecting all components |
| Frontend | HTML, CSS, JavaScript | Clean search UI |
| Version Control | Git + GitHub | — |

---

## Project Structure

```
algosearch/
├── scraper.py            # scrapes 28+ algorithm pages from CP-Algorithms.com
├── embed.py              # generates embeddings and stores them in ChromaDB
├── dsa.py                # Trie + Inverted Index built from scratch
├── ranker.py             # Min Heap ranker combining keyword + vector scores
├── llm.py                # Groq API integration with RAG prompt
├── main.py               # FastAPI backend exposing /search endpoint
├── index.html            # frontend (HTML/CSS/JS)
├── algorithms.json       # scraped algorithm data (28 algorithms)
├── inverted_index.json   # keyword → algorithm ID mapping
├── requirements.txt      # Python dependencies
└── .gitignore
```

---

## Setup and Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/srajit1204-del/Alog_Search.git
cd Alog_Search
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the scraper** (optional — algorithms.json already included)
```bash
python3 scraper.py
```

**4. Generate embeddings and store in ChromaDB**
```bash
python3 embed.py
```

**5. Build Trie + Inverted Index**
```bash
python3 dsa.py
```

**6. Start the backend**
```bash
export GROQ_API_KEY="your_groq_api_key_here"
uvicorn main:app --reload
```

**7. Open the frontend**

Open `index.html` in your browser. The backend runs on `localhost:8000`.

---

## API

**POST /search**

Request:
```json
{
  "query": "find shortest path in a weighted graph",
  "top_n": 3
}
```

Response:
```json
{
  "query": "find shortest path in a weighted graph",
  "top_result": {
    "name": "Dijkstra's Algorithm",
    "complexity": "O((V+E) log V)",
    "score": 0.623,
    "url": "https://cp-algorithms.com/graph/dijkstra.html"
  },
  "other_results": [...],
  "explanation": {
    "algorithm": "Dijkstra's Algorithm",
    "explanation": "...",
    "time_complexity": "...",
    "space_complexity": "...",
    "when_to_use": "...",
    "alternatives": "..."
  }
}
```

**GET /**

Health check — returns `{"status": "ok"}`.

---

## Algorithms Indexed

Graphs: BFS, DFS, Dijkstra, Bellman-Ford, Floyd-Warshall, Kruskal, Prim, Topological Sort, SCC, Bridges, Articulation Points

Dynamic Programming: LIS, 0-1 Knapsack, Matrix Chain Multiplication

Trees: Segment Tree, Fenwick Tree (BIT), LCA, Sparse Table, Binary Search Tree

Strings: KMP, Z Algorithm, Rabin-Karp, Suffix Array, Trie

Searching: Binary Search, Ternary Search

Math: Sieve of Eratosthenes, Euclidean Algorithm (GCD), Fast Exponentiation

Other: Union Find (DSU)

---

## Design Decisions

**Why a Trie over simple string matching?**
O(k) prefix lookup regardless of how many keywords exist. String scanning would be O(n × k) — slower as the index grows.

**Why not just use vector search alone?**
Pure vector search misses exact keyword intent. If you type "Dijkstra", vector search might return vaguely similar things. The Trie + Inverted Index hybrid ensures keyword precision first, then vector search adds semantic understanding.

**Why Groq over OpenAI?**
Free tier with high rate limits. Same RAG pipeline, different provider. Easy to swap if needed.

**Why 40/60 keyword/vector split?**
Semantic meaning matters more than exact word overlap for algorithm retrieval. But keyword matches deserve weight too. if someone types "heap", heap-based algorithms should rank higher regardless of semantic distance.
