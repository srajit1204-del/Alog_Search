"""
Day 6 — AlgoSearch FastAPI Backend
Exposes the full pipeline as a REST API.

Endpoints:
  GET  /          → health check
  POST /search    → main search endpoint
"""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from dsa import Trie, InvertedIndex
from ranker import rank
from llm import explain

# ─────────────────────────────────────────────
# APP SETUP
# CORSMiddleware lets the HTML frontend (on a different port)
# talk to this backend. Without it, the browser blocks the request.
# ─────────────────────────────────────────────

app = FastAPI(title="AlgoSearch API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (fine for dev + Streamlit Cloud)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# LOAD COMPONENTS AT STARTUP
# These are heavy (model, chromadb) so we load them once
# when the server starts, not on every request.
# ─────────────────────────────────────────────

print("Loading AlgoSearch components...")

with open("algorithms.json", "r") as f:
    algorithms = json.load(f)

with open("inverted_index.json", "r") as f:
    raw_index = json.load(f)

trie = Trie()
for algo in algorithms:
    for word in algo["name"].lower().split():
        trie.insert(word)
    for kw in algo["keywords"]:
        trie.insert(kw)

inv_index = InvertedIndex()
inv_index.index = {k: v for k, v in raw_index.items()}

model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("algorithms")

print("✓ All components loaded — server ready")


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# Pydantic models validate the incoming JSON automatically.
# If the request is missing "query", FastAPI returns a 422 error.
# ─────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_n: int = 3  # optional, defaults to 3


class AlgorithmResult(BaseModel):
    name: str
    complexity: str
    score: float
    url: str


class SearchResponse(BaseModel):
    query: str
    top_result: dict
    other_results: list
    explanation: dict


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def health_check():
    """Health check — confirms the server is running."""
    return {"status": "ok", "message": "AlgoSearch API is running"}


@app.post("/search")
def search(request: SearchRequest):
    """
    Main search endpoint.
    Takes a plain-English query, runs the full pipeline,
    returns ranked results + LLM explanation.
    """
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Step 1: Rank
    top_results = rank(query, top_n=request.top_n)

    if not top_results:
        raise HTTPException(status_code=404, detail="No algorithms found for this query")

    # Step 2: Get full algo data for top result
    top_algo_name = top_results[0]["name"]
    top_algo = next(
        (a for a in algorithms if a["name"] == top_algo_name),
        top_results[0]
    )

    # Step 3: LLM explanation
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set on server")

    parsed_explanation, _ = explain(query, top_algo)

    return {
        "query": query,
        "top_result": {
            "name": top_results[0]["name"],
            "complexity": top_results[0]["complexity"],
            "score": top_results[0]["score"],
            "url": top_algo.get("url", ""),
        },
        "other_results": top_results[1:],
        "explanation": parsed_explanation,
    }
