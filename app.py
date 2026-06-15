"""
Day 6 — AlgoSearch Streamlit Frontend
Wires together the full pipeline into a clean web UI:
  User query → Ranker → LLM → Display result card
"""

import os
import streamlit as st
import json
from sentence_transformers import SentenceTransformer
import chromadb
from dsa import Trie, InvertedIndex
from ranker import rank
from llm import explain

# ─────────────────────────────────────────────
# PAGE CONFIG
# Must be the first Streamlit call in the file.
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AlgoSearch",
    page_icon="🔍",
    layout="centered",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — clean, minimal styling
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .main { max-width: 750px; }
    .result-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
    }
    .algo-name {
        font-size: 1.6rem;
        font-weight: 700;
        color: #cba6f7;
        margin-bottom: 4px;
    }
    .complexity-badge {
        display: inline-block;
        background: #313244;
        color: #a6e3a1;
        font-family: monospace;
        font-size: 0.85rem;
        padding: 3px 10px;
        border-radius: 6px;
        margin-bottom: 16px;
    }
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6c7086;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 16px;
        margin-bottom: 4px;
    }
    .section-text {
        color: #cdd6f4;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .alt-tag {
        display: inline-block;
        background: #181825;
        border: 1px solid #45475a;
        color: #89b4fa;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 3px 3px 3px 0;
    }
    .score-pill {
        font-size: 0.75rem;
        color: #6c7086;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD ALL COMPONENTS (cached so they only load once)
#
# @st.cache_resource tells Streamlit: load this once,
# reuse it across all user interactions.
# Without caching, the model would reload on every keypress — very slow.
# ─────────────────────────────────────────────

@st.cache_resource
def load_components():
    """Load all pipeline components once at startup."""
    # Algorithms data
    with open("algorithms.json", "r") as f:
        algorithms = json.load(f)

    # Inverted index
    with open("inverted_index.json", "r") as f:
        raw_index = json.load(f)

    # Rebuild Trie
    trie = Trie()
    for algo in algorithms:
        for word in algo["name"].lower().split():
            trie.insert(word)
        for kw in algo["keywords"]:
            trie.insert(kw)

    # Rebuild Inverted Index
    inv_index = InvertedIndex()
    inv_index.index = {k: v for k, v in raw_index.items()}

    # Embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection("algorithms")

    return algorithms, trie, inv_index, model, collection


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("# 🔍 AlgoSearch")
st.markdown("*Describe your problem in plain English — get the right algorithm.*")
st.divider()

# ─────────────────────────────────────────────
# LOAD COMPONENTS
# ─────────────────────────────────────────────

with st.spinner("Loading AlgoSearch engine..."):
    algorithms, trie, inv_index, model, collection = load_components()

# ─────────────────────────────────────────────
# SEARCH INPUT
# ─────────────────────────────────────────────

query = st.text_input(
    label="Describe your problem:",
    placeholder="e.g. find shortest path in a weighted graph",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 5])
with col1:
    search_btn = st.button("Search", type="primary", use_container_width=True)

# Example queries
st.markdown(
    "<div style='color:#6c7086; font-size:0.8rem; margin-top:6px'>Try: "
    "<em>shortest path between cities</em> · "
    "<em>sort a large list efficiently</em> · "
    "<em>find pattern in a string</em> · "
    "<em>range sum queries on array</em></div>",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# SEARCH PIPELINE
# ─────────────────────────────────────────────

if search_btn and query.strip():
    with st.spinner("Finding best algorithm..."):

        # Step 1: Rank
        top_results = rank(query, top_n=3)

        if not top_results:
            st.warning("No algorithms found. Try rephrasing your query.")
        else:
            # Step 2: Get top result for LLM explanation
            top_algo_name = top_results[0]["name"]

            # Find full algorithm data from algorithms.json
            top_algo = next(
                (a for a in algorithms if a["name"] == top_algo_name),
                top_results[0]
            )

    with st.spinner("Generating explanation..."):
        # Step 3: LLM explanation
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key:
            st.error("GROQ_API_KEY not set. Run: export GROQ_API_KEY='your_key'")
            st.stop()

        parsed, _ = explain(query, top_algo)

    # ─────────────────────────────────────────
    # DISPLAY RESULT CARD
    # ─────────────────────────────────────────

    st.markdown("### Best Match")

    st.markdown(f"""
    <div class="result-card">
        <div class="algo-name">{parsed['algorithm'] or top_algo_name}</div>
        <div class="complexity-badge">⏱ {top_results[0]['complexity']}</div>

        <div class="section-label">How it works</div>
        <div class="section-text">{parsed['explanation']}</div>

        <div class="section-label">Time Complexity</div>
        <div class="section-text">{parsed['time_complexity']}</div>

        <div class="section-label">Space Complexity</div>
        <div class="section-text">{parsed['space_complexity']}</div>

        <div class="section-label">When to use</div>
        <div class="section-text">{parsed['when_to_use']}</div>

        <div class="section-label">Alternatives</div>
        <div class="section-text">{parsed['alternatives']}</div>

        <div class="score-pill">Relevance score: {top_results[0]['score']} ·
        <a href="{top_algo.get('url','')}" target="_blank" style="color:#89b4fa">Read full article →</a></div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # OTHER TOP RESULTS
    # ─────────────────────────────────────────

    if len(top_results) > 1:
        st.markdown("### Also Consider")
        cols = st.columns(len(top_results) - 1)
        for i, result in enumerate(top_results[1:]):
            with cols[i]:
                st.markdown(f"""
                <div style="background:#1e1e2e; border:1px solid #313244;
                            border-radius:8px; padding:14px;">
                    <div style="font-weight:600; color:#cba6f7; margin-bottom:6px">
                        {result['name']}
                    </div>
                    <div style="font-family:monospace; font-size:0.8rem; color:#a6e3a1">
                        {result['complexity']}
                    </div>
                    <div style="font-size:0.78rem; color:#6c7086; margin-top:6px">
                        score: {result['score']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

elif search_btn and not query.strip():
    st.warning("Please enter a problem description.")
