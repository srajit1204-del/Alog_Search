"""
Day 1 — AlgoSearch Scraper
Scrapes algorithm pages from CP-Algorithms.com and saves as JSON.
Each algorithm is stored with: name, description, complexity, use_case, keywords, url
"""

import requests
from bs4 import BeautifulSoup
import json
import time

# ─────────────────────────────────────────────
# STEP 1: Define the algorithm pages to scrape
# We manually list URLs because CP-Algorithms has no sitemap API.
# These cover the core DSA topics you'll be asked about in interviews.
# ─────────────────────────────────────────────

ALGORITHM_URLS = [
    # Sorting
    ("Merge Sort", "https://cp-algorithms.com/sequences/merge-sort.html"),

    # Graph algorithms
    ("Breadth First Search", "https://cp-algorithms.com/graph/breadth-first-search.html"),
    ("Depth First Search", "https://cp-algorithms.com/graph/depth-first-search.html"),
    ("Dijkstra's Algorithm", "https://cp-algorithms.com/graph/dijkstra.html"),
    ("Bellman-Ford Algorithm", "https://cp-algorithms.com/graph/bellman_ford.html"),
    ("Floyd-Warshall Algorithm", "https://cp-algorithms.com/graph/all-pair-shortest-path-floyd-warshall.html"),
    ("Kruskal's Algorithm", "https://cp-algorithms.com/graph/mst_kruskal.html"),
    ("Prim's Algorithm", "https://cp-algorithms.com/graph/mst_prim.html"),
    ("Topological Sort", "https://cp-algorithms.com/graph/topological-sort.html"),
    ("Strongly Connected Components", "https://cp-algorithms.com/graph/strongly-connected-components.html"),
    ("Bridges in Graph", "https://cp-algorithms.com/graph/bridge-searching.html"),
    ("Articulation Points", "https://cp-algorithms.com/graph/cutpoints.html"),
    ("A* Search", "https://cp-algorithms.com/graph/astar.html"),

    # Dynamic Programming
    ("Longest Common Subsequence", "https://cp-algorithms.com/sequences/longest-common-subsequence.html"),
    ("Longest Increasing Subsequence", "https://cp-algorithms.com/sequences/longest_increasing_subsequence.html"),
    ("0-1 Knapsack Problem", "https://cp-algorithms.com/dynamic_programming/knapsack.html"),
    ("Matrix Chain Multiplication", "https://cp-algorithms.com/dynamic_programming/matrix-chain-multiplication.html"),

    # Trees
    ("Binary Search Tree", "https://cp-algorithms.com/data_structures/treap.html"),
    ("Segment Tree", "https://cp-algorithms.com/data_structures/segment_tree.html"),
    ("Fenwick Tree (BIT)", "https://cp-algorithms.com/data_structures/fenwick.html"),
    ("Lowest Common Ancestor", "https://cp-algorithms.com/graph/lca.html"),
    ("Sparse Table", "https://cp-algorithms.com/data_structures/sparse-table.html"),

    # Searching
    ("Binary Search", "https://cp-algorithms.com/num_methods/binary_search.html"),
    ("Ternary Search", "https://cp-algorithms.com/num_methods/ternary_search.html"),

    # Strings
    ("KMP Algorithm", "https://cp-algorithms.com/string/kmp.html"),
    ("Z Algorithm", "https://cp-algorithms.com/string/z-function.html"),
    ("Rabin-Karp Algorithm", "https://cp-algorithms.com/string/rabin-karp.html"),
    ("Suffix Array", "https://cp-algorithms.com/string/suffix-array.html"),
    ("Trie", "https://cp-algorithms.com/string/aho_corasick.html"),

    # Math
    ("Sieve of Eratosthenes", "https://cp-algorithms.com/algebra/sieve-of-eratosthenes.html"),
    ("Euclidean Algorithm (GCD)", "https://cp-algorithms.com/algebra/euclid-algorithm.html"),
    ("Fast Exponentiation", "https://cp-algorithms.com/algebra/binary-exp.html"),

    # Union-Find
    ("Union Find (DSU)", "https://cp-algorithms.com/data_structures/disjoint_set_union.html"),
]


# ─────────────────────────────────────────────
# STEP 2: Extract keywords from text
# We pull important DSA-related words from the page text.
# These go into the Inverted Index later (Day 3).
# ─────────────────────────────────────────────

DSA_KEYWORDS = [
    "graph", "tree", "array", "string", "path", "shortest", "sort", "search",
    "dynamic", "greedy", "divide", "conquer", "backtrack", "recursion",
    "cycle", "connected", "component", "vertex", "edge", "weight", "directed",
    "undirected", "matrix", "subsequence", "substring", "prefix", "suffix",
    "binary", "heap", "stack", "queue", "hash", "dp", "memoization",
    "topological", "spanning", "minimum", "maximum", "optimal", "linear",
    "logarithmic", "polynomial", "flow", "matching", "bipartite", "prime",
    "modular", "segment", "fenwick", "trie", "union", "find", "disjoint",
]

def extract_keywords(text):
    """Pull DSA keywords that appear in the page text."""
    text_lower = text.lower()
    found = [kw for kw in DSA_KEYWORDS if kw in text_lower]
    return list(set(found))  # deduplicate


# ─────────────────────────────────────────────
# STEP 3: Scrape a single algorithm page
# We grab: first paragraph (description), complexity mentions, use cases
# ─────────────────────────────────────────────

def scrape_algorithm(name, url):
    """Scrape one algorithm page and return a structured dict."""
    print(f"Scraping: {name} ...")

    try:
        headers = {"User-Agent": "Mozilla/5.0 (AlgoSearch academic scraper)"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            print(f"  ✗ Failed ({response.status_code})")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Get main content area
        # CP-Algorithms uses <article> or <div class="content"> for the main text
        content = soup.find("article") or soup.find("div", class_="content") or soup.body

        if not content:
            print(f"  ✗ No content found")
            return None

        # Extract all paragraph text
        paragraphs = content.find_all("p")
        all_text = " ".join(p.get_text(separator=" ") for p in paragraphs)

        # Description = first meaningful paragraph (>50 chars)
        description = ""
        for p in paragraphs:
            text = p.get_text(separator=" ").strip()
            if len(text) > 50:
                description = text[:500]  # cap at 500 chars
                break

        # Look for complexity mentions (O(...) patterns)
        import re
        complexity_pattern = r'O\([^)]+\)'
        complexities = re.findall(complexity_pattern, all_text)
        complexity = complexities[0] if complexities else "See page"

        # Use case = second meaningful paragraph or a sentence with "when" / "used"
        use_case = ""
        for p in paragraphs[1:4]:  # check next few paragraphs
            text = p.get_text(separator=" ").strip()
            if len(text) > 50 and any(w in text.lower() for w in ["when", "used", "useful", "apply", "suitable"]):
                use_case = text[:300]
                break
        if not use_case and len(paragraphs) > 1:
            use_case = paragraphs[1].get_text(separator=" ").strip()[:300]

        # Extract keywords from full page text
        keywords = extract_keywords(all_text)
        # Always include the algorithm name words as keywords
        keywords += [w.lower() for w in name.split() if len(w) > 3]
        keywords = list(set(keywords))

        print(f"  ✓ Done — {len(keywords)} keywords, complexity: {complexity}")

        return {
            "name": name,
            "description": description,
            "complexity": complexity,
            "use_case": use_case,
            "keywords": keywords,
            "url": url,
        }

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


# ─────────────────────────────────────────────
# STEP 4: Run scraper on all URLs and save JSON
# We add a small delay between requests to be polite to the server.
# ─────────────────────────────────────────────

def main():
    algorithms = []

    for name, url in ALGORITHM_URLS:
        data = scrape_algorithm(name, url)
        if data:
            algorithms.append(data)
        time.sleep(1)  # 1 second delay between requests — polite scraping

    # Save to JSON
    output_path = "algorithms.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(algorithms, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Scraped {len(algorithms)} algorithms → saved to {output_path}")


if __name__ == "__main__":
    main()
