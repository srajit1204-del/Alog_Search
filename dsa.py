"""
Day 3 — AlgoSearch Trie + Inverted Index
Both data structures built from scratch in pure Python.

Trie         → fast keyword prefix matching O(k) where k = query length
Inverted Index → maps keywords to algorithm IDs for pre-filtering
"""

import json


# ═══════════════════════════════════════════════════════
# PART 1: TRIE
#
# A Trie is a tree where each node represents one character.
# Inserting "dijkstra" creates nodes: d → i → j → k → s → t → r → a
# Searching "dijk" walks those 4 nodes in O(4) — O(k) time.
#
# Why not just use "if keyword in string"?
# → That's O(n * k) — scan every keyword for every character.
# → Trie does it in O(k) regardless of how many keywords exist.
#
# Structure of each node:
#   children: dict mapping char → TrieNode
#   is_end:   True if this node ends a complete word
#   word:     the full word stored at this end node
# ═══════════════════════════════════════════════════════

class TrieNode:
    def __init__(self):
        self.children = {}   # char → TrieNode
        self.is_end = False  # does a word end here?
        self.word = None     # store the full word at end nodes


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """
        Insert a word into the Trie character by character.
        Time: O(k) where k = len(word)
        """
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()  # create node if missing
            node = node.children[char]             # move down
        node.is_end = True
        node.word = word  # store full word at the end

    def search_prefix(self, prefix):
        """
        Given a prefix, return all words in the Trie that start with it.
        Example: search_prefix("dijk") → ["dijkstra"]

        Step 1: walk down the Trie following the prefix characters
        Step 2: from that node, collect ALL words in the subtree (DFS)

        Time: O(k + m) where k = prefix length, m = number of matches
        """
        node = self.root

        # Walk down to the end of the prefix
        for char in prefix.lower():
            if char not in node.children:
                return []  # prefix not found at all
            node = node.children[char]

        # Now collect all words in this subtree using DFS
        results = []
        self._collect_words(node, results)
        return results

    def _collect_words(self, node, results):
        """DFS helper — collects all complete words from a subtree."""
        if node.is_end:
            results.append(node.word)
        for child in node.children.values():
            self._collect_words(child, results)

    def search_exact(self, word):
        """Check if an exact word exists in the Trie. O(k)"""
        node = self.root
        for char in word.lower():
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end


# ═══════════════════════════════════════════════════════
# PART 2: INVERTED INDEX
#
# A simple Python dict: keyword → list of algorithm IDs
#
# Example after building:
#   {
#     "shortest": [0, 2, 4],   ← Dijkstra(0), Bellman-Ford(2), BFS(4)
#     "graph":    [0, 1, 2, 3, 4, 5],
#     "sort":     [7, 8],
#   }
#
# When user queries "find shortest path":
# → keywords found: ["shortest", "path"]
# → IDs from "shortest": {0, 2, 4}
# → IDs from "path":     {0, 2, 3}
# → Intersection:        {0, 2}  ← only these go to vector search
#
# This cuts the vector search space from 28 → maybe 3-5 algorithms.
# ═══════════════════════════════════════════════════════

class InvertedIndex:
    def __init__(self):
        self.index = {}  # keyword → list of algo IDs

    def build(self, algorithms):
        """
        Build the index from the algorithms list.
        For each algorithm, add its ID to every keyword's list.
        Time: O(total keywords across all algorithms)
        """
        for algo_id, algo in enumerate(algorithms):
            for keyword in algo["keywords"]:
                keyword = keyword.lower()
                if keyword not in self.index:
                    self.index[keyword] = []
                self.index[keyword].append(algo_id)

        print(f"  ✓ Inverted Index built — {len(self.index)} unique keywords")

    def lookup(self, keyword):
        """Return list of algo IDs that contain this keyword. O(1)"""
        return self.index.get(keyword.lower(), [])

    def query(self, keywords):
        """
        Given a list of keywords, return algo IDs that match ANY of them.
        We use union (not intersection) to be more permissive —
        then the heap ranker will score by how many keywords matched.
        """
        matched_ids = set()
        for kw in keywords:
            matched_ids.update(self.lookup(kw))
        return list(matched_ids)


# ═══════════════════════════════════════════════════════
# PART 3: BUILD + SAVE
# Load algorithms.json, build both structures, run tests.
# We also save the index to a JSON file so other modules can load it.
# ═══════════════════════════════════════════════════════

def build_and_save(algorithms_path="algorithms.json"):
    print("Loading algorithms.json ...")
    with open(algorithms_path, "r", encoding="utf-8") as f:
        algorithms = json.load(f)
    print(f"  ✓ Loaded {len(algorithms)} algorithms")

    # ── Build Trie ──
    print("\nBuilding Trie ...")
    trie = Trie()

    for algo in algorithms:
        # Insert algorithm name words
        for word in algo["name"].lower().split():
            trie.insert(word)
        # Insert keywords
        for kw in algo["keywords"]:
            trie.insert(kw)

    print("  ✓ Trie built")

    # ── Build Inverted Index ──
    print("\nBuilding Inverted Index ...")
    inv_index = InvertedIndex()
    inv_index.build(algorithms)

    # Save inverted index to JSON (Trie is in-memory only — rebuilt at startup)
    with open("inverted_index.json", "w") as f:
        json.dump(inv_index.index, f, indent=2)
    print("  ✓ Inverted Index saved to inverted_index.json")

    return trie, inv_index, algorithms


# ═══════════════════════════════════════════════════════
# PART 4: TEST
# ═══════════════════════════════════════════════════════

def test(trie, inv_index, algorithms):
    print("\n--- Trie Tests ---")

    # Test 1: prefix search
    results = trie.search_prefix("dijk")
    print(f"search_prefix('dijk') → {results}")

    results = trie.search_prefix("sort")
    print(f"search_prefix('sort') → {results}")

    results = trie.search_prefix("bin")
    print(f"search_prefix('bin')  → {results}")

    # Test 2: exact search
    print(f"search_exact('graph') → {trie.search_exact('graph')}")
    print(f"search_exact('xyz')   → {trie.search_exact('xyz')}")

    print("\n--- Inverted Index Tests ---")

    # Test 3: single keyword lookup
    ids = inv_index.lookup("shortest")
    names = [algorithms[i]["name"] for i in ids]
    print(f"lookup('shortest') → {names}")

    ids = inv_index.lookup("graph")
    names = [algorithms[i]["name"] for i in ids]
    print(f"lookup('graph') → {names}")

    # Test 4: multi-keyword query
    ids = inv_index.query(["shortest", "path", "graph"])
    names = [algorithms[i]["name"] for i in ids]
    print(f"query(['shortest','path','graph']) → {names}")


if __name__ == "__main__":
    trie, inv_index, algorithms = build_and_save()
    test(trie, inv_index, algorithms)
    print("\n✅ Day 3 complete — Trie + Inverted Index working")
