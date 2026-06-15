"""
Day 5 — AlgoSearch Groq LLM Integration
Takes the top-ranked algorithm from the ranker and sends it to
Groq (Llama 3.1 70B) to generate a structured explanation.

RAG pattern:
  Retrieve (ranker.py) → Augment (add algo context to prompt) → Generate (Groq LLM)
"""

import os
from groq import Groq

# ─────────────────────────────────────────────
# STEP 1: Set up Groq client
#
# We read the API key from an environment variable.
# NEVER hardcode the key in the source file — it would get
# exposed if you push to GitHub.
#
# How to set it (run this in terminal before running the script):
#   export GROQ_API_KEY="your_key_here"
#
# Or we can set it directly in code for now (just don't commit it).
# ─────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_KEY_HERE")
client = Groq(api_key=GROQ_API_KEY)


# ─────────────────────────────────────────────
# STEP 2: The prompt template
#
# This is the "Augment" step in RAG.
# We take the retrieved algorithm data and inject it into a
# structured prompt so the LLM has real context to work from.
#
# Without RAG: LLM answers from training data (may hallucinate)
# With RAG:    LLM answers from OUR retrieved algorithm data (grounded)
#
# Interview Q: "What is RAG?"
# A: Retrieval-Augmented Generation. Instead of asking an LLM to
#    answer from memory, you first retrieve relevant documents,
#    then inject them into the prompt as context. The LLM generates
#    from that context — reducing hallucination and improving accuracy.
# ─────────────────────────────────────────────

def build_prompt(query, algorithm):
    """
    Build the RAG prompt by injecting retrieved algorithm data.
    The LLM is instructed to respond in a strict format so we can
    parse the output reliably in the frontend.
    """
    return f"""You are an expert algorithms tutor. A student described this problem:
"{query}"

The most relevant algorithm retrieved is:
- Name: {algorithm['name']}
- Description: {algorithm['description'][:400]}
- Complexity: {algorithm['complexity']}
- Use case: {algorithm['use_case'][:300]}

Based on this, provide a clear explanation in EXACTLY this format:

ALGORITHM: [algorithm name]

EXPLANATION: [2-3 sentences explaining how it works in simple terms]

TIME COMPLEXITY: [time complexity with brief reason]

SPACE COMPLEXITY: [space complexity with brief reason]

WHEN TO USE: [2-3 specific conditions when this algorithm is the right choice]

ALTERNATIVES: [2 alternative algorithms and one-line reason when to prefer them instead]

Keep each section concise. Use plain English — no latex, no markdown formatting."""


# ─────────────────────────────────────────────
# STEP 3: Call Groq API
#
# We use the chat completions endpoint with llama-3.1-70b-versatile.
# Parameters:
#   temperature=0.3 → low randomness, more factual/consistent answers
#   max_tokens=600  → enough for a complete explanation, not too long
# ─────────────────────────────────────────────

def generate_explanation(query, algorithm):
    """
    Send the query + algorithm context to Groq and return structured explanation.
    """
    prompt = build_prompt(query, algorithm)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a concise, accurate algorithms tutor. Always follow the exact output format requested."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=600,
    )

    return response.choices[0].message.content


# ─────────────────────────────────────────────
# STEP 4: Parse the LLM output into a dict
#
# The frontend (Day 6) needs individual fields, not a raw string.
# We split by the section headers and extract each part.
# ─────────────────────────────────────────────

def parse_explanation(text):
    """Parse the structured LLM output into a clean dict."""
    sections = {
        "algorithm": "",
        "explanation": "",
        "time_complexity": "",
        "space_complexity": "",
        "when_to_use": "",
        "alternatives": "",
    }

    # Map of output headers → dict keys
    markers = {
        "ALGORITHM:": "algorithm",
        "EXPLANATION:": "explanation",
        "TIME COMPLEXITY:": "time_complexity",
        "SPACE COMPLEXITY:": "space_complexity",
        "WHEN TO USE:": "when_to_use",
        "ALTERNATIVES:": "alternatives",
    }

    current_key = None
    current_lines = []

    for line in text.split("\n"):
        line = line.strip()
        matched = False
        for marker, key in markers.items():
            if line.startswith(marker):
                # Save previous section
                if current_key:
                    sections[current_key] = " ".join(current_lines).strip()
                current_key = key
                current_lines = [line[len(marker):].strip()]
                matched = True
                break
        if not matched and current_key:
            current_lines.append(line)

    # Save last section
    if current_key:
        sections[current_key] = " ".join(current_lines).strip()

    return sections


# ─────────────────────────────────────────────
# STEP 5: Full pipeline function
# This is what the frontend will call.
# ─────────────────────────────────────────────

def explain(query, algorithm):
    """Full RAG explain: retrieve context → augment prompt → generate → parse."""
    raw_text = generate_explanation(query, algorithm)
    parsed = parse_explanation(raw_text)
    return parsed, raw_text


# ─────────────────────────────────────────────
# STEP 6: Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate what the ranker would pass in
    test_algorithm = {
        "name": "Dijkstra's Algorithm",
        "description": "Dijkstra's algorithm finds the shortest path from a source vertex to all other vertices in a weighted graph with non-negative edge weights.",
        "complexity": "O((V+E) log V)",
        "use_case": "Used when you need the shortest path in a weighted graph where all edge weights are non-negative.",
        "url": "https://cp-algorithms.com/graph/dijkstra.html",
    }

    test_query = "find shortest path between two cities"

    print(f"Query: '{test_query}'")
    print(f"Algorithm: {test_algorithm['name']}")
    print("\nCalling Groq API ...\n")

    parsed, raw = explain(test_query, test_algorithm)

    print("=== RAW OUTPUT ===")
    print(raw)
    print("\n=== PARSED OUTPUT ===")
    for key, value in parsed.items():
        print(f"\n[{key.upper()}]")
        print(value)

    print("\n✅ Day 5 complete — Groq LLM integration working")
