import sys
sys.path.insert(0, "src")

from ai_deep_agent.memory.vector_store import ingest_text, retrieve_context

# Koi bhi relevant text inject karo
ingest_text("Merge sort is a divide and conquer algorithm with O(n log n) time complexity.", "algo_notes")

# Check karo ki research agent ko milta hai ya nahi
ctx = retrieve_context("merge sort time complexity")
print("RAG Context:", ctx[:200])