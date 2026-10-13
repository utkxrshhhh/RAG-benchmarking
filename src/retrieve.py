import os
import json

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer



load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

# Connect to Pinecone
pc = Pinecone(api_key=api_key)

# Connect to our index
index = pc.Index("rag-evaluation-1000-100")

# Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

with open("experiments/questions.json", "r") as file:
    questions = json.load(file)

hits = 0

for item in questions:

    query = item["question"]
    expected_pages = item["expected_pages"]

    # embedding
    query_embedding = model.encode(query)

    # Pinecone search
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=3,
        include_metadata=True
    )

    # collect pages
    retrieved_pages = []

    for match in results["matches"]:
        retrieved_pages.append(
            match["metadata"].get("page")
        )

    # Hit/Miss
    hit = any(
        page in expected_pages
        for page in retrieved_pages
    )

    if hit:
        hits += 1

    print("Question:", query)
    print("Hit:", hit)


print("\n===== SUMMARY =====")
print("Total questions:", len(questions))
print("Hits:", hits)
print("Misses:", len(questions) - hits)
   