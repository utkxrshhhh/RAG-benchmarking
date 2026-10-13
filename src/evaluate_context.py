
import os
import json
import csv
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY missing from .env")

# Connect to existing Pinecone index
pc = Pinecone(api_key=api_key)
index = pc.Index("rag-evaluation-1000-100")

# Reuse your existing embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

with open("experiments/questions.json", encoding="utf-8") as f:
    questions = json.load(f)

os.makedirs("results", exist_ok=True)
output_path = "results/context_relevance.csv"

rows = []

for i, item in enumerate(questions, start=1):
    question = item["question"]

    query_vector = model.encode([question])[0]

    response = index.query(
        vector=query_vector.tolist(),
        top_k=3,
        include_metadata=True
    )

    passages = []
    pages = []

    for match in response["matches"]:
        metadata = match.get("metadata") or {}

        page = str(metadata.get("page", "Unknown"))
        pages.append(page)

        passage = (
            metadata.get("text")
            or metadata.get("chunk")
            or metadata.get("page_content")
            or ""
        )

        if passage.strip():
            passages.append(passage)

    if not passages:
        raise ValueError(
            "Retrieved text missing from Pinecone metadata. "
            "Check src/upload_vectors.py."
        )

    # Compare question embedding with each retrieved passage
    passage_vectors = model.encode(passages)

    similarities = cosine_similarity(
        [query_vector],
        passage_vectors
    )[0]

    # Average similarity across the retrieved passages
    avg_similarity = float(similarities.mean())

    # A simple threshold-based label; treat as a heuristic
    if avg_similarity >= 0.45:
        label = "High"
    elif avg_similarity >= 0.30:
        label = "Medium"
    else:
        label = "Low"

    rows.append({
        "question": question,
        "retrieved_pages": ", ".join(pages),
        "passages_scored": len(passages),
        "average_similarity": round(avg_similarity, 4),
        "context_relevance": label
    })

    print(f"\nQuestion {i}: {question}")
    print("Retrieved pages:", pages)
    print(f"Average similarity: {avg_similarity:.4f}")
    print("Context relevance:", label)

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\n===== CONTEXT RELEVANCE SUMMARY =====")
print("Questions evaluated:", len(rows))

if rows:
    average = sum(
        row["average_similarity"] for row in rows
    ) / len(rows)

    print(f"Mean question-context similarity: {average:.4f}")

print("Saved results to:", output_path)
