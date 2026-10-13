
import os
import json

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)
index = pc.Index("rag-evaluation-1000-100")

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("experiments/questions.json", "r") as file:
    questions = json.load(file)

K = 3

total_recall = 0
total_precision = 0
total_mrr = 0

for item in questions:

    question = item["question"]
    expected_pages = item["expected_pages"]

    if not isinstance(expected_pages, list):
        expected_pages = [expected_pages]

    expected_pages = {str(p) for p in expected_pages}

    vector = model.encode(question).tolist()

    result = index.query(
        vector=vector,
        top_k=K,
        include_metadata=True
    )

    matches = result.matches

    retrieved_pages = [
        str(match.metadata.get("page"))
        for match in matches
        if match.metadata
        and match.metadata.get("page") is not None
    ]

    # Recall: relevant expected pages found
    found = set(retrieved_pages) & expected_pages

    recall = (
        len(found) / len(expected_pages)
        if expected_pages else 0
    )

    # Precision: relevant retrieved chunks
    relevant = [
        page in expected_pages
        for page in retrieved_pages
    ]

    precision = sum(relevant) / K

    # MRR: rank of first relevant result
    reciprocal_rank = 0

    for rank, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            reciprocal_rank = 1 / rank
            break

    total_recall += recall
    total_precision += precision
    total_mrr += reciprocal_rank

    print("\nQuestion:", question)
    print("Recall@3:", round(recall, 2))
    print("Precision@3:", round(precision, 2))
    print("MRR:", round(reciprocal_rank, 2))

count = len(questions)

if count == 0:
    raise ValueError("No questions found.")

print("\n===== FINAL RESULTS =====")
print("Questions:", count)
print("Recall@3:", round(total_recall / count, 3))
print("Precision@3:", round(total_precision / count, 3))
print("MRR:", round(total_mrr / count, 3))
