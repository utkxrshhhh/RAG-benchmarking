
import os
import re
import json
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

INDEX_NAME = "rag-evaluation-1000-100"
NAMESPACE = "__default__"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

K = 3
CANDIDATE_K = 20
RRF_K = 60

QUESTIONS_FILE = PROJECT_ROOT / "experiments" / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DETAILS_FILE = RESULTS_DIR / "retrieval_strategy_comparison.csv"
SUMMARY_FILE = RESULTS_DIR / "retrieval_strategy_summary.csv"

# Optional: set PINECONE_TEXT_FIELD in .env if automatic
# metadata text-field detection does not find your chunk field.
TEXT_FIELD = os.getenv("PINECONE_TEXT_FIELD", "").strip() or None


# ============================================================
# 2. CONNECT TO PINECONE AND LOAD MODELS
# ============================================================

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise RuntimeError(
        "PINECONE_API_KEY was not found. Check your project .env file."
    )

pc = Pinecone(api_key=api_key)
index = pc.Index(INDEX_NAME)

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Loading reranker...")
reranker = CrossEncoder(RERANKER_MODEL)


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def get_value(obj, key, default=None):
    """Read a field from either a dictionary or SDK object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def tokenize(text):
    """Simple tokenizer for BM25."""
    return re.findall(r"\b\w+\b", str(text).lower())


def normalize_page(page):
    """Normalize numeric page labels while preserving other labels."""
    if page is None:
        return None

    if isinstance(page, bool):
        return str(page)

    if isinstance(page, (int, float)):
        if float(page).is_integer():
            return str(int(page))
        return str(page)

    page = str(page).strip()

    try:
        number = float(page)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass

    return page


def get_expected_pages(item):
    pages = item.get("expected_pages", [])

    if not isinstance(pages, list):
        pages = [pages]

    return {
        normalize_page(page)
        for page in pages
        if page is not None
    }


def extract_matches(response):
    return get_value(response, "matches", []) or []


def match_to_document(match):
    metadata = get_value(match, "metadata", {}) or {}

    text = ""
    if TEXT_FIELD:
        text = metadata.get(TEXT_FIELD, "")
    else:
        for field in (
            "text", "chunk", "content", "page_content",
            "passage", "chunk_text", "document"
        ):
            if metadata.get(field):
                text = metadata[field]
                break

    return {
        "id": get_value(match, "id", ""),
        "text": str(text or ""),
        "page": normalize_page(metadata.get("page")),
        "metadata": metadata,
        "score": get_value(match, "score", 0.0),
    }


# ============================================================
# 4. LOAD ALL CHUNKS FROM PINECONE METADATA
# ============================================================

def load_all_documents():
    global TEXT_FIELD

    print("\nListing Pinecone vector IDs...")
    vector_ids = []

    # The installed Pinecone SDK returns ListResponse batches.
    # Each response contains a "vectors" field with vector IDs.
    for batch in index.list(namespace=NAMESPACE, limit=100):
        batch_vectors = get_value(batch, "vectors", []) or []

        for vector in batch_vectors:
            vector_id = get_value(vector, "id")

            if isinstance(vector_id, str):
                vector_ids.append(vector_id)

    # Remove duplicate IDs without changing their order.
    vector_ids = list(dict.fromkeys(vector_ids))

    print(f"Found {len(vector_ids)} vector IDs.")

    if not vector_ids:
        raise RuntimeError(
            "No vector IDs were returned. Check the index and namespace."
        )

    documents = []
    metadata_keys = set()

    # Fetch vectors and metadata in batches.
    for start in range(0, len(vector_ids), 100):
        batch_ids = vector_ids[start:start + 100]

        response = index.fetch(
            ids=batch_ids,
            namespace=NAMESPACE
        )

        vectors = get_value(response, "vectors", {}) or {}

        # Pinecone fetch normally returns a mapping of ID -> vector.
        if isinstance(vectors, dict):
            vector_items = vectors.values()
        else:
            vector_items = vectors

        for vector in vector_items:
            metadata = get_value(vector, "metadata", {}) or {}
            metadata_keys.update(metadata.keys())

            if not TEXT_FIELD:
                for field in (
                    "text", "chunk", "content", "page_content",
                    "passage", "chunk_text", "document"
                ):
                    if metadata.get(field):
                        TEXT_FIELD = field
                        break

            text = metadata.get(TEXT_FIELD, "") if TEXT_FIELD else ""

            if not text:
                continue

            documents.append({
                "id": get_value(vector, "id", ""),
                "text": str(text),
                "page": normalize_page(metadata.get("page")),
                "metadata": metadata,
            })

    if not documents:
        raise RuntimeError(
            "Vectors were found, but no chunk text was detected in metadata.\n"
            f"Available metadata keys: {sorted(metadata_keys)}\n"
            "Set PINECONE_TEXT_FIELD in your .env to the actual text field."
        )

    print(f"Loaded {len(documents)} documents.")
    print(f"Detected text field: {TEXT_FIELD}")
    print(f"Metadata keys: {sorted(metadata_keys)}")

    return documents


# ============================================================
# 5. DENSE RETRIEVAL
# ============================================================

def dense_search(query, top_n=CANDIDATE_K):
    query_vector = embedding_model.encode(query).tolist()

    response = index.query(
        vector=query_vector,
        top_k=top_n,
        include_metadata=True,
        namespace=NAMESPACE
    )

    documents = []

    for match in extract_matches(response):
        document = match_to_document(match)

        if document["id"]:
            documents.append(document)

    return documents


# ============================================================
# 6. BM25 RETRIEVAL
# ============================================================

def bm25_search(query, documents, bm25, top_n=CANDIDATE_K):
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for i in ranked_indices[:top_n]:
        document = documents[i].copy()
        document["score"] = float(scores[i])
        results.append(document)

    return results


# ============================================================
# 7. HYBRID RETRIEVAL WITH RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    top_n=CANDIDATE_K
):
    fused_scores = defaultdict(float)
    documents_by_id = {}

    for ranking in (dense_results, bm25_results):
        for rank, document in enumerate(ranking, start=1):
            doc_id = document["id"]

            if not doc_id:
                continue

            fused_scores[doc_id] += 1.0 / (RRF_K + rank)
            documents_by_id[doc_id] = document

    sorted_ids = sorted(
        fused_scores,
        key=lambda doc_id: fused_scores[doc_id],
        reverse=True
    )

    results = []

    for doc_id in sorted_ids[:top_n]:
        document = documents_by_id[doc_id].copy()
        document["score"] = fused_scores[doc_id]
        results.append(document)

    return results


# ============================================================
# 8. CROSS-ENCODER RERANKING
# ============================================================

def rerank_results(query, candidates, top_n=CANDIDATE_K):
    if not candidates:
        return []

    pairs = [
        (query, document["text"])
        for document in candidates
    ]

    scores = reranker.predict(pairs)

    reranked = []

    for document, score in zip(candidates, scores):
        item = document.copy()
        item["score"] = float(score)
        reranked.append(item)

    reranked.sort(
        key=lambda document: document["score"],
        reverse=True
    )

    return reranked[:top_n]


# ============================================================
# 9. EVALUATION METRICS
# ============================================================

def evaluate_ranking(ranking, expected_pages, k=K):
    top_results = ranking[:k]

    if not expected_pages:
        return {
            "recall_at_3": 0.0,
            "precision_at_3": 0.0,
            "mrr": 0.0,
        }

    retrieved_pages = {
        document["page"]
        for document in top_results
        if document.get("page") is not None
    }

    relevant_pages_found = retrieved_pages.intersection(expected_pages)

    # Page-label recall: how many expected page labels were retrieved?
    recall = len(relevant_pages_found) / len(expected_pages)

    # Chunk-level precision: fraction of top-k chunks from expected pages.
    relevant_chunks = sum(
        1
        for document in top_results
        if document.get("page") in expected_pages
    )

    precision = relevant_chunks / k

    # Reciprocal rank of the first chunk from an expected page.
    reciprocal_rank = 0.0

    for rank, document in enumerate(top_results, start=1):
        if document.get("page") in expected_pages:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "recall_at_3": recall,
        "precision_at_3": precision,
        "mrr": reciprocal_rank,
    }


# ============================================================
# 10. MAIN EXPERIMENT
# ============================================================

def main():
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Questions file not found: {QUESTIONS_FILE}"
        )

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not questions:
        raise RuntimeError("questions.json contains no questions.")

    documents = load_all_documents()

    print("\nPreparing BM25 index...")
    tokenized_corpus = [
        tokenize(document["text"])
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    all_rows = []
    start_all = time.perf_counter()

    for question_number, item in enumerate(questions, start=1):
        query = item["question"]
        expected_pages = get_expected_pages(item)

        print(f"\nQuestion {question_number}/{len(questions)}")
        print("Query:", query)

        # Dense retrieval
        start = time.perf_counter()
        dense_results = dense_search(query)
        dense_ms = (time.perf_counter() - start) * 1000

        # BM25 retrieval
        start = time.perf_counter()
        bm25_results = bm25_search(query, documents, bm25)
        bm25_ms = (time.perf_counter() - start) * 1000

        # Hybrid retrieval: RRF over dense and BM25 rankings
        start = time.perf_counter()
        hybrid_results = reciprocal_rank_fusion(
            dense_results,
            bm25_results
        )
        hybrid_ms = (time.perf_counter() - start) * 1000

        # Rerank the hybrid candidate set with a Cross-Encoder
        start = time.perf_counter()
        reranked_results = rerank_results(
            query,
            hybrid_results,
            top_n=CANDIDATE_K
        )
        rerank_ms = (time.perf_counter() - start) * 1000

        strategies = [
            ("dense", dense_results, dense_ms),
            ("bm25", bm25_results, bm25_ms),
            ("hybrid_rrf", hybrid_results, hybrid_ms),
            ("hybrid_reranked", reranked_results, rerank_ms),
        ]

        for strategy_name, ranking, elapsed_ms in strategies:
            metrics = evaluate_ranking(
                ranking,
                expected_pages,
                k=K
            )

            row = {
                "question_number": question_number,
                "question": query,
                "strategy": strategy_name,
                "recall_at_3": metrics["recall_at_3"],
                "precision_at_3": metrics["precision_at_3"],
                "mrr": metrics["mrr"],
                "stage_latency_ms": round(elapsed_ms, 2),
                "expected_pages": ", ".join(sorted(expected_pages)),
                "retrieved_pages_at_3": ", ".join(
                    str(document["page"])
                    for document in ranking[:K]
                    if document.get("page") is not None
                ),
            }

            all_rows.append(row)

            print(
                f"  {strategy_name:18} "
                f"Recall@3={metrics['recall_at_3']:.3f} "
                f"Precision@3={metrics['precision_at_3']:.3f} "
                f"MRR={metrics['mrr']:.3f}"
            )

    # Save per-question results
    details_df = pd.DataFrame(all_rows)
    details_df.to_csv(DETAILS_FILE, index=False)

    # Save strategy-level averages
    summary_df = (
        details_df
        .groupby("strategy", as_index=False)
        .agg(
            mean_recall_at_3=("recall_at_3", "mean"),
            mean_precision_at_3=("precision_at_3", "mean"),
            mean_mrr=("mrr", "mean"),
            mean_stage_latency_ms=("stage_latency_ms", "mean"),
        )
    )

    summary_df.to_csv(SUMMARY_FILE, index=False)

    total_seconds = time.perf_counter() - start_all

    print("\n" + "=" * 65)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("=" * 65)
    print(summary_df.to_string(index=False))

    print(f"\nDetailed results: {DETAILS_FILE}")
    print(f"Summary results:  {SUMMARY_FILE}")
    print(f"Total runtime: {total_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
