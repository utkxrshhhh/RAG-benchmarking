from __future__ import annotations

import csv
import io
import json
import math
import re
import time
from collections import Counter
from typing import Any

SUPPORTED = ["txt", "md", "csv", "json", "py", "html", "pdf"]


def supported_formats() -> list[str]:
    return SUPPORTED


def _read_upload(upload: Any) -> tuple[str, str]:
    name = getattr(upload, "name", "document.txt")
    raw = upload.getvalue() if hasattr(upload, "getvalue") else bytes(upload)
    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else "txt"
    if suffix == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF support needs pypdf installed. Reinstall the project requirements.") from exc
        reader = PdfReader(io.BytesIO(raw))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix == "csv":
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8", errors="replace"))))
        text = "\n".join(" | ".join(cell for cell in row) for row in rows)
    elif suffix == "json":
        text = json.dumps(json.loads(raw.decode("utf-8", errors="replace")), ensure_ascii=False, indent=2)
    else:
        text = raw.decode("utf-8", errors="replace")
        if suffix == "html":
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
    if not text.strip():
        raise ValueError(f"{name} does not contain extractable text.")
    return name, text


def _split(text: str, size: int, overlap: int, mode: str) -> list[str]:
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []
    if mode == "sentence":
        units = re.split(r"(?<=[.!?])\s+", text)
    elif mode == "paragraph":
        units = re.split(r"\n\s*\n", text)
    elif mode == "recursive":
        units = re.split(r"\n\s*\n|(?<=[.!?])\s+|\s+", text)
    else:
        units = []
    if not units:
        step = max(1, size - overlap)
        return [text[i:i + size].strip() for i in range(0, len(text), step) if text[i:i + size].strip()]
    chunks: list[str] = []
    current = ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > size:
            if current:
                chunks.append(current)
                current = ""
            step = max(1, size - overlap)
            chunks.extend(unit[i:i + size].strip() for i in range(0, len(unit), step) if unit[i:i + size].strip())
        elif not current:
            current = unit
        elif len(current) + len(unit) + 1 <= size:
            current += " " + unit
        else:
            chunks.append(current)
            carry = current[-overlap:] if overlap else ""
            current = (carry + " " + unit).strip()
    if current:
        chunks.append(current)
    return chunks


def build_chunks(uploads: list[Any], chunk_size: int = 600, overlap: int = 80, prechunked: bool = False) -> list[dict[str, Any]]:
    records = []
    for upload in uploads:
        if prechunked:
            records.append(upload)
            continue
        name, text = _read_upload(upload)
        for i, chunk in enumerate(_split(text, chunk_size, overlap, "recursive")):
            records.append({"id": f"{name}:{i}", "source": name, "text": chunk, "raw_text": text})
    return records


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower())


def _cosine(a: Counter, b: Counter) -> float:
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _bm25(query: list[str], docs: list[list[str]]) -> list[float]:
    if not docs:
        return []
    avg_len = sum(map(len, docs)) / len(docs) or 1
    df = Counter(token for doc in docs for token in set(doc))
    scores = []
    for doc in docs:
        tf = Counter(doc)
        score = 0.0
        for token in query:
            if tf[token]:
                idf = math.log(1 + (len(docs) - df[token] + 0.5) / (df[token] + 0.5))
                score += idf * tf[token] * 2.2 / (tf[token] + 1.2 * (0.25 + 0.75 * len(doc) / avg_len))
        scores.append(score)
    return scores


def _distance_scores(query: Counter, vectors: list[Counter], distance: str) -> list[float]:
    if distance == "jaccard":
        return [len(set(query) & set(vector)) / max(1, len(set(query) | set(vector))) for vector in vectors]
    if distance == "dot product":
        return [sum(query[token] * vector.get(token, 0) for token in query) for vector in vectors]
    if distance == "euclidean":
        return [1 / (1 + math.sqrt(sum((query[token] - vector.get(token, 0)) ** 2 for token in set(query) | set(vector)))) for vector in vectors]
    if distance == "manhattan":
        return [1 / (1 + sum(abs(query[token] - vector.get(token, 0)) for token in set(query) | set(vector))) for vector in vectors]
    return [_cosine(query, vector) for vector in vectors]


def _rank(docs: list[dict], query: str, strategy: str, distance: str = "cosine") -> list[tuple[float, dict]]:
    q = _tokens(query)
    corpus = [_tokens(d["text"]) for d in docs]
    qvec, vectors = Counter(q), [Counter(t) for t in corpus]
    bm = _bm25(q, corpus)
    dense = _distance_scores(qvec, vectors, distance)
    if strategy == "BM25 keyword":
        scores = bm
    elif strategy == "TF-IDF cosine":
        df = Counter(token for row in corpus for token in set(row))
        weighted = [Counter({t: c * math.log(1 + len(corpus) / (1 + df[t])) for t, c in Counter(row).items()}) for row in corpus]
        qweighted = Counter({t: c * math.log(1 + len(corpus) / (1 + df[t])) for t, c in qvec.items()})
        scores = _distance_scores(qweighted, weighted, distance)
    elif strategy == "Sparse term vector":
        scores = dense
    elif strategy == "Hybrid RRF":
        bm_order = sorted(range(len(docs)), key=lambda i: bm[i], reverse=True)
        dense_order = sorted(range(len(docs)), key=lambda i: dense[i], reverse=True)
        br = {i: rank for rank, i in enumerate(bm_order, 1)}
        dr = {i: rank for rank, i in enumerate(dense_order, 1)}
        scores = [1 / (60 + br[i]) + 1 / (60 + dr[i]) for i in range(len(docs))]
    else:
        fused = []
        bm_order = sorted(range(len(docs)), key=lambda i: bm[i], reverse=True)
        dense_order = sorted(range(len(docs)), key=lambda i: dense[i], reverse=True)
        br = {i: rank for rank, i in enumerate(bm_order, 1)}
        dr = {i: rank for rank, i in enumerate(dense_order, 1)}
        fused = [1 / (60 + br[i]) + 1 / (60 + dr[i]) for i in range(len(docs))]
        scores = [fused[i] + 0.002 * bm[i] for i in range(len(docs))]
    return sorted(((float(score), doc) for score, doc in zip(scores, docs)), key=lambda pair: pair[0], reverse=True)


STRATEGIES = ["BM25 keyword", "TF-IDF cosine", "Sparse term vector", "Hybrid RRF", "Hybrid + rerank"]
DISTANCES = ["cosine", "dot product", "euclidean", "manhattan", "jaccard"]
CHUNKERS = ["fixed", "sentence", "paragraph", "recursive"]


def source_texts(docs: list[dict]) -> dict[str, str]:
    """Rebuild complete source text so chunking variants are evaluated from the same corpus."""
    sources: dict[str, str] = {}
    raw_sources = {doc["source"] for doc in docs if doc.get("raw_text")}
    for doc in docs:
        source = doc["source"]
        if doc.get("raw_text"):
            sources[source] = doc["raw_text"]
        elif source not in raw_sources:
            sources[source] = f'{sources.get(source, "")}\n\n{doc["text"]}'.strip()
    return sources


def chunk_corpus(sources: dict[str, str], chunker: str, chunk_size: int, overlap: int) -> list[dict]:
    return [
        {"id": f"{source}:{index}", "source": source, "text": chunk}
        for source, text in sources.items()
        for index, chunk in enumerate(_split(text, chunk_size, overlap, chunker))
    ]


def retrieve_with(
    docs: list[dict],
    query: str,
    top_k: int,
    strategy: str,
    distance: str,
    chunker: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    variant_docs = chunk_corpus(source_texts(docs), chunker, chunk_size, overlap)
    effective_distance = "cosine" if distance not in DISTANCES else distance
    ranked = _rank(variant_docs, query, strategy, effective_distance)[:top_k]
    return [{**doc, "score": round(score, 4)} for score, doc in ranked if score > 0]


def passage_quality(query: str, passages: list[dict]) -> dict[str, float]:
    """Term-overlap proxy: how well a set of retrieved passages covers the query."""
    qterms = set(_tokens(query))
    if not qterms or not passages:
        return {"coverage": 0.0, "relevance": 0.0}
    passage_terms = [set(_tokens(passage["text"])) for passage in passages]
    covered = qterms & set().union(*passage_terms)
    relevance = sum(len(qterms & terms) / len(qterms) for terms in passage_terms) / len(passage_terms)
    return {"coverage": round(len(covered) / len(qterms), 3), "relevance": round(relevance, 3)}


def select_best(rows: list[dict]) -> dict[str, Any] | None:
    """Benchmark selector: rank configurations by a weighted composite of quality metrics, then latency."""
    if not rows:
        return None
    latencies = [row["latency (ms)"] for row in rows]
    low, high = min(latencies), max(latencies)

    def composite(row: dict) -> float:
        speed_penalty = (row["latency (ms)"] - low) / (high - low) if high > low else 0.0
        return 0.5 * row["relevance"] + 0.3 * row["MRR"] + 0.2 * row["precision@k"] - 0.05 * speed_penalty

    ranked = sorted(rows, key=composite, reverse=True)
    best, runner_up = ranked[0], ranked[1] if len(ranked) > 1 else None
    reason = (
        f"{best['retrieval']} with {best['chunking']} chunks and {best['distance']} distance had the highest "
        f"composite score (relevance {best['relevance']}, MRR {best['MRR']}, precision@k {best['precision@k']}, "
        f"{best['latency (ms)']} ms)."
    )
    if runner_up:
        reason += f" Runner-up: {runner_up['retrieval']} · {runner_up['chunking']} · {runner_up['distance']}."
    return {
        "retrieval": best["retrieval"],
        "chunking": best["chunking"],
        "distance": best["distance"],
        "composite": round(composite(best), 3),
        "relevance": best["relevance"],
        "MRR": best["MRR"],
        "precision@k": best["precision@k"],
        "latency_ms": best["latency (ms)"],
        "reason": reason,
    }


def benchmark(docs: list[dict], queries: list[str], top_k: int = 4, chunk_size: int = 600, overlap: int = 80):
    import pandas as pd

    sources = source_texts(docs)
    rows = []
    for chunker in CHUNKERS:
        variant_docs = chunk_corpus(sources, chunker, chunk_size, overlap)
        for strategy in STRATEGIES:
            applicable_distances = ["n/a (BM25)"] if strategy == "BM25 keyword" else DISTANCES
            for distance in applicable_distances:
                start = time.perf_counter()
                relevance, reciprocal, precisions = [], [], []
                for query in queries:
                    ranked = _rank(variant_docs, query, strategy, distance)[:top_k]
                    qterms = set(_tokens(query))
                    relevances = [len(qterms & set(_tokens(item["text"]))) / max(1, len(qterms)) for _, item in ranked]
                    relevance.append(sum(relevances) / max(1, len(relevances)))
                    reciprocal.append(next((1 / i for i, score in enumerate(relevances, 1) if score > 0), 0.0))
                    precisions.append(sum(score > 0 for score in relevances) / max(1, top_k))
                elapsed = (time.perf_counter() - start) * 1000
                rows.append({"chunking": chunker, "distance": distance, "retrieval": strategy, "relevance": round(sum(relevance) / len(queries), 3), "precision@k": round(sum(precisions) / len(queries), 3), "MRR": round(sum(reciprocal) / len(queries), 3), "latency (ms)": round(elapsed, 2)})
    return pd.DataFrame(rows).sort_values(["relevance", "MRR", "latency (ms)"], ascending=[False, False, True]).reset_index(drop=True)


def recommend(docs: list[dict]) -> dict[str, str | int]:
    text = " ".join(doc["text"] for doc in docs)
    avg = round(sum(len(doc["text"]) for doc in docs) / max(1, len(docs)))
    structured = sum(text.count(x) for x in ["|", "{", "}", "="]) / max(1, len(text))
    code = sum(text.count(x) for x in ["def ", "class ", "function ", "import "]) / max(1, len(text))
    if code > 0.004:
        kind, retrieval, headline = "Code", "Hybrid RRF", "Keep identifiers exact; add semantic recall"
        reason = "Code queries often depend on exact symbols and literals, while hybrid retrieval can recover concept-level matches too."
        chunking, distance = "Recursive chunks that preserve functions/classes; overlap 10–15%.", "Cosine for normalized embeddings; evaluate dot product if your model is trained for it."
    elif structured > 0.012:
        kind, retrieval, headline = "Structured", "BM25 + metadata filters", "Preserve fields and row boundaries"
        reason = "Structured content benefits from stable field labels and exact lexical matches. Avoid splitting records across chunks."
        chunking, distance = "Row/record-aware chunks; keep headers with each record.", "BM25 for exact values; cosine can supplement natural-language queries."
    elif avg > 1100:
        kind, retrieval, headline = "Long-form prose", "Hybrid + rerank", "Retrieve broadly, then focus"
        reason = "Long-form sources can bury relevant passages. A larger candidate pool plus reranking is a useful starting point."
        chunking, distance = "Recursive or paragraph-aware chunks around 500–900 characters with 10–15% overlap.", "Cosine or dot product, benchmarked against the embedding model’s training objective."
    else:
        kind, retrieval, headline = "Prose / mixed", "Hybrid RRF", "Balance meaning and exact terms"
        reason = "This corpus looks like natural-language or mixed text. Hybrid fusion is a robust baseline when queries mix concepts and named terms."
        chunking, distance = "Sentence-aware or recursive chunks around 400–800 characters with modest overlap.", "Cosine is a strong normalized-vector baseline; compare against dot product."
    return {"data_type": kind, "retrieval": retrieval, "headline": headline, "reason": reason, "chunking": chunking, "distance": distance, "avg_chunk": avg}


def retrieve(docs: list[dict], query: str, top_k: int, recommendation: dict) -> tuple[str, list[dict]]:
    ranked = _rank(docs, query, "Hybrid RRF")[:top_k]
    relevant = [(score, doc) for score, doc in ranked if score > 0]
    if not relevant:
        return (f"I couldn’t find a strong match in the uploaded corpus. For this {recommendation['data_type'].lower()} data, I’d start with **{recommendation['retrieval']}** and test your own labeled questions. Try a more specific query or add relevant documents.", [])
    sources = "\n".join(f"- **{doc['source']}** — {doc['text'][:520]}{'…' if len(doc['text']) > 520 else ''}" for _, doc in relevant)
    answer = f"For this **{recommendation['data_type'].lower()}** corpus, start with **{recommendation['retrieval']}**. {recommendation['reason']}\n\n**Evidence retrieved from your files:**\n{sources}\n\nThis is a local retrieval-only fallback based on sparse lexical signals, not a model-generated answer. Validate the strategy with relevance labels before production use."
    return answer, [doc for _, doc in relevant]
