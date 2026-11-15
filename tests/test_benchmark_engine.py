from src.benchmark_engine import _rank, _split, benchmark, recommend


def test_chunkers_preserve_source_text():
    text = "First sentence. Second sentence.\n\nA new paragraph with useful details."
    for mode in ("fixed", "sentence", "paragraph", "recursive"):
        chunks = _split(text, 28, 5, mode)
        assert chunks
        assert all(chunk.strip() for chunk in chunks)


def test_bm25_and_hybrid_return_ranked_documents():
    docs = [
        {"id": "a", "source": "a.txt", "text": "Self attention uses query key value projections."},
        {"id": "b", "source": "b.txt", "text": "BM25 ranks documents using term frequency."},
    ]
    assert _rank(docs, "self attention query", "BM25 keyword")[0][1]["id"] == "a"
    assert _rank(docs, "self attention query", "Hybrid RRF", "cosine")[0][1]["id"] == "a"


def test_recommender_detects_code_and_structured_data():
    code = [{"source": "app.py", "text": "def retrieve(query):\n    return query"}]
    structured = [{"source": "items.csv", "text": "name | price | category\nBook | 12 | learning"}]
    assert recommend(code)["data_type"] == "Code"
    assert recommend(structured)["data_type"] == "Structured"


def test_benchmark_compares_dimensions():
    docs = [
        {"id": "a", "source": "a.md", "text": "Self attention uses query key value projections.", "raw_text": "Self attention uses query key value projections."},
        {"id": "b", "source": "b.md", "text": "BM25 ranks documents using term frequency.", "raw_text": "BM25 ranks documents using term frequency."},
    ]
    results = benchmark(docs, ["self attention"], top_k=2, chunk_size=80, overlap=10)
    assert {"chunking", "distance", "retrieval", "relevance", "MRR"}.issubset(results.columns)
    assert set(results["chunking"]) == {"fixed", "sentence", "paragraph", "recursive"}
    assert "jaccard" in set(results["distance"])
    assert "BM25 keyword" in set(results["retrieval"])
