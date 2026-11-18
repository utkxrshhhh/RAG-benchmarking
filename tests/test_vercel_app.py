from pathlib import Path

from fastapi.testclient import TestClient

import app as app_module
from app import app


client = TestClient(app)

CORPUS = {
    "files": [
        {"source": "notes.md", "text": "Self-attention uses query, key, and value vectors to build contextual representations."},
        {"source": "nlp.md", "text": "BERT is a bidirectional encoder. Its contextual embeddings depend on surrounding words."},
    ],
    "chunk_size": 400,
    "overlap": 40,
}


def test_vercel_fastapi_entrypoint_and_headers():
    response = client.get("/")
    assert response.status_code == 200
    assert "RAG Lab" in response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    health = client.get("/api/health").json()
    assert health["ok"] is True and health["service"] == "rag-lab"
    assert {"provider", "model", "free"} <= set(health["llm"])


def test_recommendation_and_chat_routes(monkeypatch):
    recommendation = client.post("/api/recommend", json=CORPUS)
    assert recommendation.status_code == 200
    assert recommendation.json()["recommendation"]["data_type"] == "Prose / mixed"
    assert recommendation.json()["documents"] == 2

    async def fake_agent(question, history, docs, top_k, advice):
        assert question == "What is self-attention?"
        assert history[0].content == "What is attention?"
        return "Self-attention builds contextual representations [notes.md].", [
            {"source": "notes.md", "snippet": "Query, key, and value vectors."}
        ]

    monkeypatch.setattr(app_module, "run_rag_agent", fake_agent)
    answer = client.post(
        "/api/ask",
        json={
            **CORPUS,
            "question": "What is self-attention?",
            "history": [{"role": "user", "content": "What is attention?"}],
        },
    )
    assert answer.status_code == 200
    assert answer.json()["mode"] == "llm"
    assert "Self-attention builds contextual representations" in answer.json()["answer"]
    assert answer.json()["sources"][0]["source"] == "notes.md"


def test_llm_billing_error_returns_grounded_fallback(monkeypatch):
    async def unavailable_agent(*args):
        raise ExceptionGroup(
            "agent request failed",
            [RuntimeError("AI Gateway requires a valid credit card on file")],
        )

    monkeypatch.setattr(app_module, "run_rag_agent", unavailable_agent)
    response = client.post("/api/ask", json={**CORPUS, "question": "What is self-attention?"})
    assert response.status_code == 200
    assert response.json()["mode"] == "retrieval-fallback"
    assert "GROQ_API_KEY" in response.json()["notice"]
    assert "notes.md" in response.json()["answer"]


def test_benchmark_route_and_input_validation():
    response = client.post("/api/benchmark", json={**CORPUS, "questions": ["What is self-attention?"], "top_k": 2})
    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert {"chunking", "distance", "retrieval", "MRR"}.issubset(results[0])
    assert client.post("/api/recommend", json={**CORPUS, "overlap": 300}).status_code == 422


def test_oversized_corpus_fails_with_a_json_message():
    oversized = {
        "files": [{"source": "large.txt", "text": "x" * 1_500_001}],
        "chunk_size": 600,
        "overlap": 80,
    }
    response = client.post("/api/recommend", json=oversized)
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert any("1.5 MB" in error["msg"] for error in response.json()["detail"])


def test_pdf_upload_returns_text_and_json_errors():
    pdf_path = Path(__file__).parents[1] / "data" / "documents" / "2005.11401v4.pdf"
    response = client.post(
        "/api/parse",
        files={"file": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert "Retrieval-Augmented Generation" in response.json()["text"]

    invalid = client.post(
        "/api/parse",
        files={"file": ("broken.pdf", b"not a valid PDF", "application/pdf")},
    )
    assert invalid.status_code == 400
    assert invalid.headers["content-type"].startswith("application/json")
    assert "Could not extract text" in invalid.json()["detail"]
