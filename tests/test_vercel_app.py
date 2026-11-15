from fastapi.testclient import TestClient

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
    assert client.get("/api/health").json() == {"ok": True, "service": "rag-lab"}


def test_recommendation_and_chat_routes():
    recommendation = client.post("/api/recommend", json=CORPUS)
    assert recommendation.status_code == 200
    assert recommendation.json()["recommendation"]["data_type"] == "Prose / mixed"
    assert recommendation.json()["documents"] == 2

    answer = client.post("/api/ask", json={**CORPUS, "question": "What is self-attention?"})
    assert answer.status_code == 200
    assert "Evidence retrieved from your files" in answer.json()["answer"]
    assert "notes.md" in answer.json()["answer"]


def test_benchmark_route_and_input_validation():
    response = client.post("/api/benchmark", json={**CORPUS, "questions": ["What is self-attention?"], "top_k": 2})
    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert {"chunking", "distance", "retrieval", "MRR"}.issubset(results[0])
    assert client.post("/api/recommend", json={**CORPUS, "overlap": 300}).status_code == 422
