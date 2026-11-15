# RAG Lab — Strategy Workbench

An interactive research workbench for uploading a corpus, benchmarking retrieval strategies, and getting a data-type-aware starting recommendation. The existing Pinecone evaluation scripts remain available in `src/`.

## Run the workbench

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload `.txt`, `.md`, `.csv`, `.json`, `.py`, `.html`, or `.pdf` files. The app keeps uploads in the current Streamlit session and does not persist them. You can tune chunk size/overlap, benchmark one or more questions, and inspect the full comparison matrix.

## Compared strategies

- Chunk boundaries: fixed-size, sentence-aware, paragraph-aware, recursive
- Retrieval/ranking: BM25, TF-IDF cosine, sparse term-vector similarity, reciprocal-rank hybrid fusion, and a lightweight hybrid rerank
- Sparse-vector distance: cosine, dot product, Euclidean, Manhattan, and Jaccard
- Metrics: term-overlap relevance, Precision@k, MRR, and measured latency

The UI labels the sparse scoring methods honestly: these are not neural embedding vectors. The benchmark uses query-term overlap as a relevance proxy because it does not currently accept relevance labels. Use judged query/document pairs to select a production configuration; the highest-scoring result is not a universal guarantee. The strategy advisor is a retrieval-backed heuristic and does not generate LLM answers.

## Existing Pinecone pipeline

The original scripts under `src/` still support the separate Pinecone-backed experiment flow. Configure `PINECONE_API_KEY`, the matching index name, and metadata text field before running those scripts. Use `experiments/questions.json` as the sample set of questions with expected page labels.

## Tests

```bash
python -m pytest
```

The upload workbench tests run independently from the Pinecone and transformer-model integrations.
