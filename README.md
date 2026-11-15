# RAG Lab — Strategy Workbench

An interactive RAG workbench with a Vercel-compatible FastAPI app and a Streamlit local workbench. Upload a corpus, benchmark retrieval strategies, and get a data-type-aware starting recommendation. The existing Pinecone evaluation scripts remain available in `src/`.

## Run the workbench

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements-local.txt
streamlit run streamlit_app.py
```

Upload `.txt`, `.md`, `.csv`, `.json`, `.py`, `.html`, or `.pdf` files. PDFs must be selectable-text files under 4 MB; the combined extracted corpus is capped at 1.5 MB to stay within Vercel Function request limits. The app keeps uploads in the current browser session and does not persist them. You can tune chunk size/overlap, benchmark one or more questions, and inspect the full comparison matrix.

## Compared strategies

- Chunk boundaries: fixed-size, sentence-aware, paragraph-aware, recursive
- Retrieval/ranking: BM25, TF-IDF cosine, sparse term-vector similarity, reciprocal-rank hybrid fusion, and a lightweight hybrid rerank
- Sparse-vector distance: cosine, dot product, Euclidean, Manhattan, and Jaccard
- Metrics: term-overlap relevance, Precision@k, MRR, and measured latency

The benchmark uses sparse lexical scores and query-term overlap as a relevance proxy, not neural embeddings or labeled ground truth. Use judged query/document pairs before selecting a production configuration. The chat agent uses the Vercel AI SDK for Python and Vercel AI Gateway, calls corpus search as a tool, and cites retrieved passages. Prompts and retrieved passages are sent to the Gateway; uploads are not stored by this app.

## Existing Pinecone pipeline

Vercel discovers `app.py` as a FastAPI entrypoint; it serves the web workbench and `/api/*` routes. The original scripts under `src/` still support the separate Pinecone-backed experiment flow. Install `requirements-legacy.txt`, configure `PINECONE_API_KEY`, and use `experiments/questions.json` for the sample page labels.

## Tests

```bash
python -m pytest
```

The Vercel runtime installs the FastAPI, PDF, and Python AI SDK dependencies from `requirements.txt`. Vercel OIDC authenticates AI Gateway calls without a project API key. AI Gateway billing must be enabled for LLM responses; until then, the chat falls back to local retrieval. `requirements-local.txt` adds the Streamlit workbench and tests; `requirements-legacy.txt` adds the optional Pinecone and transformer pipeline dependencies.
