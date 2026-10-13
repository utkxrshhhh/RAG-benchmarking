# RAG Evaluation

This project provides a structured workspace for evaluating retrieval-augmented generation (RAG) pipelines.

## Structure

- `data/` — datasets and evaluation fixtures
- `src/` — source code for the RAG pipeline and evaluation logic
- `experiments/` — experiment configs, notebooks, and run scripts
- `results/` — saved metrics, reports, and outputs
- `venv/` — Python virtual environment
- `requirements.txt` — project dependencies

## Getting started

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   .\venv\Scripts\activate    # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your evaluation code under `src/` and experiment scripts under `experiments/`.

## Notes

This repository is intentionally scaffolded so you can extend it with:
- document indexing
- retriever benchmarking
- answer generation evaluation
- quality and latency metrics
