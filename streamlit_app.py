from __future__ import annotations

import hashlib

import streamlit as st

from src.benchmark_engine import (
    benchmark,
    build_chunks,
    recommend,
    retrieve,
    supported_formats,
)

st.set_page_config(page_title="RAG Lab — Strategy Workbench", page_icon="◈", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#16212c; --muted:#667785; --line:#e4e9ec; --teal:#0b7b70; --mint:#e7f5f1; }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; color:var(--ink); }
    .stApp { background: #f7f9f8; }
    section[data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
    .hero { padding:1.1rem 0 .8rem; }
    .eyebrow { color:var(--teal); text-transform:uppercase; letter-spacing:.16em; font-size:.71rem; font-weight:800; }
    .hero h1 { font-size:2.4rem; letter-spacing:-.055em; line-height:1.12; margin:.4rem 0; }
    .hero p { color:var(--muted); max-width:680px; }
    .card { background:white; border:1px solid var(--line); border-radius:14px; padding:1.15rem 1.25rem; }
    .metric-label { color:var(--muted); font-size:.76rem; text-transform:uppercase; letter-spacing:.09em; font-weight:700; }
    .metric-value { font-size:1.65rem; font-weight:800; letter-spacing:-.04em; margin-top:.2rem; }
    .tag { display:inline-block; background:var(--mint); color:var(--teal); border-radius:99px; padding:.3rem .65rem; font-size:.76rem; font-weight:700; }
    div[data-testid="stMetric"] { background:white; border:1px solid var(--line); padding:1rem; border-radius:12px; }
    .stButton > button[kind="primary"] { background:#0b7b70; border-color:#0b7b70; }
    [data-testid="stChatMessage"] { background:white; border:1px solid var(--line); border-radius:14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "documents" not in st.session_state:
    st.session_state.documents = []
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### ◈ RAG Lab")
    st.caption("Strategy workbench")
    st.divider()
    st.markdown("**1 · Add your corpus**")
    uploads = st.file_uploader(
        "Upload source files",
        type=supported_formats(),
        accept_multiple_files=True,
        help="Text, Markdown, CSV, JSON, Python, HTML and PDF are supported.",
    )
    if uploads:
        upload_signature = (
            tuple((upload.name, hashlib.sha256(upload.getvalue()).hexdigest()) for upload in uploads),
            chunk_size,
            overlap,
        )
        if upload_signature != st.session_state.get("upload_signature"):
            try:
                st.session_state.documents = build_chunks(uploads, chunk_size=chunk_size, overlap=overlap)
                st.session_state.upload_signature = upload_signature
                st.session_state.pop("results", None)
                st.success(f"Indexed {len(st.session_state.documents)} chunks")
            except Exception as exc:
                st.error(f"Could not read these files: {exc}")
    if st.session_state.documents and st.button("Clear corpus"):
        st.session_state.documents = []
        st.session_state.messages = []
        st.session_state.pop("results", None)
        st.rerun()
    st.caption("Documents stay in this session; nothing is saved to a database.")
    st.divider()
    st.markdown("**2 · Tune the experiment**")
    chunk_size = st.slider("Chunk size (characters)", 200, 1600, 600, 100)
    overlap = st.slider("Chunk overlap", 0, min(300, chunk_size // 2), 80, 20)
    top_k = st.slider("Results per question", 1, 10, 4)

st.markdown(
    '<div class="hero"><div class="eyebrow">Retrieval evaluation studio</div>'
    '<h1>Find the right RAG recipe<br/>for your data.</h1>'
    '<p>Compare chunking, retrieval, ranking and distance strategies on your own corpus. '
    'Then ask the strategy advisor why a configuration fits.</p></div>',
    unsafe_allow_html=True,
)

if not st.session_state.documents:
    st.info("Upload documents in the left panel to start a real benchmark. A small sample corpus is available below to explore the workflow.")
    if st.button("Load sample research corpus", type="primary"):
        sample = [
            {"id": "attention-1", "source": "transformer-notes.md", "text": "Self-attention lets each token compare itself with every other token in a sequence. Query, key, and value projections are used to compute contextual representations. Multi-head attention runs several attention operations in parallel."},
            {"id": "attention-2", "source": "transformer-notes.md", "text": "A Transformer block combines multi-head self-attention with a feed-forward network, residual connections, and layer normalization. Positional encodings provide order information."},
            {"id": "bert-1", "source": "nlp-guide.md", "text": "BERT is a bidirectional encoder model pretrained with masked language modeling. Its contextual embeddings depend on both left and right context and are useful for classification and extraction."},
            {"id": "bert-2", "source": "nlp-guide.md", "text": "For relation extraction, BERT encodes a sentence and a classification head predicts the relation between marked entities. Fine-tuning adapts the encoder to labeled examples."},
            {"id": "vectors-1", "source": "retrieval-guide.md", "text": "Cosine similarity compares the angle between normalized embedding vectors. It is a strong default for semantic search. Dot product is useful when the embedding model was trained for it."},
            {"id": "vectors-2", "source": "retrieval-guide.md", "text": "BM25 is a lexical ranking function that scores term overlap with document length normalization and inverse document frequency. Hybrid search combines lexical and dense rankings."},
        ]
        st.session_state.documents = build_chunks(sample, chunk_size=chunk_size, overlap=overlap, prechunked=True)
        st.rerun()

if st.session_state.documents:
    docs = st.session_state.documents
    advice = recommend(docs)
    left, right = st.columns([1.75, 1], gap="large")
    with left:
        st.markdown("#### Corpus profile")
        data_cols = st.columns(4)
        data_cols[0].metric("Documents", len({d["source"] for d in docs}))
        data_cols[1].metric("Chunks", len(docs))
        data_cols[2].metric("Detected type", advice["data_type"])
        data_cols[3].metric("Avg. chunk", f'{advice["avg_chunk"]} chars')
        st.markdown("#### Strategy benchmark")
        st.caption("Runs the same questions across retrieval, distance, and chunking choices. The scores use term-overlap relevance because no labeled ground truth was uploaded; they are not a quality guarantee.")
        query_text = st.text_area("Benchmark questions (one per line)", "What is self-attention?\nHow does BERT create contextual embeddings?\nWhat is the best distance metric for semantic search?", height=105)
        if st.button("Run comparison", type="primary", use_container_width=True):
            queries = [q.strip() for q in query_text.splitlines() if q.strip()]
            if not queries:
                st.warning("Add at least one question.")
            else:
                st.session_state.results = benchmark(docs, queries, top_k=top_k, chunk_size=chunk_size, overlap=overlap)
        results = st.session_state.get("results")
        if results is not None and not results.empty:
            st.dataframe(results.head(12), use_container_width=True, hide_index=True)
            winner = results.iloc[0]
            st.success(f"Best observed on these questions: **{winner['retrieval']}** with **{winner['chunking']}** chunks and **{winner['distance']}** distance · relative relevance {winner['relevance']:.3f}.")
            with st.expander("View all strategy combinations"):
                st.dataframe(results, use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Recommended starting point")
        st.markdown(f'<div class="card"><span class="tag">{advice["retrieval"]}</span><h3 style="margin:.8rem 0 .35rem">{advice["headline"]}</h3><p style="color:#667785">{advice["reason"]}</p><hr style="border:0;border-top:1px solid #e4e9ec"><b>Chunking</b><p>{advice["chunking"]}</p><b>Distance</b><p>{advice["distance"]}</p></div>', unsafe_allow_html=True)
        st.markdown("#### Strategy advisor")
        st.caption("Ask about your data profile or the benchmark results.")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        prompt = st.chat_input("e.g. Why hybrid search for this corpus?")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            answer, hits = retrieve(docs, prompt, top_k=top_k, recommendation=advice)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

    st.divider()
    with st.expander("What is being compared?"):
        st.markdown("**Retrieval:** BM25 keyword, TF-IDF cosine, sparse term-vector similarity, reciprocal-rank hybrid fusion, and hybrid with a lightweight rerank.  \\n**Distance:** cosine, dot product, Euclidean, Manhattan, and Jaccard over sparse term vectors (BM25 does not use a vector distance).  \n**Chunking:** fixed-size, sentence-aware, paragraph-aware, and recursive boundary strategies.  \n**Metrics:** Precision@k, reciprocal rank, relative relevance, and elapsed time. Results are local heuristics; validate against labeled relevance judgments before production use.")
else:
    st.markdown("#### Included strategy matrix")
    st.dataframe(
        [{"Layer": "Chunking", "Options": "Fixed · sentence-aware · paragraph-aware · recursive"}, {"Layer": "Retrieval", "Options": "BM25 · TF-IDF · vector · hybrid RRF · reranked hybrid"}, {"Layer": "Distance", "Options": "Cosine · dot product · Euclidean · Manhattan · Jaccard"}, {"Layer": "Evaluation", "Options": "Precision@k · MRR · relevance · latency"}],
        use_container_width=True,
        hide_index=True,
    )
