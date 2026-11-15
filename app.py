from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator
from pypdf import PdfReader

from src.benchmark_engine import benchmark, build_chunks, recommend, retrieve

app = FastAPI(title="RAG Lab", description="RAG strategy benchmarking and corpus-aware recommendations")


class SourceText(BaseModel):
    source: str = Field(min_length=1, max_length=240)
    text: str = Field(min_length=1, max_length=2_000_000)


class CorpusRequest(BaseModel):
    files: list[SourceText] = Field(min_length=1, max_length=20)
    chunk_size: int = Field(default=600, ge=200, le=1600)
    overlap: int = Field(default=80, ge=0, le=300)

    @model_validator(mode="after")
    def validate_corpus(self):
        if self.overlap > self.chunk_size // 2:
            raise ValueError("Chunk overlap must be at most half the chunk size.")
        if sum(len(item.text) for item in self.files) > 2_000_000:
            raise ValueError("The total uploaded text must be 2 million characters or less.")
        return self


class BenchmarkRequest(CorpusRequest):
    questions: list[str] = Field(min_length=1, max_length=20)
    top_k: int = Field(default=4, ge=1, le=10)


class AskRequest(CorpusRequest):
    question: str = Field(min_length=2, max_length=800)
    top_k: int = Field(default=4, ge=1, le=10)


class TextUpload:
    def __init__(self, name: str, text: str):
        self.name = name
        self.text = text

    def getvalue(self) -> bytes:
        return self.text.encode("utf-8")


def make_chunks(request: CorpusRequest) -> list[dict]:
    uploads = [TextUpload(item.source, item.text) for item in request.files]
    return build_chunks(uploads, chunk_size=request.chunk_size, overlap=request.overlap)


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = "max-age=63072000"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home():
    return HTMLResponse(PAGE)


@app.get("/api/health")
async def health():
    return {"ok": True, "service": "rag-lab"}


@app.post("/api/parse")
async def parse_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF uploads use this endpoint.")
    payload = await file.read(4_000_001)
    if len(payload) > 4_000_000:
        raise HTTPException(status_code=413, detail="PDFs must be 4 MB or smaller.")
    try:
        reader = PdfReader(BytesIO(payload))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="This PDF could not be read.") from exc
    if not text:
        raise HTTPException(status_code=422, detail="No selectable text was found in this PDF.")
    return {"source": file.filename[:240], "text": text[:2_000_000]}


@app.post("/api/recommend")
async def get_recommendation(request: CorpusRequest):
    docs = make_chunks(request)
    advice = recommend(docs)
    return {"recommendation": advice, "documents": len({doc["source"] for doc in docs}), "chunks": len(docs)}


@app.post("/api/benchmark")
async def compare_strategies(request: BenchmarkRequest):
    questions = [question.strip() for question in request.questions if question.strip()]
    if not questions:
        raise HTTPException(status_code=422, detail="Add at least one non-empty benchmark question.")
    docs = make_chunks(request)
    results = benchmark(
        docs,
        questions,
        top_k=request.top_k,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
    )
    return {"results": results.to_dict(orient="records"), "recommendation": recommend(docs)}


@app.post("/api/ask")
async def ask_advisor(request: AskRequest):
    docs = make_chunks(request)
    advice = recommend(docs)
    answer, _ = retrieve(docs, request.question, request.top_k, advice)
    return {"answer": answer, "recommendation": advice}


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#f7f9f8"><title>RAG Lab — Strategy Workbench</title>
<style>
:root{color-scheme:light;--ink:#15232c;--muted:#62727d;--line:#e1e8e7;--teal:#08786e;--mint:#e8f5f1;--paper:#f7f9f8;--white:#fff;--red:#ae3e36}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}button,input,textarea{font:inherit}button{cursor:pointer}.shell{max-width:1220px;margin:auto;padding:34px 28px 70px}.top{display:flex;align-items:center;justify-content:space-between;margin-bottom:42px}.brand{font-weight:800;letter-spacing:-.04em;font-size:18px}.brand span{color:var(--teal)}.status{font:12px ui-monospace,monospace;color:var(--muted);border:1px solid var(--line);border-radius:99px;padding:7px 11px}.hero{max-width:740px;margin-bottom:32px}.eyebrow{color:var(--teal);font-size:11px;font-weight:800;letter-spacing:.17em;text-transform:uppercase}.hero h1{font-size:clamp(36px,5vw,58px);letter-spacing:-.065em;line-height:1.04;margin:10px 0 13px}.hero p,.muted{color:var(--muted)}.layout{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(300px,.85fr);gap:20px;align-items:start}.stack{display:grid;gap:18px}.card{background:var(--white);border:1px solid var(--line);border-radius:16px;padding:20px 22px;box-shadow:0 4px 18px #142b2006}.card h2{font-size:16px;letter-spacing:-.02em;margin:0 0 6px}.card p{margin:.25rem 0;color:var(--muted);font-size:13px}.drop{margin-top:16px;border:1px dashed #adc7c1;background:#f9fcfb;border-radius:12px;padding:18px;text-align:center}.drop input{max-width:100%}.small{font-size:12px;color:var(--muted)}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:17px}.metric{background:#f7f9f8;border:1px solid var(--line);border-radius:11px;padding:12px}.metric label{display:block;color:var(--muted);font-size:10px;letter-spacing:.08em;text-transform:uppercase;font-weight:700}.metric strong{display:block;margin-top:4px;font-size:18px;letter-spacing:-.04em}.controls{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:14px}.control label{display:block;font-size:12px;color:var(--muted);margin-bottom:4px}.control input{width:100%;accent-color:var(--teal)}textarea{width:100%;resize:vertical;min-height:94px;border:1px solid var(--line);border-radius:10px;padding:11px 12px;color:var(--ink);background:#fff;outline-color:var(--teal)}.primary{background:var(--teal);color:white;border:0;border-radius:9px;padding:10px 16px;font-weight:700}.primary:hover{background:#07675f}.secondary{background:#fff;color:var(--ink);border:1px solid var(--line);border-radius:9px;padding:9px 13px;font-weight:700}.actions{display:flex;gap:9px;align-items:center;margin-top:12px}.tag{display:inline-flex;background:var(--mint);color:var(--teal);border-radius:99px;padding:4px 10px;font-size:11px;font-weight:800}.advice h3{font-size:21px;line-height:1.2;letter-spacing:-.04em;margin:10px 0}.advice p{font-size:13px}.advice h4{font-size:12px;margin:18px 0 3px}.table-wrap{overflow:auto;margin-top:14px}table{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}th,td{text-align:left;padding:9px 8px;border-bottom:1px solid var(--line)}th{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}.chat{display:grid;gap:10px;margin-top:12px;max-height:430px;overflow:auto}.bubble{border-radius:12px;padding:11px 13px;font-size:13px;white-space:pre-wrap;overflow-wrap:anywhere}.bubble.user{background:#edf3f1;margin-left:34px}.bubble.assistant{background:#f7f9f8;border:1px solid var(--line);margin-right:14px}.chat-form{display:flex;gap:8px;margin-top:12px}.chat-form input{min-width:0;flex:1;border:1px solid var(--line);border-radius:9px;padding:10px 12px;outline-color:var(--teal)}.notice{color:var(--muted);font-size:12px;margin-top:11px}.error{color:var(--red)}.hidden{display:none}.strategy{display:flex;flex-wrap:wrap;gap:7px;margin-top:14px}.strategy span{border:1px solid var(--line);border-radius:99px;padding:5px 9px;font-size:11px;color:#465761}.foot{margin-top:22px;color:var(--muted);font-size:11px}@media(prefers-color-scheme:dark){:root{color-scheme:dark;--ink:#e6eeeb;--muted:#a0b0b3;--line:#2c3a3d;--teal:#46b8aa;--mint:#16322e;--paper:#0d1417;--white:#141e21;--red:#ff8176}.drop{background:#101b1d;border-color:#35524d}.metric{background:#101a1c}textarea,.chat-form input{background:#0f191b;color:var(--ink)}.secondary{background:#141e21;color:var(--ink)}.bubble.user{background:#1b2c2c}.bubble.assistant{background:#101a1c}.status{color:#bdc8c7}}
@media(max-width:850px){.layout{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}@media(max-width:520px){.shell{padding:20px 14px 48px}.top{margin-bottom:28px}.hero h1{font-size:40px}.card{padding:16px}.controls{grid-template-columns:1fr}.chat-form{align-items:stretch}.chat-form button{padding-inline:10px}}
</style></head>
<body><main class="shell">
<header class="top"><div class="brand"><span>◈</span> RAG Lab</div><div class="status">LOCAL HEURISTICS · NO DATA STORED</div></header>
<section class="hero"><div class="eyebrow">Retrieval evaluation studio</div><h1>Find the right RAG recipe for your data.</h1><p>Benchmark chunking, retrieval and sparse-distance strategies on your corpus. Ask the strategy advisor for an evidence-backed starting point.</p></section>
<div class="layout"><div class="stack">
<section class="card"><h2>1 · Add your corpus</h2><p>Upload text, Markdown, CSV, JSON, Python, HTML or text-based PDF files.</p><div class="drop"><input id="files" type="file" multiple accept=".txt,.md,.csv,.json,.py,.html,.pdf"><div id="file-status" class="small" aria-live="polite">Files stay in this browser session; nothing is saved.</div></div><div class="metrics"><div class="metric"><label>Documents</label><strong id="doc-count">0</strong></div><div class="metric"><label>Chunks</label><strong id="chunk-count">0</strong></div><div class="metric"><label>Data profile</label><strong id="data-type">—</strong></div><div class="metric"><label>Avg. chunk</label><strong id="avg-chunk">—</strong></div></div>
<div class="controls"><div class="control"><label for="chunk-size">Chunk size</label><input id="chunk-size" type="range" min="200" max="1600" step="100" value="600"><span id="size-label" class="small">600 characters</span></div><div class="control"><label for="overlap">Overlap</label><input id="overlap" type="range" min="0" max="300" step="20" value="80"><span id="overlap-label" class="small">80 characters</span></div><div class="control"><label for="top-k">Results per question</label><input id="top-k" type="number" min="1" max="10" value="4"></div></div>
<div class="actions"><button id="sample" class="secondary" type="button">Load sample corpus</button><button id="clear" class="secondary" type="button">Clear</button></div></section>
<section class="card"><h2>2 · Compare strategies</h2><p>Use questions you expect real users to ask. Without labeled relevance judgments, scores are term-overlap proxies.</p><label class="small" for="questions">Benchmark questions (one per line)</label><textarea id="questions">What is self-attention?
How does BERT create contextual embeddings?
What is the best distance metric for semantic search?</textarea><div class="actions"><button id="run" class="primary" type="button">Run comparison</button><span id="benchmark-status" class="small" aria-live="polite"></span></div><div id="winner" class="notice"></div><div id="results" class="table-wrap hidden"></div></section>
</div><div class="stack"><section class="card advice"><h2>Recommended starting point</h2><p>Corpus-aware configuration based on format signals.</p><div id="recommendation" class="notice">Upload documents or load the sample corpus to get a recommendation.</div><div class="strategy"><span>BM25</span><span>TF-IDF</span><span>Hybrid RRF</span><span>Cosine</span><span>Jaccard</span></div></section>
<section class="card"><h2>Strategy advisor</h2><p>Ask about your corpus or the benchmark. Answers cite retrieved source snippets.</p><div id="chat" class="chat" aria-live="polite"></div><form id="chat-form" class="chat-form"><input id="prompt" type="text" maxlength="800" placeholder="Why hybrid search for my data?" autocomplete="off" aria-label="Ask the strategy advisor"><button class="primary" type="submit">Ask</button></form></section>
<section class="card"><h2>What gets compared?</h2><p>Four chunk boundaries; BM25, TF-IDF, sparse term vectors, hybrid RRF and a lightweight rerank; cosine, dot product, Euclidean, Manhattan and Jaccard distances.</p><p class="notice">These are transparent lexical baselines, not neural embeddings. The advisor is heuristic-based and does not generate LLM answers. Validate with judged query/document pairs.</p></section></div></div>
<footer class="foot">RAG Lab · Uploads are processed per request and are not retained by the server.</footer></main>
<script>
const state={files:[],recommendation:null};const $=id=>document.getElementById(id);const size=()=>Number($("chunk-size").value);const overlap=()=>Math.min(Number($("overlap").value),Math.floor(size()/2));const topK=()=>Math.max(1,Math.min(10,Number($("top-k").value)||4));
const sample=[{source:"transformer-notes.md",text:"Self-attention lets each token compare itself with every other token in a sequence. Query, key, and value projections are used to compute contextual representations. Multi-head attention runs several attention operations in parallel.\n\nA Transformer block combines multi-head self-attention with a feed-forward network, residual connections, and layer normalization. Positional encodings provide order information."},{source:"nlp-guide.md",text:"BERT is a bidirectional encoder model pretrained with masked language modeling. Its contextual embeddings depend on both left and right context and are useful for classification and extraction. For relation extraction, BERT encodes a sentence and a classification head predicts the relation between marked entities."},{source:"retrieval-guide.md",text:"Cosine similarity compares the angle between normalized embedding vectors. It is a strong default for semantic search. Dot product is useful when the embedding model was trained for it. BM25 is a lexical ranking function that scores term overlap. Hybrid search combines lexical and dense rankings."}];
function payload(){return{files:state.files,chunk_size:size(),overlap:overlap()}}async function api(path,body){const response=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw Error(data.detail||"Request failed");return data}
function showAdvice(a){state.recommendation=a;$("data-type").textContent=a.data_type;$("avg-chunk").textContent=`${a.avg_chunk} chars`;$("recommendation").innerHTML=`<span class="tag">${escapeHtml(a.retrieval)}</span><h3>${escapeHtml(a.headline)}</h3><p>${escapeHtml(a.reason)}</p><h4>Chunking</h4><p>${escapeHtml(a.chunking)}</p><h4>Distance</h4><p>${escapeHtml(a.distance)}</p>`}
function escapeHtml(value){return String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]))}
function renderResults(rows){const wrap=$("results");wrap.innerHTML="";const table=document.createElement("table");table.innerHTML="<thead><tr><th>Chunking</th><th>Distance</th><th>Retrieval</th><th>Relevance</th><th>Precision@k</th><th>MRR</th><th>Latency</th></tr></thead>";const body=document.createElement("tbody");rows.slice(0,12).forEach(row=>{const tr=document.createElement("tr");[row.chunking,row.distance,row.retrieval,row.relevance,row["precision@k"],row.MRR,`${row["latency (ms)"]} ms`].forEach(value=>{const td=document.createElement("td");td.textContent=value;tr.appendChild(td)});body.appendChild(tr)});table.appendChild(body);wrap.appendChild(table);wrap.classList.remove("hidden")}
async function refreshCorpus(){if(!state.files.length){$("doc-count").textContent="0";$("chunk-count").textContent="0";$("data-type").textContent="—";$("avg-chunk").textContent="—";$("recommendation").textContent="Upload documents or load the sample corpus to get a recommendation.";return}try{const result=await api("/api/recommend",payload());$("doc-count").textContent=result.documents;$("chunk-count").textContent=result.chunks;showAdvice(result.recommendation)}catch(error){$("file-status").textContent=error.message;$("file-status").classList.add("error")}}
$("chunk-size").addEventListener("input",()=>{$("size-label").textContent=`${size()} characters`;if(Number($("overlap").value)>size()/2){$("overlap").value=overlap()}$("overlap-label").textContent=`${overlap()} characters`;refreshCorpus()});$("overlap").addEventListener("input",()=>{$("overlap-label").textContent=`${overlap()} characters`;refreshCorpus()});
$("files").addEventListener("change",async event=>{state.files=[];$("file-status").classList.remove("error");for(const file of event.target.files){if(file.size>4_000_000){$("file-status").textContent=`${file.name} exceeds the 4 MB file limit.`;$("file-status").classList.add("error");continue}try{let text;if(file.name.toLowerCase().endsWith(".pdf")){const form=new FormData();form.append("file",file);const response=await fetch("/api/parse",{method:"POST",body:form});const data=await response.json();if(!response.ok)throw Error(data.detail||"PDF could not be parsed");text=data.text}else{text=await file.text()}if(text.trim())state.files.push({source:file.name,text})}catch(error){$("file-status").textContent=error.message;$("file-status").classList.add("error")}}$("file-status").textContent=`${state.files.length} file(s) ready · files stay in this browser session.`;await refreshCorpus()});
$("sample").addEventListener("click",async()=>{state.files=sample;$("file-status").textContent="Sample research corpus loaded.";await refreshCorpus()});$("clear").addEventListener("click",()=>{state.files=[];$("files").value="";$("chat").replaceChildren();$("results").classList.add("hidden");$("winner").textContent="";$("file-status").textContent="Files stay in this browser session; nothing is saved.";refreshCorpus()});
$("run").addEventListener("click",async()=>{const questions=$("questions").value.split(/\n/).map(q=>q.trim()).filter(Boolean);if(!state.files.length){$("benchmark-status").textContent="Add documents first.";return}if(!questions.length){$("benchmark-status").textContent="Add at least one question.";return}$("run").disabled=true;$("benchmark-status").textContent="Comparing strategy combinations…";try{const result=await api("/api/benchmark",{...payload(),questions,top_k:topK()});renderResults(result.results);const best=result.results[0];$("winner").textContent=`Top observed: ${best.retrieval} · ${best.chunking} chunks · ${best.distance} · relevance ${best.relevance}.`;showAdvice(result.recommendation);$("benchmark-status").textContent=`${result.results.length} configurations compared.`}catch(error){$("benchmark-status").textContent=error.message}finally{$("run").disabled=false}});
$("chat-form").addEventListener("submit",async event=>{event.preventDefault();const question=$("prompt").value.trim();if(!question)return;if(!state.files.length){appendMessage("assistant","Upload documents or load the sample corpus before asking.");return}appendMessage("user",question);$("prompt").value="";try{const result=await api("/api/ask",{...payload(),question,top_k:topK()});showAdvice(result.recommendation);appendMessage("assistant",result.answer)}catch(error){appendMessage("assistant",error.message)}});function appendMessage(role,text){const div=document.createElement("div");div.className=`bubble ${role}`;div.innerHTML=role==="assistant"?escapeHtml(text).replace(/\*\*(.+?)\*\*/g,"<strong>$1</strong>").replace(/^- (.*)$/gm,"• $1").replace(/\n/g,"<br>"):escapeHtml(text);$("chat").appendChild(div);div.scrollIntoView({block:"nearest"})}
</script></body></html>'''
