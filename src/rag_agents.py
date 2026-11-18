from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Literal

import ai
from pydantic import BaseModel, Field, field_validator

from src.benchmark_engine import CHUNKERS, DISTANCES, STRATEGIES, _tokens, passage_quality, retrieve_with

CHAT_MODEL_ID = "anthropic/claude-sonnet-5.5"
STRATEGY_MODEL_ID = "anthropic/claude-haiku-5.5"
JUDGE_MODEL_ID = "anthropic/claude-sonnet-5.5"
MAX_MEMORIES = 20
MAX_AGENT_SEARCHES = 3
CHAT_TIMEOUT_SECONDS = 40
STRATEGY_TIMEOUT_SECONDS = 30
JUDGE_TIMEOUT_SECONDS = 30


class StrategyConfig(BaseModel):
    retrieval: str
    chunking: str = "recursive"
    distance: str = "cosine"

    @field_validator("retrieval")
    @classmethod
    def known_retrieval(cls, value: str) -> str:
        if value not in STRATEGIES:
            raise ValueError(f"Retrieval must be one of: {', '.join(STRATEGIES)}.")
        return value

    @field_validator("chunking")
    @classmethod
    def known_chunking(cls, value: str) -> str:
        if value not in CHUNKERS:
            raise ValueError(f"Chunking must be one of: {', '.join(CHUNKERS)}.")
        return value

    @field_validator("distance")
    @classmethod
    def known_distance(cls, value: str) -> str:
        if value not in [*DISTANCES, "n/a (BM25)"]:
            raise ValueError(f"Distance must be one of: {', '.join(DISTANCES)}.")
        return value

    def label(self) -> str:
        distance = "" if self.retrieval == "BM25 keyword" else f" · {self.distance}"
        return f"{self.retrieval} · {self.chunking}{distance}"


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4_000)


class CandidateScore(BaseModel):
    strategy: str
    score: float
    note: str = ""


class Verdict(BaseModel):
    winner: str
    reason: str
    scores: list[CandidateScore]
    memories: list[str] = []


def default_config(advice: dict) -> StrategyConfig:
    retrieval = advice.get("retrieval", "Hybrid RRF")
    return StrategyConfig(retrieval=retrieval if retrieval in STRATEGIES else "Hybrid RRF")


FIRST_PERSON = re.compile(r"^(?:the user|i)(?:'m| am| is)?\s+", re.I)


def memory_key(fact: str) -> str:
    text = FIRST_PERSON.sub("", fact.casefold().rstrip(". "))
    return re.sub(r"\b(prefers|likes|wants)\b", lambda m: m.group(1)[:-1], text)


def merge_memories(existing: list[str], new: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for fact in [*existing, *new]:
        fact = " ".join(fact.split())[:300]
        key = memory_key(fact)
        if fact and key not in seen:
            seen.add(key)
            merged.append(fact)
    return merged[-MAX_MEMORIES:]


MEMORY_PATTERNS = [
    (re.compile(r"\b(?:please\s+)?remember(?:\s+that)?\s+(.{3,240})", re.I), "{0}"),
    (re.compile(r"\bmy name is\s+([A-Za-z][\w\- ]{0,40})", re.I), "The user's name is {0}"),
    (re.compile(r"\bi (?:prefer|like|want)\s+(.{3,200})", re.I), "The user prefers {0}"),
    (re.compile(r"\bi(?:'m| am) (?:working on|building)\s+(.{3,200})", re.I), "The user is working on {0}"),
]


def extract_memories(message: str) -> list[str]:
    facts = []
    for pattern, template in MEMORY_PATTERNS:
        match = pattern.search(message)
        if match:
            facts.append(template.format(match.group(1).strip().rstrip(".!?")))
    return facts


def memory_block(memories: list[str]) -> str:
    if not memories:
        return "Long-term memory: (empty)."
    return "Long-term memory about this user and session:\n" + "\n".join(f"- {fact}" for fact in memories)


def history_messages(history: list[ChatTurn]) -> list[Any]:
    messages = []
    for turn in history:
        messages.append(ai.user_message(turn.content) if turn.role == "user" else ai.assistant_message(turn.content))
    return messages


def contextual_query(question: str, history: list[ChatTurn]) -> str:
    """Expand short follow-ups ("what about BERT?") with the previous user turn so retrieval keeps context."""
    previous = next((turn.content for turn in reversed(history) if turn.role == "user"), "")
    if previous and len(_tokens(question)) <= 6:
        return f"{question} {previous}"
    return question


def format_passages(passages: list[dict]) -> str:
    return "\n\n".join(
        f'[{index}] Source: {passage["source"]}\nPassage: {passage["text"][:900]}'
        for index, passage in enumerate(passages, start=1)
    )


def as_sources(passages: list[dict]) -> list[dict]:
    return [
        {"source": passage["source"], "snippet": passage["text"][:900], "score": passage.get("score")}
        for passage in passages
    ]


async def collect_text(stream: Any) -> str:
    async for _event in stream:
        pass
    return str(stream.output).strip()


# ---------- Chatbot agent ----------


async def run_chat_agent(
    *,
    question: str,
    history: list[ChatTurn],
    memories: list[str],
    docs: list[dict],
    config: StrategyConfig,
    top_k: int,
    chunk_size: int,
    overlap: int,
    advice: dict,
    leaderboard: list[dict],
) -> tuple[str, list[dict], list[str]]:
    retrieved: dict[str, dict] = {}
    learned: list[str] = []
    search_calls = 0

    @ai.tool
    async def search_corpus(query: str) -> str:
        """Search the user's uploaded corpus with the benchmark-selected retrieval strategy."""
        nonlocal search_calls
        if search_calls >= MAX_AGENT_SEARCHES:
            return "Search limit reached. Answer with the evidence already returned."
        search_calls += 1
        passages = retrieve_with(
            docs, query, top_k, config.retrieval, config.distance, config.chunking, chunk_size, overlap
        )
        if not passages:
            return "No relevant passages were found in the uploaded files."
        for passage in passages:
            retrieved[passage["id"]] = passage
        return format_passages(passages)

    @ai.tool
    async def remember(fact: str) -> str:
        """Save a durable fact about the user or their goals (name, preferences, project) to long-term memory."""
        learned.append(fact)
        return "Saved to memory."

    leaderboard_text = "\n".join(
        f"- {row.get('retrieval')} · {row.get('chunking')} · {row.get('distance')}: relevance {row.get('relevance')}, "
        f"MRR {row.get('MRR')}, precision@k {row.get('precision@k')}, {row.get('latency (ms)')} ms"
        for row in leaderboard[:5]
    ) or "No benchmark has been run yet."

    system = (
        "You are RAG Lab's conversational research agent. You chat naturally, remember context, and answer "
        "questions about the user's uploaded corpus.\n"
        "- For any factual question about the corpus, call search_corpus first and ground claims only in returned "
        "passages. Cite filenames in square brackets, e.g. [guide.pdf]. If evidence is missing, say so.\n"
        "- For greetings or small talk, reply briefly without searching.\n"
        "- When the user shares a durable fact (name, preference, goal), call remember once with a short sentence.\n"
        "- Use the conversation history and memory to resolve follow-up questions.\n"
        f"Active retrieval strategy (selected from the benchmark): {config.label()}.\n"
        f"Corpus profile: {advice['data_type']}. Heuristic recommendation: {advice['retrieval']}.\n"
        f"Top benchmark results:\n{leaderboard_text}\n"
        f"{memory_block(memories)}"
    )
    messages = [ai.system_message(system), *history_messages(history), ai.user_message(question)]

    agent = ai.Agent(tools=[search_corpus, remember])
    async with asyncio.timeout(CHAT_TIMEOUT_SECONDS):
        async with agent.run(ai.get_model(CHAT_MODEL_ID), messages) as stream:
            answer = await collect_text(stream)

    return answer, as_sources(list(retrieved.values())), learned


# ---------- Strategy arena ----------


def strategy_configs(base: StrategyConfig) -> list[StrategyConfig]:
    distance = base.distance if base.distance in DISTANCES else "cosine"
    return [
        StrategyConfig(
            retrieval=strategy,
            chunking=base.chunking,
            distance="n/a (BM25)" if strategy == "BM25 keyword" else distance,
        )
        for strategy in STRATEGIES
    ]


def extractive_answer(question: str, passages: list[dict]) -> str:
    """Local fallback: pick the passage sentences that best overlap the question."""
    if not passages:
        return "This strategy did not retrieve any relevant passages for the question."
    qterms = set(_tokens(question))
    sentences = []
    for passage in passages:
        for sentence in re.split(r"(?<=[.!?])\s+", passage["text"]):
            overlap = len(qterms & set(_tokens(sentence)))
            if overlap and 20 <= len(sentence) <= 600:
                sentences.append((overlap, sentence.strip(), passage["source"]))
    if not sentences:
        top = passages[0]
        return f'{top["text"][:420].strip()}… [{top["source"]}]'
    sentences.sort(key=lambda item: item[0], reverse=True)
    picked, used = [], set()
    for _, sentence, source in sentences:
        if sentence not in used:
            used.add(sentence)
            picked.append(f"{sentence} [{source}]")
        if len(picked) == 3:
            break
    return " ".join(picked)


async def answer_with_strategy(
    question: str, history: list[ChatTurn], memories: list[str], config: StrategyConfig, passages: list[dict]
) -> str:
    system = (
        "You answer questions using ONLY the retrieved passages below. Be concise (under 160 words). "
        "Cite filenames in square brackets. If the passages do not answer the question, say so plainly.\n"
        f"{memory_block(memories)}\n\nRetrieved passages ({config.label()}):\n"
        f"{format_passages(passages) or 'No passages were retrieved.'}"
    )
    messages = [ai.system_message(system), *history_messages(history[-6:]), ai.user_message(question)]
    async with asyncio.timeout(STRATEGY_TIMEOUT_SECONDS):
        async with ai.stream(ai.get_model(STRATEGY_MODEL_ID), messages) as stream:
            return await collect_text(stream)


def local_verdict(candidates: list[dict]) -> dict:
    """Judge fallback: rank by passage coverage + relevance, breaking ties by retrieval latency."""
    for candidate in candidates:
        metrics = candidate["metrics"]
        candidate["judge_score"] = round(10 * (0.6 * metrics["coverage"] + 0.4 * metrics["relevance"]), 1)
        candidate["judge_note"] = f'Covers {round(metrics["coverage"] * 100)}% of question terms.'
    best = max(candidates, key=lambda item: (item["judge_score"], -item["latency_ms"]))
    return {
        "winner": best["strategy"],
        "reason": (
            f'{best["strategy"]} retrieved passages covering the most question terms '
            f'({round(best["metrics"]["coverage"] * 100)}% coverage, relevance {best["metrics"]["relevance"]}). '
            "Scored locally with lexical overlap because the LLM judge was unavailable."
        ),
        "judge": "local-heuristic",
        "memories": [],
    }


async def llm_verdict(question: str, candidates: list[dict], memories: list[str]) -> dict:
    listing = "\n\n".join(
        f'### Strategy: {item["strategy"]}\nConfig: {item["label"]}\n'
        f'Retrieval metrics: coverage {item["metrics"]["coverage"]}, relevance {item["metrics"]["relevance"]}, '
        f'{len(item["sources"])} passages, {item["latency_ms"]} ms\n'
        f'Answer:\n{item["answer"]}\n'
        f'Evidence:\n{format_passages([{"source": s["source"], "text": s["snippet"][:400]} for s in item["sources"]]) or "none"}'
        for item in candidates
    )
    system = (
        "You are the RAG Lab judge agent. Compare candidate answers produced by different retrieval strategies for "
        "the same question. Score each from 0 to 10 for groundedness in its evidence, completeness, directness and "
        "correct citations. Penalize claims not supported by the evidence. Pick exactly one winner using the exact "
        "strategy name. Give a 1-2 sentence reason. Also extract any durable facts the user revealed about "
        "themselves in the question (name, preferences, goals) into memories; otherwise return an empty list.\n"
        f"{memory_block(memories)}"
    )
    messages = [ai.system_message(system), ai.user_message(f"Question: {question}\n\n{listing}")]
    async with asyncio.timeout(JUDGE_TIMEOUT_SECONDS):
        async with ai.stream(ai.get_model(JUDGE_MODEL_ID), messages, output_type=Verdict) as stream:
            async for _event in stream:
                pass
            verdict: Verdict = stream.output

    names = {item["strategy"] for item in candidates}
    if verdict.winner not in names:
        raise ValueError(f"Judge picked an unknown strategy: {verdict.winner}")
    scores = {score.strategy: score for score in verdict.scores}
    for item in candidates:
        score = scores.get(item["strategy"])
        item["judge_score"] = round(min(10.0, max(0.0, score.score)), 1) if score else None
        item["judge_note"] = score.note if score else ""
    return {"winner": verdict.winner, "reason": verdict.reason, "judge": "llm", "memories": verdict.memories}


async def run_strategy_arena(
    *,
    question: str,
    history: list[ChatTurn],
    memories: list[str],
    docs: list[dict],
    base: StrategyConfig,
    top_k: int,
    chunk_size: int,
    overlap: int,
) -> dict:
    query = contextual_query(question, history)
    candidates = []
    for config in strategy_configs(base):
        start = time.perf_counter()
        passages = retrieve_with(
            docs, query, top_k, config.retrieval, config.distance, config.chunking, chunk_size, overlap
        )
        candidates.append({
            "strategy": config.retrieval,
            "label": config.label(),
            "config": config.model_dump(),
            "passages": passages,
            "sources": as_sources(passages),
            "metrics": passage_quality(query, passages),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        })

    results = await asyncio.gather(
        *(
            answer_with_strategy(question, history, memories, StrategyConfig(**item["config"]), item["passages"])
            for item in candidates
        ),
        return_exceptions=True,
    )
    notices = []
    for item, result in zip(candidates, results):
        if isinstance(result, BaseException) or not result:
            item["answer"] = extractive_answer(question, item["passages"])
            item["mode"] = "extractive-fallback"
            if isinstance(result, BaseException):
                notices.append(result)
        else:
            item["answer"] = result
            item["mode"] = "llm"
        del item["passages"]

    try:
        if all(item["mode"] != "llm" for item in candidates):
            raise RuntimeError("No LLM answers to judge.")
        verdict = await llm_verdict(question, candidates, memories)
    except Exception as exc:
        notices.append(exc)
        verdict = local_verdict(candidates)

    winner = next(item for item in candidates if item["strategy"] == verdict["winner"])
    candidates.sort(key=lambda item: (item["strategy"] != verdict["winner"], -(item.get("judge_score") or 0)))
    return {"winner": winner, "verdict": verdict, "candidates": candidates, "errors": notices}
