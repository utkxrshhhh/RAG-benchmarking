"""Resolve which LLM backs the chatbot.

Free OpenAI-compatible providers are preferred when their key is set; otherwise calls go to Vercel AI Gateway.
Model IDs can be overridden with LLM_MODEL (chat + judge) and LLM_FAST_MODEL (per-strategy answers).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import ai
from ai.providers.openai import OpenAIChatCompletionsProtocol

Role = Literal["chat", "strategy", "judge"]


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    key_envs: tuple[str, ...]
    base_url: str
    model: str
    fast_model: str


PROVIDERS = (
    ProviderSpec(
        "Groq",
        ("GROQ_API_KEY",),
        "https://api.groq.com/openai/v1",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ),
    ProviderSpec(
        "Google Gemini",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ),
    ProviderSpec(
        "OpenRouter",
        ("OPENROUTER_API_KEY",),
        "https://openrouter.ai/api/v1",
        "meta-llama/llama-3.3-70b-instruct:free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ),
)

GATEWAY_MODELS = {
    "chat": "anthropic/claude-sonnet-5.5",
    "strategy": "anthropic/claude-haiku-5.5",
    "judge": "anthropic/claude-sonnet-5.5",
}


def _load_local_env() -> None:
    """The FastAPI dev server does not read .env files itself, so pick up the project's local env file."""
    env_file = Path(__file__).resolve().parent.parent / ".env.development.local"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and name and not name.startswith("#"):
            os.environ.setdefault(name.removeprefix("export ").strip(), value.strip().strip('"').strip("'"))


_load_local_env()


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _active() -> tuple[ProviderSpec, str] | None:
    custom_key, custom_url = _env("LLM_API_KEY"), _env("LLM_BASE_URL")
    if custom_key and custom_url:
        model = _env("LLM_MODEL") or "gpt-4o-mini"
        return ProviderSpec("Custom", ("LLM_API_KEY",), custom_url, model, model), custom_key
    for spec in PROVIDERS:
        for env_name in spec.key_envs:
            if key := _env(env_name):
                return spec, key
    return None


def model_for(role: Role) -> ai.Model:
    active = _active()
    if active is None:
        return ai.get_model(GATEWAY_MODELS[role])
    spec, key = active
    model_id = (_env("LLM_FAST_MODEL") or spec.fast_model) if role == "strategy" else (_env("LLM_MODEL") or spec.model)
    provider = ai.get_provider("openai", base_url=spec.base_url, api_key=key)
    return ai.Model(id=model_id, provider=provider, protocol=OpenAIChatCompletionsProtocol())


def describe() -> dict:
    active = _active()
    if active is None:
        return {"provider": "Vercel AI Gateway", "model": GATEWAY_MODELS["chat"], "free": False}
    spec, _ = active
    return {"provider": spec.name, "model": _env("LLM_MODEL") or spec.model, "free": spec.name != "Custom"}
