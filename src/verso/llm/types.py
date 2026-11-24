"""Requests, responses and configuration of model calls."""

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """One chat completion call. messages follow the OpenAI chat format (images as data-URI parts)."""

    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = 0.0
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    max_tokens_param: str = "max_tokens"
    send_seed: bool = True
    json_schema: dict[str, Any] | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    prompt_version: str = ""


class TokenLogprob(BaseModel):
    """One generated token with its log-probability and the most likely alternatives."""

    token: str
    logprob: float
    top: list[tuple[str, float]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Text and metadata of one completion; `cached` marks a response served from the cache."""

    text: str
    finish_reason: str | None = None
    model: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    logprobs: list[TokenLogprob] | None = None
    latency_s: float = 0.0
    cached: bool = False

    @property
    def truncated(self) -> bool:
        """The output hit the token limit (OpenAI-compatible "length"; Gemini reports MAX_TOKENS)."""
        return (self.finish_reason or "").lower() in ("length", "max_tokens")


class LLMConfig(BaseModel):
    """Where and how to call a model. The API key is read from the environment variable named here."""

    provider: str = "openai_compatible"
    model: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float | None = 0.0
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    max_tokens_param: str = "max_tokens"
    send_seed: bool = True
    structured_output: str = "json_schema"
    timeout_s: float = 120.0
    extra: dict[str, Any] = Field(default_factory=dict)
    cache_path: str | None = ".cache/llm.sqlite"
