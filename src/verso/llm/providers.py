"""Model providers. A provider turns a ChatRequest into a ChatResponse; caching is the client's job."""

import logging
import os
import re
import time
from collections.abc import Callable
from typing import Any, Protocol

from ..core.errors import VersoError
from .types import ChatRequest, ChatResponse, TokenLogprob

log = logging.getLogger(__name__)

STRUCTURED_OUTPUT_MODES = ("json_schema", "guided_json", "none")
MAX_RETRIES = 5
RETRY_BASE_S = 2.0


class ProviderError(VersoError):
    """The provider could not be configured or the call failed."""


class Provider(Protocol):
    """What LLMClient needs from a provider."""

    @property
    def provider_id(self) -> str:
        """Namespace of the response cache, so responses of different endpoints are never mixed."""

    def complete(self, request: ChatRequest) -> ChatResponse:
        """One chat-completion call; caching is the client's job."""


def build_params(request: ChatRequest, structured_output: str = "json_schema") -> dict[str, Any]:
    """Keyword arguments of chat.completions.create for a request."""
    if structured_output not in STRUCTURED_OUTPUT_MODES:
        raise ProviderError(f"unknown structured_output mode: {structured_output}")
    params: dict[str, Any] = {"model": request.model, "messages": request.messages}
    extra = dict(request.extra)
    extra_body = dict(extra.pop("extra_body", {}))
    params.update(extra)
    for key in ("temperature", "top_p", "presence_penalty"):
        if getattr(request, key) is not None:
            params[key] = getattr(request, key)
    if request.send_seed and request.seed is not None:
        params["seed"] = request.seed
    if request.max_tokens is not None:
        params[request.max_tokens_param] = request.max_tokens
    if request.top_k is not None:
        extra_body["top_k"] = request.top_k
    if request.logprobs:
        params["logprobs"] = True
        if request.top_logprobs:
            params["top_logprobs"] = request.top_logprobs
    if request.json_schema is not None:
        if structured_output == "json_schema":
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": request.json_schema, "strict": True},
            }
        elif structured_output == "guided_json":
            extra_body["guided_json"] = request.json_schema
        else:
            raise ProviderError("this model does not enforce JSON schemas (structured_output = none)")
    if extra_body:
        params["extra_body"] = extra_body
    return params


def parse_logprobs(choice: Any) -> list[TokenLogprob] | None:
    """Token log-probabilities of an OpenAI response choice, or None when the server returned none."""
    content = getattr(getattr(choice, "logprobs", None), "content", None)
    if not content:
        return None
    return [TokenLogprob(token=t.token, logprob=t.logprob, top=[(a.token, a.logprob) for a in (t.top_logprobs or [])])
            for t in content]


_CONTEXT_FULL = re.compile(r"maximum context length is (\d+) tokens and your request has (\d+) input tokens")
MIN_OUTPUT_TOKENS = 256
CONTEXT_MARGIN = 8


def output_room(message: str) -> int | None:
    """The output tokens that fit the context a refusal message reports, or None."""
    m = _CONTEXT_FULL.search(message)
    if not m:
        return None
    room = int(m.group(1)) - int(m.group(2)) - CONTEXT_MARGIN
    return room if room >= MIN_OUTPUT_TOKENS else None


class OpenAICompatibleProvider:
    """Any endpoint of the OpenAI chat-completions API: OpenAI, Gemini OpenAI-compat, OpenRouter, vLLM."""

    def __init__(self, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY", timeout_s: float = 120.0,
                 structured_output: str = "json_schema") -> None:
        from openai import OpenAI

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ProviderError(f"environment variable {api_key_env} is not set")
        self.base_url = base_url
        self.structured_output = structured_output
        # Disable SDK-level silent retries so our explicit backoff controls 429/5xx cleanly.
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)

    @property
    def provider_id(self) -> str:
        """Cache namespace: the endpoint's base URL."""
        return f"openai_compatible:{self.base_url or 'default'}"

    def complete(self, request: ChatRequest) -> ChatResponse:
        """One chat-completion call with polite exponential backoff on transient rate-limit / server errors."""
        from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, RateLimitError

        params = build_params(request, self.structured_output)
        start = time.perf_counter()
        out = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                out = self._client.chat.completions.create(**params)
                break
            except BadRequestError as ex:
                room = output_room(str(ex))
                if room is not None and request.max_tokens_param in params:
                    out = self._client.chat.completions.create(
                        **{**params, request.max_tokens_param: min(room, params[request.max_tokens_param])}
                    )
                    break
                raise
            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as ex:
                if attempt >= MAX_RETRIES:
                    raise
                delay = RETRY_BASE_S * (2 ** attempt)
                log.warning("transient API error (%s); retrying in %.1fs (%d/%d)",
                            type(ex).__name__, delay, attempt + 1, MAX_RETRIES)
                time.sleep(delay)
        assert out is not None
        choice = out.choices[0]
        usage = out.usage.model_dump(exclude_none=True) if out.usage else {}
        return ChatResponse(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            model=out.model or request.model,
            usage={k: v for k, v in usage.items() if isinstance(v, int)},
            logprobs=parse_logprobs(choice),
            latency_s=time.perf_counter() - start,
        )


class FakeProvider:
    """Deterministic provider for tests: responder(request) -> text."""

    provider_id = "fake"

    def __init__(self, responder: Callable[[ChatRequest], str], finish_reason: str = "stop") -> None:
        self.responder = responder
        self.finish_reason = finish_reason
        self.calls = 0

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        return ChatResponse(text=self.responder(request), finish_reason=self.finish_reason, model=request.model)
