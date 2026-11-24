"""LLMClient: provider plus response cache."""

import logging
from typing import Any

from .cache import ResponseCache, cache_key
from .providers import OpenAICompatibleProvider, Provider, ProviderError
from .types import ChatRequest, ChatResponse, LLMConfig

log = logging.getLogger(__name__)


class LLMClient:
    """Sends chat requests through a provider, with an optional on-disk response cache."""

    def __init__(self, provider: Provider, cache: ResponseCache | None = None, config: LLMConfig | None = None) -> None:
        self.provider = provider
        self.cache = cache
        self.config = config

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Response to the request, from the cache when the same request was answered before."""
        key = cache_key(self.provider.provider_id, request)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return ChatResponse(**{**hit, "cached": True})
        response = self.provider.complete(request)
        if response.truncated:
            log.warning("model %s stopped at the token limit (prompt %s)", request.model, request.prompt_version)
        if self.cache is not None:
            self.cache.put(key, response.model_dump(mode="json", exclude={"cached"}))
        return response

    def complete(self, messages: list[dict[str, Any]], prompt_version: str = "", config: LLMConfig | None = None,
                 **overrides: Any) -> ChatResponse:
        """Send messages with the decoding settings of an LLMConfig (the client's own by default)."""
        config = config or self.config
        if config is None:
            raise ProviderError("no LLMConfig given to build the request")
        decoding = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "presence_penalty": config.presence_penalty,
            "seed": config.seed,
            "max_tokens": config.max_tokens,
            "max_tokens_param": config.max_tokens_param,
            "send_seed": config.send_seed,
        }
        request = ChatRequest(model=config.model, messages=messages, prompt_version=prompt_version,
                              **{"extra": dict(config.extra), **decoding, **overrides})
        return self.chat(request)


def client_from_config(config: LLMConfig) -> LLMClient:
    """Client for an LLMConfig: an OpenAI-compatible provider, plus a response cache when cache_path is set."""
    if config.provider != "openai_compatible":
        raise ProviderError(f"unknown provider: {config.provider}")
    provider = OpenAICompatibleProvider(config.base_url, config.api_key_env, config.timeout_s, config.structured_output)
    cache = ResponseCache(config.cache_path) if config.cache_path else None
    return LLMClient(provider, cache, config)
