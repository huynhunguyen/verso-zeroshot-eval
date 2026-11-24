"""The single entry point for calling LLMs and VLMs.

Every model call in VERSO goes through LLMClient, which caches responses on disk (a rerun gives the same
answer without a new call) and records the model, prompt version, finish reason, token usage and latency.
The model itself is chosen by a profile (configs/models/<name>.yaml): by default Qwen3-VL-8B-Instruct or
Qwen3-VL-32B-Instruct served by vLLM on the lab GPU server. Providers speak the OpenAI-compatible chat API,
which also covers OpenRouter and Gemini's compatibility endpoint; FakeProvider serves tests without network.

    profile = load_profile("qwen3-vl-8b-instruct", profiles_dir)
    client = client_from_config(config_from_profile(profile, base_url="http://<server>:8000/v1"))
    response = client.complete([user_message("Which symbol is this?", images=[crop_path])], logprobs=True, top_logprobs=5)
"""

from .cache import ResponseCache
from .client import LLMClient, client_from_config
from .messages import assistant_message, image_part, system_message, user_message
from .models import DecodingSettings, ModelProfile, ProfileError, available_profiles, config_from_profile, load_profile
from .prompts import Prompt, load_prompt
from .providers import FakeProvider, OpenAICompatibleProvider, Provider, ProviderError, build_params
from .types import ChatRequest, ChatResponse, LLMConfig, TokenLogprob

__all__ = ["ChatRequest", "ChatResponse", "DecodingSettings", "FakeProvider", "LLMClient", "LLMConfig", "ModelProfile",
           "OpenAICompatibleProvider", "ProfileError", "Prompt", "Provider", "ProviderError", "ResponseCache",
           "TokenLogprob", "assistant_message", "available_profiles", "build_params", "client_from_config",
           "config_from_profile", "image_part", "load_profile", "load_prompt", "system_message", "user_message"]
