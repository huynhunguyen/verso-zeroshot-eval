"""Model profiles kept in configs/models/<name>.yaml."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ..core.errors import VersoError
from .types import LLMConfig


class ProfileError(VersoError):
    """A model profile is missing or invalid."""


DECODING_MODES = ("default", "recommended")


class DecodingSettings(BaseModel):
    """Sampling settings published by a model's developer (model card or API guide).

    temperature may be null for reasoning models (e.g. OpenAI o3) that do not accept a custom temperature.
    extra holds provider-specific parameters sent verbatim (e.g. reasoning_effort).
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    source: str = ""


class ModelProfile(BaseModel):
    """Contents of one configs/models/<name>.yaml file."""

    name: str
    family: str
    hf_id: str = ""
    served_name: str
    parameters_b: float | None = None
    vision: bool = True
    logprobs: bool = False
    structured_output: str = "none"
    thinking: bool = False
    max_model_len: int = 128000
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens_param: str = "max_tokens"
    send_seed: bool = True
    temperature: float | None = 0.0
    top_p: float | None = None
    max_tokens: int | None = 8192
    recommended_decoding: DecodingSettings | None = None
    serve_args: list[str] = Field(default_factory=list)
    notes: str = ""


def load_profile(name_or_path: str | Path, profiles_dir: str | Path | None = None) -> ModelProfile:
    """Profile from a YAML file path, or by name from profiles_dir."""
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        if profiles_dir is None:
            raise ProfileError(f"profile {name_or_path!r} given by name but no profiles directory")
        path = Path(profiles_dir) / f"{name_or_path}.yaml"
    if not path.exists():
        raise ProfileError(f"model profile not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return ModelProfile(**data)
    except Exception as ex:  # noqa: BLE001
        raise ProfileError(f"invalid model profile {path}: {ex}") from ex


def available_profiles(profiles_dir: str | Path) -> list[str]:
    """Names of the profiles in a directory."""
    return sorted(p.stem for p in Path(profiles_dir).glob("*.yaml"))


def config_from_profile(profile: ModelProfile, base_url: str | None = None, api_key_env: str | None = None,
                        cache_path: str | None = ".cache/llm.sqlite", decoding: str = "default",
                        **overrides: Any) -> LLMConfig:
    """LLMConfig for a profile; base_url and api_key_env override the profile's endpoint."""
    if decoding not in DECODING_MODES:
        raise ProfileError(f"unknown decoding mode: {decoding}")
    if decoding == "recommended":
        if profile.recommended_decoding is None:
            raise ProfileError(f"profile {profile.name} has no recommended_decoding")
        sampling = profile.recommended_decoding.model_dump(exclude={"source"})
    else:
        sampling = {"temperature": profile.temperature, "top_p": profile.top_p}
    settings = {
        "model": profile.served_name,
        "base_url": base_url or profile.base_url,
        "api_key_env": api_key_env or profile.api_key_env,
        "max_tokens": profile.max_tokens,
        "max_tokens_param": profile.max_tokens_param,
        "send_seed": profile.send_seed,
        "structured_output": profile.structured_output,
        "cache_path": cache_path,
        **sampling,
    }
    return LLMConfig(**{**settings, **overrides})
