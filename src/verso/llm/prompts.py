"""Prompts of the blocks that call a model: instructions (configs/prompts/<name>.txt) and worked examples written from
self-authored problems (configs/prompts/<name>_examples.json, rule P-05); and the JSON object of a reply."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXAMPLES_SUFFIX = "_examples.json"
_FENCED = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.S)


def extract_json(reply: str) -> str:
    """The JSON object of a reply: the body of a fenced block when there is one, otherwise the text from the first
    { to the last }; the reply itself when it has no braces (the checks then report it is not JSON)."""
    fenced = _FENCED.search(reply)
    if fenced:
        return fenced.group(1)
    start, end = reply.find("{"), reply.rfind("}")
    return reply[start:end + 1] if 0 <= start < end else reply


@dataclass(frozen=True)
class Prompt:
    """Instructions and worked examples; version carries a hash of both, so any edit changes it."""

    version: str
    instructions: str
    examples: tuple[dict[str, Any], ...] = ()


def load_prompt(directory: str | Path, name: str) -> Prompt:
    """The prompt `name` of a prompts directory: <name>.txt and, when present, <name>_examples.json."""
    directory = Path(directory)
    instructions = (directory / f"{name}.txt").read_text(encoding="utf-8")
    path = directory / f"{name}{EXAMPLES_SUFFIX}"
    examples = tuple(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else ()
    digest = hashlib.sha256((instructions + json.dumps(examples, ensure_ascii=False, sort_keys=True)).encode("utf-8"))
    return Prompt(f"{name}@{digest.hexdigest()[:12]}", instructions, examples)
