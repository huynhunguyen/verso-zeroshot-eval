"""The MLLM solving a problem directly, from the diagram and the problem text (plan section 5.3.1; decisions D30-D32):
the request of the zero-shot baseline, and the answer the formal pipeline is weighed against in VERSO (decision D73).

One user turn: the diagram, then the prompt's instructions with the problem text inserted verbatim. The decoding is
the caller's (the developer's recommended sampling, D31). A response cut at the token limit gets one more call
that asks for the final answer only (D32). The same request gives the same cached response, so VERSO's direct
solution of a question is the baseline's response to it.

With answer_call (the plain prompt of a user who pastes the problem alone, which asks for no answer format), a
response without a JSON answer, or cut at the token limit, gets one more call in the same conversation that asks the
model to write the final answer of its solution as the JSON object, without solving again: the answer is then read
from the model's own words, whatever their form, rather than from the last number of the text.
"""

import hashlib
from pathlib import Path
from typing import Any

from ..core.types import DirectSolution
from ..llm import ChatResponse, LLMClient, LLMConfig, user_message
from .extraction import extract, to_result

FORCE_INSTRUCTION = "Stop reasoning. Write only the final JSON answer block."
ANSWER_INSTRUCTION = (
    "Write only the final answer of your solution above, without solving the problem again, as one JSON object inside "
    "a ```json code block, with these keys: \"status\": \"ok\" if the problem has an answer, \"no_solution\" if it "
    "has none; \"value\": the answer as a plain decimal number, when there is one numeric answer; \"values\": a list "
    "of plain decimal numbers, when there are several; \"unit\": the unit of the value or values, such as \"A\", "
    "\"V\", \"Ω\", \"μF\", \"W\", or \"\" when the answer has no unit; \"text\": the answer in words, only when it "
    "is not a number.")
FORCE_MAX_TOKENS = 256
# The seed of every direct call and the seconds one call may take (the zero-shot baseline's).
SEED = 0
TIMEOUT_S = 600.0


def load_prompt(directory: str | Path, name: str) -> tuple[str, str]:
    """(version, template) of a prompt file. The version carries a hash of the text, so any edit changes it."""
    text = (Path(directory) / f"{name}.txt").read_text(encoding="utf-8")
    return f"{name}@{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}", text


def build_messages(template: str, question: str, image) -> list[dict[str, Any]]:
    """The single user turn: the diagram, then the instructions with the problem text inserted verbatim."""
    return [user_message(template.replace("{question}", question), images=[image])]


def response_record(response: ChatResponse) -> dict[str, Any]:
    return {"text": response.text, "finish_reason": response.finish_reason, "model": response.model,
            "usage": response.usage, "latency_s": response.latency_s, "cached": response.cached,
            "truncated": response.truncated}


def _best(texts: list[tuple[str, str]]) -> tuple[str, str | None, dict[str, Any] | None]:
    """(method, source, extracted answer) from the responses, in order of preference: a JSON answer before a fallback
    number, and within each, the force-answer response before the first one."""
    found = [(source, extract(text)) for source, text in texts]
    for wanted in ("json", "fallback"):
        for source, extraction in found:
            if extraction.method == wanted:
                return wanted, source, extraction.answer
    return "none", None, None


def extraction_of(texts: list[tuple[str, str]]) -> dict[str, Any]:
    """The extraction record of the responses: {method, source, answer}."""
    method, source, extracted = _best(texts)
    return {"method": method, "source": source, "answer": extracted}


def _ask(client: LLMClient, config: LLMConfig, prompt_version: str, messages: list[dict[str, Any]],
         force_max_tokens: int, answer_call: bool = False) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """(record of one question, the answer extracted from it)."""
    record = {"response": None, "forced": None, "error": None}
    try:
        first = client.complete(messages, prompt_version=prompt_version, config=config)
        record["response"] = response_record(first)
        texts = [("first", first.text)]
        asked = answer_call and (first.truncated or extract(first.text).method != "json")
        if first.truncated or asked:
            instruction, suffix = (ANSWER_INSTRUCTION, "+answer") if answer_call else (FORCE_INSTRUCTION, "+force")
            follow_up = messages + [{"role": "assistant", "content": first.text},
                                    {"role": "user", "content": instruction}]
            forced = client.complete(follow_up, prompt_version=f"{prompt_version}{suffix}", config=config,
                                     max_tokens=force_max_tokens)
            record["forced"] = response_record(forced)
            texts.insert(0, ("forced", forced.text))
    except Exception as ex:  # noqa: BLE001 — a failed call is recorded and graded wrong; a rerun resumes from the cache
        record["error"] = f"{type(ex).__name__}: {ex}"
        return {**record, "extraction": extraction_of([])}, None
    method, source, extracted = _best(texts)
    return {**record, "extraction": {"method": method, "source": source, "answer": extracted}}, extracted


def answer_question(client: LLMClient, config: LLMConfig, prompt_version: str, messages: list[dict[str, Any]],
                    force_max_tokens: int = FORCE_MAX_TOKENS, answer_call: bool = False) -> dict[str, Any]:
    """Record of one question: the response, the force-answer call when the response was cut (with answer_call, the
    call asking for the answer whenever the response gives none as JSON), the answer."""
    return _ask(client, config, prompt_version, messages, force_max_tokens, answer_call)[0]


def solve_directly(question: str, image, client: LLMClient, config: LLMConfig, prompt: tuple[str, str],
                   force_max_tokens: int = FORCE_MAX_TOKENS, answer_call: bool = False) -> DirectSolution:
    """The direct solution of a problem with a (version, template) prompt."""
    version, template = prompt
    record, extracted = _ask(client, config, version, build_messages(template, question, image), force_max_tokens,
                             answer_call)
    return DirectSolution(**record, result=to_result(extracted))
