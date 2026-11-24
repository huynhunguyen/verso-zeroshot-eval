"""Final answer of a free-form model response (the zero-shot baseline, plan section 5.3.1, decision D30, and the
direct solution inside VERSO, decision D73).

The prompt asks the model to end with a JSON object: status, value or values, unit, text. extract() reads the
last such object in the response; when there is none, it falls back to the last number written in the text,
with the unit written after it when that unit can be read. to_result() turns the answer into a SolveResult in
the grader's units, so that the shared grader grades it like a VERSO result. A unit the grader would not convert
makes the answer unreadable; it is never silently read at scale 1.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

from ..core.types import SolveResult, SolveStatus
from ..core.units import BASE_UNITS, PREFIX, read_answer

METHODS = ("json", "fallback", "none")
ANSWER_KEYS = ("status", "value", "values", "unit", "text")
# Units the grader compares at scale 1 besides the SI base units (dataset answers use all of them; siemens, S, in
# the external set CircuitSense).
SCALE_ONE_UNITS = ("", "-", "—", "lần", "VA", "VAR", "%", "rad", "°", "g", "V^2", "A/s", "S")
_ALIASES = {"ohm": "Ω", "ohms": "Ω", "Ohm": "Ω", "Ohms": "Ω", "var": "VAR", "VAr": "VAR", "Var": "VAR",
            "deg": "°", "times": "lần"}
# A number not glued to a label such as R1, and the word written right after it.
_NUMBER = re.compile(r"(?<![\w.,])(-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)(?:\s*([^\s.,;:()]+))?")


class UnitError(ValueError):
    """A unit the grader cannot convert."""


@dataclass(frozen=True)
class Extraction:
    method: str
    answer: dict[str, Any] | None


def normalize_unit(unit: str | None) -> str:
    """The unit as the grader writes it ('uF' -> 'μF', 'ohm' -> 'Ω'); UnitError when the grader cannot convert it."""
    unit = (unit or "").strip()
    unit = _ALIASES.get(unit, unit)
    if len(unit) > 1 and unit[0] == "u" and unit[1:] in BASE_UNITS:
        unit = "μ" + unit[1:]
    if unit in SCALE_ONE_UNITS or unit in BASE_UNITS:
        return unit
    if len(unit) > 1 and unit[0] in PREFIX and unit[1:] in BASE_UNITS:
        return unit
    raise UnitError(f"unit not recognized: {unit!r}")


def _last_answer_object(text: str) -> dict[str, Any] | None:
    """The JSON object with an answer key that ends last in the text (the outer one when objects are nested)."""
    decoder = json.JSONDecoder()
    best, best_rank = None, None
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        rank = (end, -start)
        if isinstance(obj, dict) and any(k in obj for k in ANSWER_KEYS) and (best_rank is None or rank > best_rank):
            best, best_rank = obj, rank
    return best


def extract(text: str) -> Extraction:
    """The final answer of a response: its last JSON answer object, else its last number."""
    text = text or ""
    obj = _last_answer_object(text)
    if obj is not None:
        return Extraction("json", obj)
    numbers = _NUMBER.findall(text)
    if not numbers:
        return Extraction("none", None)
    value, word = numbers[-1]
    try:
        unit = normalize_unit(word)
    except UnitError:
        unit = ""
    return Extraction("fallback", {"status": "ok", "value": value, "unit": unit})


def _si(value: Any, unit: str) -> float:
    return read_answer(str(value), unit)[0]


def to_result(answer: dict[str, Any] | None) -> SolveResult:
    """SolveResult for an extracted answer; status ERROR when there is no numeric answer to grade."""
    if answer is None:
        return SolveResult(status=SolveStatus.ERROR, message="no answer extracted")
    if str(answer.get("status", "ok")).strip().lower() == "no_solution":
        return SolveResult(status=SolveStatus.NO_SOLUTION)
    try:
        unit = normalize_unit(answer.get("unit"))
        values = answer.get("values")
        if isinstance(values, list) and values:
            si = [_si(v, unit) for v in values]
            if len(si) > 1:
                return SolveResult(status=SolveStatus.MULTIPLE, values=si)
            return SolveResult(status=SolveStatus.OK, value=si[0])
        if answer.get("value") not in (None, ""):
            return SolveResult(status=SolveStatus.OK, value=_si(answer["value"], unit))
    except UnitError as ex:
        return SolveResult(status=SolveStatus.ERROR, message=str(ex))
    except Exception as ex:  # noqa: BLE001 — any unreadable number is an answer that cannot be graded
        return SolveResult(status=SolveStatus.ERROR, message=f"number not readable: {ex}")
    return SolveResult(status=SolveStatus.ERROR, message="no numeric answer")
