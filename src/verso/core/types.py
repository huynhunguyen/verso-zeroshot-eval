"""Minimal data contracts needed by the zero-shot MLLM evaluation baseline."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SolveStatus(str, Enum):
    """How solving one problem ended."""

    OK = "ok"
    NO_SOLUTION = "no_solution"
    UNDERDETERMINED = "underdetermined"
    MULTIPLE = "multiple"
    ERROR = "error"
    TIMEOUT = "timeout"


class SolveResult(BaseModel):
    """Outcome of solving one problem, in SI units."""

    status: SolveStatus
    value: float | None = None
    angle: float | None = None
    values: list[float] = Field(default_factory=list)
    message: str = ""
    depends_on: list[str] = Field(default_factory=list)


class DirectSolution(BaseModel):
    """The MLLM's direct solution of a problem: response, force-answer response, error, extraction and SI result."""

    response: dict[str, Any] | None = None
    forced: dict[str, Any] | None = None
    error: str | None = None
    extraction: dict[str, Any] = Field(default_factory=lambda: {"method": "none", "source": None, "answer": None})
    result: SolveResult | None = None
