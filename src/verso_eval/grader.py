"""Grade a result against the dataset answer (exact shared grader of VERSO, plan section 5.1).

Rules
    number           SI value within 1% or within the rounding of the answer's last digit (verso_eval.answers)
    several roots    the answer "a hoặc b" is a set; the result must give every root and nothing else (D5),
                     and a single-answer question answered with several roots is wrong as well
    reactive power   answers in var are compared by magnitude (D4)
    no solution      the answer "Vô nghiệm" is right only when the solver reports no solution
    text answers     wait for a hand grade (decision D29): grade_results marks them pending, and a report with a
                     pending question gives no accuracy until manual grades are applied.
"""

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from verso.core.types import SolveResult, SolveStatus

from . import gold
from .answers import RELATIVE_TOLERANCE, answer_si, is_no_solution, matches

_ALTERNATIVES = re.compile(r"\s+hoặc\s+")


class UngradableAnswerError(ValueError):
    """The answer text has no numeric reading."""


@dataclass(frozen=True)
class ExpectedAnswer:
    """What a question's answer requires: no solution, or values (a set when there are several roots)."""

    no_solution: bool = False
    values: tuple[float, ...] = ()
    scale: float = 1.0
    magnitude_only: bool = False
    tolerance: float | None = None


def is_reactive_power_unit(unit: str | None) -> bool:
    """Whether a unit is var, kvar, ... (reactive power is graded by magnitude, decision D4)."""
    return (unit or "").strip().lower().endswith("var")


def expected_answer(text: str, unit: str | None) -> ExpectedAnswer:
    """ExpectedAnswer read from an answer text such as '1.5', '0.317 hoặc 0.921' or 'Vô nghiệm'."""
    if is_no_solution(text):
        return ExpectedAnswer(no_solution=True)
    values, scale = [], 1.0
    for part in _ALTERNATIVES.split(text.strip()):
        try:
            value, scale = answer_si(part, unit)
        except Exception as ex:  # noqa: BLE001
            raise UngradableAnswerError(f"answer {text!r} has no numeric reading") from ex
        values.append(value)
    return ExpectedAnswer(values=tuple(values), scale=scale, magnitude_only=is_reactive_power_unit(unit))


def result_values(result: SolveResult) -> list[float]:
    """Values a SolveResult gives: one when OK, every root when MULTIPLE, none otherwise."""
    if result.status == SolveStatus.OK:
        return [result.value]
    if result.status == SolveStatus.MULTIPLE:
        return list(result.values)
    return []


def grade(result: SolveResult, expected: ExpectedAnswer) -> bool:
    """Whether a SolveResult is right for an ExpectedAnswer."""
    if expected.no_solution:
        return result.status == SolveStatus.NO_SOLUTION
    got, want = result_values(result), list(expected.values)
    if not got or len(got) != len(want):
        return False
    if expected.magnitude_only:
        got, want = [abs(v) for v in got], [abs(v) for v in want]
    if expected.tolerance is not None:
        return all(abs(g - w) <= max(RELATIVE_TOLERANCE * abs(w), expected.tolerance)
                   for g, w in zip(sorted(got), sorted(want)))
    return all(matches(g, w, expected.scale) for g, w in zip(sorted(got), sorted(want)))


class ManualGradeError(ValueError):
    """A hand grade does not match a question waiting for one."""


@dataclass
class GradeReport:
    """Grades of a set of results, overall and by stratum."""

    correct: dict[str, bool] = field(default_factory=dict)
    pending: list[str] = field(default_factory=list)
    key_check: dict[str, bool] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return not self.pending

    def accuracy(self, ids: Iterable[str] | None = None) -> float | None:
        ids = list(ids) if ids is not None else [*self.correct, *self.pending]
        if set(ids) & set(self.pending):
            return None
        graded = [i for i in ids if i in self.correct]
        return sum(self.correct[i] for i in graded) / len(graded) if graded else None

    def by(self, stratum_of: Callable[[str], str]) -> dict[str, tuple[int | None, int, int]]:
        totals, right, waiting = Counter(), Counter(), Counter()
        for qid, ok in self.correct.items():
            totals[stratum_of(qid)] += 1
            right[stratum_of(qid)] += ok
        for qid in self.pending:
            totals[stratum_of(qid)] += 1
            waiting[stratum_of(qid)] += 1
        return {s: (None if self.pending else right[s], totals[s], waiting[s]) for s in sorted(totals)}

    def apply_manual(self, grades: Mapping[str, bool]) -> None:
        unexpected = sorted(set(grades) - set(self.pending))
        if unexpected:
            raise ManualGradeError(f"hand grades for questions that are not waiting for one: {', '.join(unexpected)}")
        for qid, ok in grades.items():
            self.correct[qid] = ok
        self.pending = [qid for qid in self.pending if qid not in grades]


def needs_hand_grade(answer: str, unit: str | None) -> bool:
    """Whether an answer is text, so that it is graded by hand (decision D29)."""
    try:
        expected_answer(answer, unit)
    except UngradableAnswerError:
        return True
    return False


def grade_results(results: Mapping[str, SolveResult], items: Mapping[str, Mapping] | None = None,
                  answer_key: Mapping[str, list[str]] | None = None, by_key: bool = False) -> GradeReport:
    """Grade {question id: SolveResult} against the answers of `items`."""
    items = gold.dataset_items() if items is None else items
    report = GradeReport()
    for qid, result in results.items():
        item = items[qid]
        if needs_hand_grade(item["answer"], item["unit"]):
            keyed = (answer_key or {}).get(qid)
            right = grade(result, expected_answer(" hoặc ".join(keyed), item["unit"])) if keyed else None
            if by_key and right is not None:
                report.correct[qid] = right
                continue
            report.pending.append(qid)
            if right is not None:
                report.key_check[qid] = right
            continue
        report.correct[qid] = grade(result, expected_answer(item["answer"], item["unit"]))
    return report


def topic_of(qid: str) -> str:
    """Topic code of a question id: 'DLO053' -> 'DLO'."""
    return qid.rstrip("0123456789")


TEXT_DISAGREES, TEXT_AGREES = "text disagrees with drawing", "text agrees with drawing"


def drawing_stratum() -> Callable[[str], str]:
    """The stratum of a question by whether its text and drawing disagree (rule P-12)."""
    tagged = gold.text_drawing_mismatch()
    return lambda qid: TEXT_DISAGREES if qid in tagged else TEXT_AGREES


def circuit_strata() -> dict[str, str]:
    """{question id: 'uses circuit' or 'formula only'}."""
    return gold.circuit_strata()
