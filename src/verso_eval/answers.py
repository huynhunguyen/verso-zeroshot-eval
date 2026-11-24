"""Dataset answers: numeric reading, unit scale, and the comparison rule for one value.

verso_eval.grader builds the full grading rules (several roots, reactive power by magnitude, no solution,
answer key for text answers) on top of these.
"""

import unicodedata as ud

from verso.core.units import read_answer as answer_si  # noqa: F401 — the grader's name for it

# The dataset writes "no solution" answers in Vietnamese; this prefix is matched against the data.
NO_SOLUTION_ANSWER = "vô nghiệm"
RELATIVE_TOLERANCE = 0.01
# Answers are rounded to 2 decimals in their own unit, so half a unit of the last digit plus a margin.
ROUNDING_TOLERANCE = 0.00501


def is_no_solution(answer: str) -> bool:
    """Whether a dataset answer says the problem has no solution ("Vô nghiệm")."""
    return ud.normalize("NFC", answer).strip().lower().startswith(NO_SOLUTION_ANSWER)


def matches(got: float, expected: float, scale: float) -> bool:
    """Within 1% of the answer, or within the rounding of its last digit in the answer's unit."""
    return abs(got - expected) <= max(RELATIVE_TOLERANCE * abs(expected), ROUNDING_TOLERANCE * scale)
