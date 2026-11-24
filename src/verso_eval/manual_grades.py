"""Hand grades of text-answer questions (decision D29): one CSV file per run, runs/manual_grades/<run id>.csv.

Columns: id, correct (1 right, 0 wrong), grader, note. Questions with a text answer stay pending in the grader's
report until a grade is recorded here, and a report with a pending question gives no accuracy.

    report.apply_manual(load(run_id))
"""

import csv
from pathlib import Path

from . import paths
from .grader import ManualGradeError

COLUMNS = ("id", "correct", "grader", "note")


def grades_path(run_id: str) -> Path:
    return paths.MANUAL_GRADES_DIR / f"{run_id}.csv"


def load(run_id: str, path: Path | None = None) -> dict[str, bool]:
    """{question id: right or wrong} recorded for a run; empty when no file exists yet."""
    path = path or grades_path(run_id)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ("id", "correct") if c not in (reader.fieldnames or ())]
        if missing:
            raise ManualGradeError(f"{path.name}: missing columns {', '.join(missing)}")
        grades = {}
        for row in reader:
            qid, value = row["id"].strip(), row["correct"].strip()
            if value not in ("0", "1"):
                raise ManualGradeError(f"{path.name}: {qid}: correct must be 1 or 0, got {value!r}")
            if qid in grades:
                raise ManualGradeError(f"{path.name}: {qid} is graded twice")
            grades[qid] = value == "1"
    return grades
