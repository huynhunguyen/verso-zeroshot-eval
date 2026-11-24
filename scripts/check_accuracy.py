"""Print quantitative accuracy (1,710 auto-graded questions), per-topic breakdown, and token usage
for any finished OR in-progress run under runs/<run_id>/.

Usage:
    python scripts/check_accuracy.py <run_id>
    python scripts/check_accuracy.py --latest
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verso.direct import to_result
from verso_eval import gold, grader, paths
from verso_eval.cli import configure_output


def find_run_dir(arg: str) -> Path:
    if arg == "--latest":
        dirs = sorted([d for d in paths.RUNS_DIR.iterdir() if d.is_dir() and (d / "meta.json").exists()])
        if not dirs:
            raise SystemExit("No runs found under runs/")
        return dirs[-1]
    d = paths.RUNS_DIR / arg
    if not d.exists():
        raise SystemExit(f"Run folder not found: {d}")
    return d


def load_records(run_dir: Path) -> tuple[list[dict], bool]:
    trace = run_dir / "trace.jsonl"
    if trace.exists() and trace.stat().st_size > 0:
        rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows, True
    chk = run_dir / "checkpoint_questions.jsonl"
    if chk.exists():
        rows = [json.loads(line)["value"]["data"] for line in chk.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows, False
    raise SystemExit(f"Neither trace.jsonl nor checkpoint_questions.jsonl found in {run_dir}")


def main() -> int:
    configure_output()
    parser = argparse.ArgumentParser(description="Inspect quantitative accuracy of a zero-shot run.")
    parser.add_argument("run_id", nargs="?", default="--latest", help="Run ID under runs/ (or --latest)")
    args = parser.parse_args()

    run_dir = find_run_dir(args.run_id)
    records, finished = load_records(run_dir)
    items = gold.dataset_items()
    results = {r["id"]: to_result(r["extraction"]["answer"]) for r in records}
    report = grader.grade_results(results, items)

    ok = sum(report.correct.values())
    graded = len(report.correct)
    pending = len(report.pending)
    errors = sum(1 for r in records if r.get("error"))
    truncated = sum(1 for r in records if (r.get("response") or {}).get("truncated"))

    prompt_tok = 0
    comp_tok = 0
    for r in records:
        for key in ("response", "forced"):
            resp = r.get(key)
            if resp and isinstance(resp.get("usage"), dict):
                prompt_tok += resp["usage"].get("prompt_tokens", 0)
                comp_tok += resp["usage"].get("completion_tokens", 0)

    state = "FINISHED" if finished else "IN PROGRESS (from checkpoint)"
    print(f"Run:     {run_dir.name} [{state}]")
    print(f"Done:    {len(records)}/1765 questions (auto-graded: {graded}, pending hand-grade: {pending})")
    if graded:
        print(f"AutoAcc: {ok}/{graded} = {ok / graded * 100:.2f}%  (note: excludes {pending} text-answer questions)")
    print(f"Health:  call_errors={errors}, truncated={truncated}")
    print(f"Tokens:  prompt={prompt_tok:,}, completion={comp_tok:,}")

    by_topic: dict[str, Counter] = {}
    for qid, is_ok in report.correct.items():
        t = grader.topic_of(qid)
        by_topic.setdefault(t, Counter())
        by_topic[t]["tot"] += 1
        by_topic[t]["ok" if is_ok else "err"] += 1
    if by_topic:
        print("\nBy topic (auto-graded quantitative only):")
        for t in sorted(by_topic):
            bt = by_topic[t]
            print(f"  {t:4s}: {bt['ok']:4d}/{bt['tot']:<4d} ({bt['ok'] / bt['tot'] * 100:6.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
