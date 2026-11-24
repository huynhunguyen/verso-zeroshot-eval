"""Zero-shot MLLM baseline for VERSO commercial-model benchmarking (decisions D28-D32).

One fixed prompt (configs/prompts/mllm_zero_shot_v1.txt), one call per question through verso.llm,
and one "force answer" call when a response hits the token limit (D32). Responses are cached in
.cache/llm.sqlite, and every finished question is saved to the run's checkpoint so a stopped run
can resume cleanly with --resume <run_id>.

Usage (from repository root):
    # 1. Smoke test 5 questions first (checks API key, model name, cost, and extraction):
    python -m verso_eval.baselines.mllm_zero_shot run --profile gemini-3-pro --max-tokens 8192 --limit 5

    # 2. Full evaluation on all 1,765 VN questions:
    python -m verso_eval.baselines.mllm_zero_shot run --profile gemini-3-pro --max-tokens 8192 \\
        --workers 4 --note "E-20 commercial baseline: Gemini 3 Pro" \\
        --server-info "Google AI Studio OpenAI-compat API, gemini-3.1-pro-preview"

    # 3. Resume an interrupted run:
    python -m verso_eval.baselines.mllm_zero_shot run --profile gemini-3-pro --max-tokens 8192 \\
        --workers 4 --note "..." --server-info "..." --resume <RUN_ID>
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from verso import direct
from verso.core.types import SolveResult
from verso.direct import FORCE_INSTRUCTION, FORCE_MAX_TOKENS, answer_question, build_messages, to_result  # noqa: F401
from verso.direct.vote import groups
from verso.llm import LLMClient, LLMConfig, client_from_config, config_from_profile, load_profile

from .. import gold, manual_grades, paths
from ..cli import configure_output
from ..grader import GradeReport, circuit_strata, drawing_stratum, grade_results, topic_of
from ..runner import Run, evaluation_refusal

RUN_NAME = "mllm-zero-shot"
PROMPT_NAME = "mllm_zero_shot_v1"
DEFAULT_WORKERS = 4
DEFAULT_TIMEOUT_S = direct.TIMEOUT_S
SETS = ("eval",)
PROGRESS_EVERY = 25
EARLY_STOP_CALLS = 10
VOTE_SAME = ("set", "profile", "prompt_version", "max_tokens", "force_max_tokens", "answer_call")


class CallsFailingError(RuntimeError):
    """The first questions of a run all failed to reach the model, so the run was stopped."""


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a local .env file without overriding already-set variables."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_prompt(name: str = PROMPT_NAME) -> tuple[str, str]:
    """(version, template) of a prompt of configs/prompts."""
    return direct.load_prompt(paths.PROMPTS_DIR, name)


def progress_line(done: int, total: int, errors: int, truncated: int, elapsed_s: float, resumed: int = 0) -> str:
    fresh = done - resumed
    left_s = elapsed_s / fresh * (total - done) if fresh else 0.0
    return (f"{done}/{total} questions, {errors} call errors, {truncated} truncated, "
            f"{elapsed_s / 60:.1f} min elapsed, about {left_s / 60:.1f} min left")


def run_questions(items: Mapping[str, Mapping], image_of: Callable[[Mapping], Any], client: LLMClient,
                  config: LLMConfig, prompt: tuple[str, str], workers: int = DEFAULT_WORKERS,
                  force_max_tokens: int = FORCE_MAX_TOKENS, progress: Callable[[str], None] | None = None,
                  progress_every: int = PROGRESS_EVERY, checkpoint=None,
                  answer_call: bool = False) -> list[dict[str, Any]]:
    version, template = prompt

    def one(qid):
        item = items[qid]
        messages = build_messages(template, item["question"], image_of(item))
        record = answer_question(client, config, version, messages, force_max_tokens, answer_call)
        return {"id": qid, "topic": item.get("topic") or topic_of(qid), "image": item["id_image"], **record}

    records = {q: r for q, r in (checkpoint.load() if checkpoint is not None else {}).items() if q in items}
    resumed = len(records)
    errors = sum(bool(r["error"]) for r in records.values())
    truncated = sum(bool(r["response"] and r["response"]["truncated"]) for r in records.values())
    started = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for future in as_completed([pool.submit(one, qid) for qid in sorted(items) if qid not in records]):
            record = future.result()
            records[record["id"]] = record
            if checkpoint is not None:
                checkpoint.save(record["id"], record)
            errors += bool(record["error"])
            truncated += bool(record["response"] and record["response"]["truncated"])
            done = len(records)
            if done == min(EARLY_STOP_CALLS, len(items)) and errors == done:
                first = next(r["error"] for r in records.values())
                raise CallsFailingError(f"the first {done} questions all failed to reach the model ({first})")
            if progress and (done % progress_every == 0 or done == len(items)):
                progress(progress_line(done, len(items), errors, truncated, time.perf_counter() - started, resumed))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return [records[qid] for qid in sorted(records)]


def grade_records(records: list[dict[str, Any]], items: Mapping[str, Mapping],
                  hand_grades: Mapping[str, bool] | None = None,
                  grade: Callable[[Mapping, Mapping], GradeReport] = grade_results) -> GradeReport:
    results = {r["id"]: to_result(r["extraction"]["answer"]) for r in records}
    report = grade(results, items)
    if hand_grades:
        report.apply_manual(hand_grades)
    return report


def _percentile(values: list[int], p: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def summarize(records: list[dict[str, Any]], report: GradeReport,
              strata: Mapping[str, Callable[[str], str]]) -> dict[str, Any]:
    responses = [r["response"] for r in records if r["response"]]
    truncated = [r["id"] for r in records if r["response"] and r["response"]["truncated"]]
    rescued = [r["id"] for r in records if r["id"] in truncated and r["extraction"]["source"] == "forced"]
    without_force = GradeReport(correct={**report.correct, **{q: False for q in rescued if q in report.correct}},
                                pending=list(report.pending))
    completion = [x["usage"].get("completion_tokens") for x in responses if "completion_tokens" in x["usage"]]
    every_call = responses + [r["forced"] for r in records if r["forced"]]
    return {
        "questions": len(records),
        "graded": len(report.correct),
        "pending_hand_grade": len(report.pending),
        "accuracy": report.accuracy(),
        "accuracy_without_force_answer": without_force.accuracy(),
        "by": {name: report.by(fn) for name, fn in strata.items()},
        "truncated": len(truncated),
        "rescued_by_force_answer": len(rescued),
        "extraction": dict(Counter(r["extraction"]["method"] for r in records)),
        "call_errors": sum(1 for r in records if r["error"]),
        "completion_tokens": {"p50": _percentile(completion, 50), "p99": _percentile(completion, 99),
                              "max": max(completion, default=None)},
        "tokens_total": {key: sum(x["usage"].get(key, 0) for x in every_call)
                         for key in ("prompt_tokens", "completion_tokens")},
        "latency_s_total": round(sum(x["latency_s"] for x in every_call), 1),
        "cached_responses": sum(1 for x in every_call if x["cached"]),
        "models_reported": sorted({x["model"] for x in every_call if x["model"]}),
    }


def load_set(name: str = "eval") -> tuple[dict[str, Mapping], Callable[[Mapping], Any], dict[str, Callable[[str], str]],
                                          Callable[[Mapping, Mapping], GradeReport]]:
    if name != "eval":
        raise ValueError(f"unsupported dataset {name!r}; this standalone package contains only 'eval'")
    dataset = gold.dataset_items()
    items = {qid: dataset[qid] for qid in gold.evaluation_ids()}
    return (items, gold.image_path, {"topic": topic_of, "circuit": circuit_strata().get, "text_drawing": drawing_stratum()},
            grade_results)


def _print_summary(run_id: str, summary: dict[str, Any]) -> None:
    accuracy = summary["accuracy"]
    shown = f"{accuracy:.1%}" if accuracy is not None else f"not reported ({summary['pending_hand_grade']} pending hand grades)"
    print(f"run {run_id}: {summary['questions']} questions, accuracy {shown}")
    print(f"  truncated {summary['truncated']}, rescued by force answer {summary['rescued_by_force_answer']}, "
          f"extraction {summary['extraction']}, call errors {summary['call_errors']}")
    print(f"  tokens_total {summary['tokens_total']}, completion tokens {summary['completion_tokens']}")
    if accuracy is None:
        print(f"  -> To view auto-graded quantitative accuracy (1,710 questions) by topic, run:")
        print(f"     python scripts/check_accuracy.py {run_id}")


def run(args: argparse.Namespace) -> int:
    _load_dotenv(paths.ROOT / ".env")
    is_sample = args.limit is not None
    if not is_sample:
        refused = evaluation_refusal(args.note, args.server_info)
        if refused:
            print(refused, file=sys.stderr)
            return 2
    profile = load_profile(args.profile, paths.MODEL_PROFILES_DIR)
    overrides: dict[str, Any] = {
        "max_tokens": args.max_tokens if args.max_tokens is not None else profile.max_tokens,
        "seed": args.seed,
        "timeout_s": args.timeout,
    }
    if args.model:
        overrides["model"] = args.model
    config = config_from_profile(profile, base_url=args.base_url, api_key_env=args.api_key_env,
                                 cache_path=str(paths.LLM_CACHE), decoding="recommended", **overrides)
    items, image_of, strata, grade = load_set(args.set)
    if is_sample:
        items = {qid: items[qid] for qid in sorted(items)[:args.limit]}
    prompt = load_prompt(args.prompt or PROMPT_NAME)
    settings = {
        "baseline": "mllm_zero_shot", "set": args.set, "questions": len(items), "profile": profile.name,
        "hf_id": profile.hf_id, "served_name": config.model, "base_url": config.base_url,
        "decoding": {k: getattr(config, k) for k in ("temperature", "top_p", "top_k", "presence_penalty", "seed", "extra")},
        "decoding_source": profile.recommended_decoding.source if profile.recommended_decoding else "",
        "max_tokens": config.max_tokens, "max_tokens_param": config.max_tokens_param,
        "force_max_tokens": FORCE_MAX_TOKENS, "prompt_version": prompt[0], "workers": args.workers,
        "note": args.note, "server_info": args.server_info, "answer_call": args.answer_call,
    }
    client = client_from_config(config)
    run_label = f"{RUN_NAME}-{profile.name}-{args.set}" + (f"-sample{args.limit}" if is_sample else "")
    try:
        with Run(run_label, settings, resume=args.resume, run_id=args.run_id) as record:
            saved = record.checkpoint("questions")
            records = run_questions(items, image_of, client, config, prompt, args.workers,
                                    progress=lambda line: print(line, file=sys.stderr, flush=True),
                                    checkpoint=saved, answer_call=args.answer_call)
            report = grade_records(records, items, grade=grade)
            for r in records:
                record.log({**r, "correct": report.correct.get(r["id"])})
            summary = summarize(records, report, strata)
            record.set_summary(summary)
    except CallsFailingError as ex:
        saved.path.unlink(missing_ok=True)
        print(f"stopped: {ex}", file=sys.stderr)
        return 1
    _print_summary(record.run_id, summary)
    return 0


def regrade(args: argparse.Namespace) -> int:
    source = paths.RUNS_DIR / args.run_id
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (source / "trace.jsonl").read_text(encoding="utf-8").splitlines() if line]
    items, _, strata, grade = load_set(config["set"])
    hand_grades = manual_grades.load(args.run_id)
    summary = summarize(records, grade_records(records, items, hand_grades, grade), strata)
    settings = {"regrade_of": args.run_id, "manual_grades": str(manual_grades.grades_path(args.run_id)),
                "hand_grades": len(hand_grades)}
    with Run(f"{RUN_NAME}-regrade", settings) as record:
        record.set_summary(summary)
    _print_summary(record.run_id, summary)
    return 0


def majority(runs: list[list[dict[str, Any]]]) -> dict[str, tuple[SolveResult, list[tuple[SolveResult, int]]]]:
    answers: dict[str, list[SolveResult]] = {}
    for records in runs:
        for r in records:
            answers.setdefault(r["id"], []).append(to_result(r["extraction"]["answer"]))
    out = {}
    for qid, results in answers.items():
        if len(results) == len(runs):
            votes = groups(results)
            out[qid] = (votes[0][0] if votes else results[0], votes)
    return out


def vote(args: argparse.Namespace) -> int:
    sources = [paths.RUNS_DIR / run_id for run_id in args.run_ids]
    configs = [json.loads((s / "config.json").read_text(encoding="utf-8")) for s in sources]
    differing = [k for k in VOTE_SAME if len({json.dumps(c.get(k)) for c in configs}) > 1]
    seeds = [c["decoding"]["seed"] for c in configs]
    refused = (f"the runs differ in {', '.join(differing)}" if differing else
               "the runs must have different seeds" if len(set(seeds)) < len(seeds) else
               "give at least two runs" if len(configs) < 2 else None)
    if refused:
        print(refused, file=sys.stderr)
        return 2
    runs = [[json.loads(line) for line in (s / "trace.jsonl").read_text(encoding="utf-8").splitlines() if line]
            for s in sources]
    items, _, strata, grade = load_set(configs[0]["set"])
    chosen = majority(runs)
    report = grade({qid: answer for qid, (answer, _) in chosen.items()}, items)
    sizes = Counter(votes[0][1] if votes else 0 for _, votes in chosen.values())
    summary = {
        "questions": len(chosen), "graded": len(report.correct), "pending_hand_grade": len(report.pending),
        "accuracy": report.accuracy(), "by": {name: report.by(fn) for name, fn in strata.items()},
        "largest_group": {str(n): sizes[n] for n in sorted(sizes)},
    }
    settings = {"vote_of": list(args.run_ids), "seeds": seeds, **{k: configs[0].get(k) for k in VOTE_SAME},
                "note": args.note}
    with Run(f"{RUN_NAME}-vote-{configs[0]['profile']}-{configs[0]['set']}", settings) as record:
        for qid in sorted(chosen):
            answer, votes = chosen[qid]
            record.log({"id": qid, "answer": answer.model_dump(mode="json"),
                        "votes": [[a.model_dump(mode="json"), n] for a, n in votes],
                        "correct": report.correct.get(qid)})
        record.set_summary(summary)
    accuracy = summary["accuracy"]
    shown = f"{accuracy:.1%}" if accuracy is not None else f"not reported ({summary['pending_hand_grade']} pending)"
    print(f"run {record.run_id}: majority of {len(configs)} runs, {summary['questions']} questions, accuracy {shown}")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_output()
    parser = argparse.ArgumentParser(prog="python -m verso_eval.baselines.mllm_zero_shot", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    go = commands.add_parser("run", help="ask the model every question of the eval set and grade the answers")
    go.add_argument("--set", choices=SETS, default="eval")
    go.add_argument("--profile", required=True, help="model profile name in configs/models")
    go.add_argument("--model", help="override the exact API model snapshot ID from the profile")
    go.add_argument("--base-url", help="override the OpenAI-compatible endpoint URL from the profile")
    go.add_argument("--api-key-env", help="override the environment variable holding the API key")
    go.add_argument("--max-tokens", type=int, help="output token limit (defaults to the profile's max_tokens)")
    go.add_argument("--seed", type=int, default=direct.SEED)
    go.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    go.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds per model call")
    go.add_argument("--prompt", help=f"prompt file name in configs/prompts (default {PROMPT_NAME})")
    go.add_argument("--limit", type=int, help="smoke test on first N questions only (does not require --note)")
    go.add_argument("--note", help="why this run is made (required for a full 1,765-question run)")
    go.add_argument("--server-info", help="API provider and exact model snapshot ID (required for a full run)")
    go.add_argument("--resume", metavar="RUN_ID", help="resume an interrupted run with the same command and settings")
    go.add_argument("--run-id", help="optional explicit run id")
    go.add_argument("--answer-call", action="store_true",
                    help="ask for the final answer as JSON after a response that gives none (for mllm_plain_v1)")
    again = commands.add_parser("regrade", help="grade a finished run again with its hand grades")
    again.add_argument("run_id")
    poll = commands.add_parser("vote", help="grade the majority answer of runs with different seeds (D75)")
    poll.add_argument("run_ids", nargs="+", metavar="RUN_ID", help="finished runs, the first one's answer winning ties")
    poll.add_argument("--note", help="why the vote is made")
    args = parser.parse_args(argv)
    return {"run": run, "regrade": regrade, "vote": vote}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
