"""Experiment runs: one folder per run under runs/, and one line per run in runs/registry.csv.

    with Run("checks", config) as run:
        run.log({...})                   # one JSON line per item in trace.jsonl
        run.write_text("report.md", ...) # any other artifact
        run.set_summary({...})           # metrics; also copied into the registry
        run.checkpoint("solve", types)   # tasks kept on disk as they finish (verso_eval.checkpoint)

A run folder holds config.json (the full configuration), meta.json (git commit, uncommitted changes,
Python and package versions, command line), trace.jsonl, summary.json, the checkpoints and any artifacts. The
registry lists every run, including failed ones, so a reported number can be traced to the code that produced it.

A run stopped half way goes on with Run(name, config, resume=<run id>) (task IF-09): the same folder, the tasks its
checkpoints hold are not run again. It is allowed only on the commit the run started at, with a clean working tree
(and a run started on a clean one), and with the same configuration except the keys that do not change results
(the number of workers). Every resumption is written to meta.json (principle P-06). A run that ended with an error
has its failed line in the registry; the resumed run adds its own line when it ends.
"""

import csv
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from importlib import metadata

from . import paths
from .checkpoint import Checkpoint

REGISTRY_NAME = "registry.csv"
REGISTRY_COLUMNS = ["run_id", "name", "started_at", "finished_at", "status", "git_commit", "git_dirty", "command", "summary"]
TRACKED_PACKAGES = ("verso", "sympy", "pydantic", "openai", "jsonschema", "networkx")
# Configuration keys that may change when a run is resumed: they do not change its results.
MAY_DIFFER = ("workers",)


class ResumeError(ValueError):
    """A run cannot go on: it does not exist, or its code, data or configuration changed since it started."""


def _now():
    return datetime.now().astimezone()


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "run"


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def working_tree_dirty():
    """Whether the repository has uncommitted changes (None when git is not available). The records of earlier runs
    under runs/ do not count: they are outputs, not code or data."""
    status = _git("status", "--porcelain", "--", ".", f":!{paths.RUNS_DIR.relative_to(paths.ROOT).as_posix()}")
    return bool(status) if status is not None else None


def evaluation_refusal(note, server_info):
    """Why a run on the evaluation set that calls a model may not start, or None (rules P-06, P-07, P-14): it needs
    a note saying why it is run, the server's information and a clean working tree."""
    if not note:
        return "a run on the evaluation set needs --note saying why it is run"
    if not server_info:
        return "a run on the evaluation set that calls a model needs --server-info (server, model weights)"
    if working_tree_dirty():
        return "a run on the evaluation set needs a clean working tree (commit first)"
    return None


def _package_versions():
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


class Run:
    def __init__(self, name, config, runs_dir=None, resume=None, run_id=None):
        """resume: the id of a run to go on with; run_id: the id of a new run (default: its start time and name; a
        campaign names the runs of its steps, verso_eval.campaign)."""
        if resume and run_id:
            raise ValueError("give the id of a run to go on with, or of a new run, not both")
        self.name = name
        self.config = config
        self.runs_dir = runs_dir or paths.RUNS_DIR
        self.resume = resume
        self.started = _now()
        self.run_id = resume or run_id or f"{self.started.strftime('%Y%m%d-%H%M%S')}-{_slug(name)}"
        self.dir = self.runs_dir / self.run_id
        self.summary = {}
        self._trace = None
        self._checkpoints = []

    def __enter__(self):
        if self.resume:
            self._reopen()
        else:
            self.dir.mkdir(parents=True, exist_ok=False)
            self._dump("config.json", self.config)
            self.meta = {
                "run_id": self.run_id,
                "name": self.name,
                "started_at": self.started.isoformat(timespec="seconds"),
                "git_commit": _git("rev-parse", "HEAD"),
                "git_dirty": working_tree_dirty(),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "packages": _package_versions(),
                "command": " ".join(sys.argv),
            }
            self._dump("meta.json", self.meta)
        self._trace = open(self.dir / "trace.jsonl", "w", encoding="utf-8", newline="\n")
        return self

    def _reopen(self):
        """Check that the run may go on and record the resumption."""
        if not (self.dir / "meta.json").exists():
            raise ResumeError(f"there is no run {self.run_id} under {self.runs_dir}")
        meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        old = json.loads((self.dir / "config.json").read_text(encoding="utf-8"))
        new = json.loads(json.dumps(self.config, ensure_ascii=False, default=str))
        if meta["name"] != self.name:
            raise ResumeError(f"run {self.run_id} is a {meta['name']} run, not {self.name}")
        if meta.get("git_dirty") is not False:
            raise ResumeError(f"run {self.run_id} started on a working tree with uncommitted changes, so the code it "
                              "ran cannot be matched; start a new run")
        commit = _git("rev-parse", "HEAD")
        if commit != meta["git_commit"]:
            raise ResumeError(f"run {self.run_id} started at commit {meta['git_commit'][:7]}, but the repository is at "
                              f"{(commit or '?')[:7]}; check out that commit to go on")
        if working_tree_dirty() is not False:
            raise ResumeError("the working tree has uncommitted changes; commit or stash them to go on")
        changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
        if any(k not in MAY_DIFFER for k in changed):
            raise ResumeError(f"run {self.run_id} was started with another configuration "
                              f"({', '.join(k for k in changed if k not in MAY_DIFFER)} differ); start a new run")
        meta.setdefault("resumed", []).append({"at": _now().isoformat(timespec="seconds"), "command": " ".join(sys.argv),
                                               "changed": {k: new.get(k) for k in changed}})
        self.meta = meta
        self._dump("meta.json", meta)

    def checkpoint(self, name, types=()):
        """The checkpoint `name` of this run (checkpoint_<name>.jsonl), holding values of the given pydantic types."""
        made = Checkpoint(self.dir / f"checkpoint_{name}.jsonl", types)
        self._checkpoints.append(made)
        return made

    def log(self, record):
        self._trace.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def write_text(self, filename, text):
        (self.dir / filename).write_text(text, encoding="utf-8", newline="\n")

    def set_summary(self, summary):
        self.summary = dict(summary)

    def __exit__(self, exc_type, exc, tb):
        self._trace.close()
        for c in self._checkpoints:
            c.close()
        status = "ok" if exc_type is None else f"failed: {exc_type.__name__}"
        finished = _now()
        resumed = [r["at"] for r in self.meta.get("resumed", [])]
        self._dump("summary.json", {**self.summary, "status": status, "finished_at": finished.isoformat(timespec="seconds"),
                                    **({"resumed_at": resumed} if resumed else {})})
        self._append_registry({
            "run_id": self.run_id,
            "name": self.name,
            "started_at": self.meta["started_at"],
            "finished_at": finished.isoformat(timespec="seconds"),
            "status": status,
            "git_commit": self.meta["git_commit"] or "",
            "git_dirty": self.meta["git_dirty"],
            "command": self.meta["command"],
            "summary": json.dumps(self.summary, ensure_ascii=False),
        })
        return False

    def _dump(self, filename, obj):
        self.write_text(filename, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

    def _append_registry(self, row):
        registry = self.runs_dir / REGISTRY_NAME
        new = not registry.exists()
        with open(registry, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS, lineterminator="\n")
            if new:
                writer.writeheader()
            writer.writerow(row)
