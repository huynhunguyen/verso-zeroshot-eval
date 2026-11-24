"""Checkpoints of a long run: every finished task is written to disk at once, so a run stopped half way
goes on from where it stopped instead of starting over (task IF-09)."""

import json
import os
from collections.abc import Hashable, Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class CheckpointError(ValueError):
    """A checkpoint holds a value that cannot be read back."""


class Checkpoint:
    def __init__(self, path: str | Path, types: Iterable[type[BaseModel]] = ()):
        self.path = Path(path)
        self.types = {t.__name__: t for t in types}
        self._file = None

    def load(self) -> dict[Hashable, Any]:
        """{key: value} of every task written before; a last line cut short is removed from the file."""
        if not self.path.exists():
            return {}
        data = self.path.read_bytes()
        done, whole = {}, 0
        for line in data.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break
            done[_key(record["key"])] = self._decode(record["value"])
            whole += len(line)
        if whole < len(data):
            with open(self.path, "r+b") as f:
                f.truncate(whole)
        return done

    def save(self, key: Hashable, value: Any) -> None:
        """Append one finished task and force it to disk."""
        if self._file is None:
            self._file = open(self.path, "a", encoding="utf-8", newline="\n")
        self._file.write(json.dumps({"key": key, "value": self._encode(value)}, ensure_ascii=False) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _encode(self, value):
        if isinstance(value, BaseModel):
            name = type(value).__name__
            if name not in self.types:
                raise CheckpointError(f"a {name} cannot be checkpointed here; name its type when making the checkpoint")
            return {"model": name, "data": value.model_dump(mode="json")}
        return {"data": value}

    def _decode(self, value):
        if "model" in value:
            if value["model"] not in self.types:
                raise CheckpointError(f"the checkpoint {self.path} holds a {value['model']}, which is not among its types")
            return self.types[value["model"]].model_validate(value["data"])
        return value["data"]


def _key(key):
    return tuple(_key(k) for k in key) if isinstance(key, list) else key
