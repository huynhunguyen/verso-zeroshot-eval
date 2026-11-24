"""On-disk cache of model responses, keyed by a hash of everything that determines the response."""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .types import ChatRequest

_SCHEMA = "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, response TEXT NOT NULL)"
# Seconds a writer waits for the lock, so that calls made from several threads do not fail with "database is locked".
_LOCK_TIMEOUT_S = 30.0


def cache_key(provider_id: str, request: ChatRequest) -> str:
    """SHA-256 of the provider, model, messages, decoding parameters and prompt version."""
    payload = {"provider": provider_id, "request": request.model_dump(mode="json")}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    """SQLite table of responses (JSON dicts) keyed by cache_key."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=_LOCK_TIMEOUT_S) as db:
            db.execute(_SCHEMA)

    def get(self, key: str) -> dict[str, Any] | None:
        """Stored response for a key, or None."""
        with sqlite3.connect(self.path, timeout=_LOCK_TIMEOUT_S) as db:
            row = db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, response: dict[str, Any]) -> None:
        """Store the response for a key, replacing any earlier one."""
        with sqlite3.connect(self.path, timeout=_LOCK_TIMEOUT_S) as db:
            db.execute("INSERT OR REPLACE INTO responses (key, response) VALUES (?, ?)",
                       (key, json.dumps(response, ensure_ascii=False)))
