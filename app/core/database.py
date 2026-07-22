"""
SQLite persistence layer for the Neural Gateway model registry.
"""

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

DB_PATH = os.getenv("NEURAL_GATEWAY_DB_PATH", "neural_gateway_registry.db")

_lock = threading.Lock()
_local = threading.local()  # per-thread connection storage for true thread safety

_model_cache: list | None = None
_model_cache_ts: float = 0.0
_MODEL_CACHE_TTL: float = 60.0


def _get_connection() -> sqlite3.Connection:
    """Get or create a per-thread database connection (thread-safe)."""
    if not hasattr(_local, "connection") or _local.connection is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=True)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        _local.connection = conn
    return _local.connection


def init_db() -> None:
    """Create the required tables if they do not exist."""
    with _lock:
        conn = _get_connection()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS models ("
            "  name TEXT PRIMARY KEY,"
            "  provider TEXT,"
            "  tier TEXT,"
            "  data TEXT"
            ")"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS parser_feedback ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  prompt TEXT UNIQUE,"
            "  correct_family TEXT"
            ")"
        )
        conn.commit()


def upsert_model(model: Dict[str, Any]) -> None:
    """Insert or update a model entry."""
    with _lock:
        conn = _get_connection()
        conn.execute(
            "INSERT INTO models (name, provider, tier, data) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(name) DO UPDATE SET provider=excluded.provider,"
            " tier=excluded.tier, data=excluded.data",
            (model["name"], model["provider"], model["tier"], json.dumps(model)),
        )
        conn.commit()
    global _model_cache
    _model_cache = None  # invalidate cache


def bulk_upsert_models(models: List[Dict[str, Any]]) -> None:
    """Insert or update multiple model entries in a single transaction."""
    if not models:
        return
    with _lock:
        conn = _get_connection()
        tuples = [
            (m["name"], m["provider"], m["tier"], json.dumps(m))
            for m in models
        ]
        conn.executemany(
            "INSERT INTO models (name, provider, tier, data) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(name) DO UPDATE SET provider=excluded.provider,"
            " tier=excluded.tier, data=excluded.data",
            tuples,
        )
        conn.commit()
    global _model_cache
    _model_cache = None  # invalidate cache



def get_all_models() -> List[Dict[str, Any]]:
    """Return every model in the registry. Served from in-memory cache when fresh."""
    global _model_cache, _model_cache_ts
    now = time.time()
    if _model_cache is not None and (now - _model_cache_ts) < _MODEL_CACHE_TTL:
        return _model_cache
    with _lock:
        conn = _get_connection()
        rows = conn.execute("SELECT data FROM models").fetchall()
    result = [json.loads(row[0]) for row in rows]
    _model_cache = result
    _model_cache_ts = now
    return result


def get_model(name: str) -> Optional[Dict[str, Any]]:
    """Look up a single model by canonical name."""
    with _lock:
        conn = _get_connection()
        row = conn.execute(
            "SELECT data FROM models WHERE name = ?", (name,)
        ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def delete_model(name: str) -> bool:
    """Remove a model. Returns True if a row was deleted."""
    with _lock:
        conn = _get_connection()
        cursor = conn.execute("DELETE FROM models WHERE name = ?", (name,))
        conn.commit()
        return cursor.rowcount > 0  # read rowcount inside the lock while cursor is valid


def add_feedback(prompt: str, correct_family: str) -> None:
    """Add a feedback example to the memory bank."""
    with _lock:
        conn = _get_connection()
        # ON CONFLICT DO UPDATE preserves AUTOINCREMENT id (INSERT OR REPLACE would reset it)
        conn.execute(
            "INSERT INTO parser_feedback (prompt, correct_family) VALUES (?, ?)"
            " ON CONFLICT(prompt) DO UPDATE SET correct_family=excluded.correct_family",
            (prompt, correct_family)
        )
        conn.commit()


def get_feedback_examples(limit: int = 10) -> List[Dict[str, str]]:
    """Retrieve random or recent feedback examples to inject into the parser."""
    with _lock:
        conn = _get_connection()
        # Order by random to get diverse examples in the prompt, or by id DESC for most recent
        rows = conn.execute(
            "SELECT prompt, correct_family FROM parser_feedback ORDER BY RANDOM() LIMIT ?", (limit,)
        ).fetchall()
    return [{"prompt": row[0], "correct_family": row[1]} for row in rows]


def get_all_feedback() -> List[Dict[str, str]]:
    """Retrieve all feedback examples to inject into the heuristic parser's dynamic vocabulary."""
    with _lock:
        conn = _get_connection()
        rows = conn.execute(
            "SELECT prompt, correct_family FROM parser_feedback"
        ).fetchall()
    return [{"prompt": row[0], "correct_family": row[1]} for row in rows]
