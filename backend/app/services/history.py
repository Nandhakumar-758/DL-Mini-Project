"""SQLite store for opt-in prediction history and app settings.

Typed text is only written when the user has switched history on. The database
lives outside the model-artifact directory.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    text TEXT NOT NULL,
    mode TEXT NOT NULL,
    top_k INTEGER NOT NULL,
    suggestions TEXT NOT NULL
);
"""


class HistoryStore:
    def __init__(self, db_path: Path, max_items: int = 200) -> None:
        self.db_path = Path(db_path)
        self.max_items = max_items
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------------------------------------------------------------- settings
    def history_enabled(self) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key='history_enabled'").fetchone()
        return bool(row and row[0] == "1")  # off unless the user opted in

    def set_history_enabled(self, enabled: bool) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO settings(key, value) VALUES('history_enabled', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                ("1" if enabled else "0",),
            )

    # ----------------------------------------------------------------- history
    def add(self, text: str, mode: str, top_k: int, suggestions: list[str]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO history(created_at, text, mode, top_k, suggestions) VALUES(?,?,?,?,?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    text,
                    mode,
                    top_k,
                    json.dumps(suggestions, ensure_ascii=False),
                ),
            )
            conn.execute(
                "DELETE FROM history WHERE id NOT IN (SELECT id FROM history ORDER BY id DESC LIMIT ?)",
                (self.max_items,),
            )

    def list(self, limit: int = 100) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, text, mode, top_k, suggestions FROM history "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": r[0],
                "created_at": r[1],
                "text": r[2],
                "mode": r[3],
                "top_k": r[4],
                "suggestions": json.loads(r[5]),
            }
            for r in rows
        ]

    def clear(self) -> int:
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM history")
            return cursor.rowcount
