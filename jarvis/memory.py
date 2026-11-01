"""Conversation memory: SQLite, no server."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_ts ON turns(ts);
"""


class Memory:
    def __init__(self, db_path: Path):
        self.db = sqlite3.connect(db_path)
        self.db.executescript(_SCHEMA)

    def append(self, role: str, content: str) -> None:
        self.db.execute(
            "INSERT INTO turns (ts, role, content) VALUES (?, ?, ?)",
            (time.time(), role, content),
        )
        self.db.commit()

    def last_turns(self, n: int) -> list[tuple[str, str]]:
        rows = self.db.execute(
            "SELECT role, content FROM turns ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return list(reversed(rows))
