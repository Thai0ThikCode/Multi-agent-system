"""
storage/database.py
Lightweight SQLite persistence for full-run state and per-agent audit log
entries. Uses the stdlib `sqlite3` module (no ORM) to keep the dependency
surface minimal; everything is stored as JSON blobs plus a few indexed
columns for fast dashboard queries.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from config import APP_CONFIG
from schemas import AgentState, AgentLogEntry

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    final_direction TEXT,
    confidence_score REAL,
    state_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    step TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
"""


@contextmanager
def _connect(db_path: str | None = None):
    path = db_path or APP_CONFIG.db_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    with _connect(db_path):
        pass


def upsert_run_state(state: AgentState, db_path: str | None = None) -> None:
    final_direction = state.synthesis_report.final_direction.value if state.synthesis_report else None
    confidence = state.synthesis_report.confidence_score if state.synthesis_report else None
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, started_at, final_direction, confidence_score, state_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                final_direction=excluded.final_direction,
                confidence_score=excluded.confidence_score,
                state_json=excluded.state_json
            """,
            (state.run_id, state.started_at, final_direction, confidence, state.model_dump_json()),
        )


def append_agent_log(run_id: str, entry: AgentLogEntry, db_path: str | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO agent_logs (run_id, agent_name, step, timestamp, entry_json) VALUES (?, ?, ?, ?, ?)",
            (run_id, entry.agent_name, entry.step, entry.timestamp, entry.model_dump_json()),
        )


def load_run_state(run_id: str, db_path: str | None = None) -> AgentState | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT state_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        return AgentState.model_validate_json(row[0])


def list_runs(limit: int = 25, db_path: str | None = None) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT run_id, started_at, final_direction, confidence_score FROM runs "
            "ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"run_id": r[0], "started_at": r[1], "final_direction": r[2], "confidence_score": r[3]}
            for r in rows
        ]


def get_logs_for_run(run_id: str, db_path: str | None = None) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT entry_json FROM agent_logs WHERE run_id = ? ORDER BY id ASC", (run_id,)
        ).fetchall()
        return [json.loads(r[0]) for r in rows]
