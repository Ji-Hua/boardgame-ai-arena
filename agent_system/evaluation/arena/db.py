"""SQLite storage for arena game results."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from agent_system.evaluation.arena.models import GameRecord

DEFAULT_DB_PATH = Path("arena_results.db")


def init_db(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create (or open) the SQLite database and ensure the schema exists."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_a TEXT NOT NULL,
            agent_b TEXT NOT NULL,
            winner TEXT,
            num_steps INTEGER NOT NULL,
            seed INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def insert_game(conn: sqlite3.Connection, record: GameRecord) -> None:
    """Insert a single game record."""
    conn.execute(
        "INSERT INTO games (agent_a, agent_b, winner, num_steps, seed, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            record.agent_a,
            record.agent_b,
            record.winner,
            record.num_steps,
            record.seed,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def fetch_all_games(conn: sqlite3.Connection) -> list[GameRecord]:
    """Read all game records from the database."""
    rows = conn.execute(
        "SELECT agent_a, agent_b, winner, num_steps, seed FROM games"
    ).fetchall()
    return [
        GameRecord(agent_a=r[0], agent_b=r[1], winner=r[2], num_steps=r[3], seed=r[4])
        for r in rows
    ]
