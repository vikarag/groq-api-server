"""Persistent usage tracking with SQLite.

Records every API request (model, tokens, duration, timestamp) and provides
query methods for the dashboard. Data survives service restarts.
"""

import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "usage.db"


class UsageTracker:
    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                duration_ms INTEGER NOT NULL DEFAULT 0,
                is_error INTEGER NOT NULL DEFAULT 0,
                endpoint TEXT NOT NULL DEFAULT 'chat'
            );

            CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_usage_model ON usage_log(model);
        """)
        conn.commit()

    def record(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        duration_ms: int = 0,
        is_error: bool = False,
        endpoint: str = "chat",
    ):
        """Record a single API request."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO usage_log
               (model, prompt_tokens, completion_tokens, total_tokens,
                duration_ms, is_error, endpoint)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (model, prompt_tokens, completion_tokens, total_tokens,
             duration_ms, 1 if is_error else 0, endpoint),
        )
        conn.commit()

    def get_summary(self, hours: int = 24) -> dict:
        """Get aggregate usage stats for the last N hours."""
        conn = self._get_conn()
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        rows = conn.execute(
            """SELECT
                 model,
                 COUNT(*) as request_count,
                 SUM(total_tokens) as total_tokens,
                 SUM(prompt_tokens) as prompt_tokens,
                 SUM(completion_tokens) as completion_tokens,
                 AVG(duration_ms) as avg_duration_ms,
                 SUM(CASE WHEN is_error = 1 THEN 1 ELSE 0 END) as error_count
               FROM usage_log
               WHERE timestamp >= ?
               GROUP BY model
               ORDER BY request_count DESC""",
            (cutoff,),
        ).fetchall()

        models = {}
        for row in rows:
            models[row["model"]] = {
                "request_count": row["request_count"],
                "total_tokens": row["total_tokens"] or 0,
                "prompt_tokens": row["prompt_tokens"] or 0,
                "completion_tokens": row["completion_tokens"] or 0,
                "avg_duration_ms": round(row["avg_duration_ms"] or 0),
                "error_count": row["error_count"],
            }

        # Totals
        total_row = conn.execute(
            """SELECT
                 COUNT(*) as total_requests,
                 SUM(total_tokens) as total_tokens,
                 SUM(CASE WHEN is_error = 1 THEN 1 ELSE 0 END) as total_errors
               FROM usage_log
               WHERE timestamp >= ?""",
            (cutoff,),
        ).fetchone()

        return {
            "period_hours": hours,
            "total_requests": total_row["total_requests"] or 0,
            "total_tokens": total_row["total_tokens"] or 0,
            "total_errors": total_row["total_errors"] or 0,
            "models": models,
        }

    def get_recent(self, limit: int = 50) -> list[dict]:
        """Get the most recent usage log entries."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT timestamp, model, prompt_tokens, completion_tokens,
                      total_tokens, duration_ms, is_error, endpoint
               FROM usage_log
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]

    def get_hourly_stats(self, hours: int = 24) -> list[dict]:
        """Get hourly aggregated stats for charting."""
        conn = self._get_conn()
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        rows = conn.execute(
            """SELECT
                 strftime('%Y-%m-%d %H:00', timestamp) as hour,
                 model,
                 COUNT(*) as request_count,
                 SUM(total_tokens) as total_tokens
               FROM usage_log
               WHERE timestamp >= ?
               GROUP BY hour, model
               ORDER BY hour ASC""",
            (cutoff,),
        ).fetchall()

        return [dict(row) for row in rows]

    def get_all_time_stats(self) -> dict:
        """Get all-time totals."""
        conn = self._get_conn()
        row = conn.execute(
            """SELECT
                 COUNT(*) as total_requests,
                 SUM(total_tokens) as total_tokens,
                 SUM(prompt_tokens) as prompt_tokens,
                 SUM(completion_tokens) as completion_tokens,
                 MIN(timestamp) as first_request,
                 MAX(timestamp) as last_request
               FROM usage_log"""
        ).fetchone()

        return {
            "total_requests": row["total_requests"] or 0,
            "total_tokens": row["total_tokens"] or 0,
            "prompt_tokens": row["prompt_tokens"] or 0,
            "completion_tokens": row["completion_tokens"] or 0,
            "first_request": row["first_request"],
            "last_request": row["last_request"],
        }


# Singleton
usage_tracker = UsageTracker()
