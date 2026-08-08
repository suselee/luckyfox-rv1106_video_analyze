from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def local_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def utc_now_iso() -> str:
    """Backward-compatible name; timestamps are local ISO values."""
    return local_now_iso()


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


class Database:
    _BUSY_TIMEOUT_MS = 30_000

    def __init__(self, path: Path):
        self.path = path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=self._BUSY_TIMEOUT_MS / 1000,
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={self._BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def migrate(self) -> None:
        with self.connect() as conn:
            # WAL is persistent for the database file. Setting it once during
            # startup avoids lock-taking journal mode checks on every connection.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_name TEXT NOT NULL,
                    stream_role TEXT NOT NULL CHECK (stream_role IN ('low', 'high')),
                    path TEXT NOT NULL UNIQUE,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    processed_at TEXT,
                    analysis_attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    deleted_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_segments_role_time
                ON segments(stream_role, started_at, ended_at);

                CREATE INDEX IF NOT EXISTS idx_segments_pending
                ON segments(stream_role, processed_at, analysis_attempts, ended_at);

                CREATE TABLE IF NOT EXISTS moments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0,
                    source_low_segment_id INTEGER,
                    source_segment_id INTEGER,
                    source_stream_role TEXT,
                    source_started_at TEXT NOT NULL,
                    source_ended_at TEXT NOT NULL,
                    clip_path TEXT NOT NULL UNIQUE,
                    metadata_path TEXT NOT NULL,
                    analysis_backend TEXT NOT NULL DEFAULT 'vlm',
                    category TEXT NOT NULL DEFAULT 'semantic',
                    selection_score REAL NOT NULL DEFAULT 0,
                    clip_started_at TEXT,
                    clip_ended_at TEXT,
                    trigger_key TEXT,
                    favorited INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(source_low_segment_id) REFERENCES segments(id),
                    FOREIGN KEY(source_segment_id) REFERENCES segments(id)
                );

                CREATE INDEX IF NOT EXISTS idx_moments_created_at
                ON moments(created_at DESC);

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS board_sessions (
                    session_key TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL DEFAULT '',
                    camera_id TEXT NOT NULL,
                    session_start REAL NOT NULL,
                    identity TEXT NOT NULL,
                    score REAL NOT NULL,
                    best_ts REAL NOT NULL,
                    last_event_at REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('active', 'ready', 'saved', 'skipped')
                    ),
                    moment_id INTEGER,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    requeue_tag TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(moment_id) REFERENCES moments(id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_board_sessions_status
                ON board_sessions(status, updated_at);

                CREATE TABLE IF NOT EXISTS board_events (
                    event_key TEXT PRIMARY KEY,
                    session_key TEXT NOT NULL,
                    event_state TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    FOREIGN KEY(session_key) REFERENCES board_sessions(session_key)
                        ON DELETE CASCADE
                );
                """
            )
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(moments)").fetchall()
            }
            additions = (
                ("analysis_backend", "TEXT NOT NULL DEFAULT 'vlm'"),
                ("category", "TEXT NOT NULL DEFAULT 'semantic'"),
                ("selection_score", "REAL NOT NULL DEFAULT 0"),
                ("clip_started_at", "TEXT"),
                ("clip_ended_at", "TEXT"),
                ("trigger_key", "TEXT"),
                ("source_segment_id", "INTEGER"),
                ("source_stream_role", "TEXT"),
            )
            for name, declaration in additions:
                if name not in columns:
                    conn.execute(f"ALTER TABLE moments ADD COLUMN {name} {declaration}")
            conn.execute(
                "UPDATE moments SET selection_score=confidence "
                "WHERE selection_score=0 AND confidence!=0"
            )
            conn.execute(
                "UPDATE moments SET clip_started_at=source_started_at "
                "WHERE clip_started_at IS NULL"
            )
            conn.execute(
                "UPDATE moments SET clip_ended_at=source_ended_at "
                "WHERE clip_ended_at IS NULL"
            )
            conn.execute(
                "UPDATE moments SET source_segment_id=source_low_segment_id "
                "WHERE source_segment_id IS NULL AND source_low_segment_id IS NOT NULL"
            )
            conn.execute(
                "UPDATE moments SET source_stream_role='low' "
                "WHERE source_stream_role IS NULL AND source_low_segment_id IS NOT NULL"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_moments_trigger_key "
                "ON moments(trigger_key) WHERE trigger_key IS NOT NULL"
            )
            board_columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(board_sessions)").fetchall()
            }
            if "requeue_tag" not in board_columns:
                conn.execute("ALTER TABLE board_sessions ADD COLUMN requeue_tag TEXT")
            # The retired detector-comparison tables are intentionally kept.
            # Existing deployments can contain manually reviewed results, and
            # a normal application startup must never destroy those records.
            self._migrate_created_at_to_local(conn)

    @staticmethod
    def _migrate_created_at_to_local(conn: sqlite3.Connection) -> None:
        """Convert legacy SQLite CURRENT_TIMESTAMP values from UTC to local ISO."""
        for table in ("segments", "moments", "events"):
            rows = conn.execute(
                f"SELECT id, created_at FROM {table} WHERE created_at NOT LIKE '%T%'"
            ).fetchall()
            updates: list[tuple[str, int]] = []
            for row in rows:
                value = str(row["created_at"] or "").strip()
                if not value:
                    continue
                try:
                    parsed = datetime.fromisoformat(value)
                except ValueError:
                    continue
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                updates.append(
                    (parsed.astimezone().isoformat(timespec="seconds"), int(row["id"]))
                )
            if updates:
                conn.executemany(
                    f"UPDATE {table} SET created_at=? WHERE id=?", updates
                )

    def add_event(self, event_type: str, message: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO events(event_type, message, created_at) VALUES (?, ?, ?)",
                (event_type, message[:2000], local_now_iso()),
            )

    @staticmethod
    def _decode_board_session(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        session = row_to_dict(row) if isinstance(row, sqlite3.Row) else dict(row)
        raw = session.pop("payload_json", "")
        try:
            session["payload"] = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            session["payload"] = {}
        session["key"] = session.pop("session_key")
        return session

    def record_board_event(
        self,
        *,
        event_key: str,
        session_key: str,
        session_id: str,
        camera_id: str,
        session_start: float,
        identity: str,
        score: float,
        best_ts: float,
        last_event_at: float,
        payload: dict[str, Any],
        event_state: str,
    ) -> tuple[dict[str, Any], bool]:
        """Persist and merge one board event, returning the session and dedupe flag."""
        now = local_now_iso()
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        terminal_event = event_state in {"end", "hit"}
        with self.connect() as conn:
            existing_event = conn.execute(
                "SELECT session_key FROM board_events WHERE event_key=?", (event_key,)
            ).fetchone()
            if existing_event:
                row = conn.execute(
                    "SELECT * FROM board_sessions WHERE session_key=?",
                    (str(existing_event["session_key"]),),
                ).fetchone()
                if row is None:
                    raise RuntimeError("board event references a missing session")
                return self._decode_board_session(row), False

            row = conn.execute(
                "SELECT * FROM board_sessions WHERE session_key=?", (session_key,)
            ).fetchone()
            if row is None:
                status = "ready" if terminal_event else "active"
                conn.execute(
                    """
                    INSERT INTO board_sessions(
                        session_key, session_id, camera_id, session_start,
                        identity, score, best_ts, last_event_at, payload_json,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_key,
                        session_id,
                        camera_id,
                        session_start,
                        identity,
                        score,
                        best_ts,
                        last_event_at,
                        payload_json,
                        status,
                        now,
                        now,
                    ),
                )
            elif str(row["status"]) not in {"saved", "skipped"}:
                current_score = float(row["score"])
                replace_best = score >= current_score
                merged_identity = (
                    "confirmed"
                    if identity == "confirmed" or row["identity"] == "confirmed"
                    else identity
                )
                merged_status = "ready" if terminal_event else str(row["status"])
                conn.execute(
                    """
                    UPDATE board_sessions
                    SET session_start=?, identity=?, score=?, best_ts=?,
                        last_event_at=?, payload_json=?, status=?, updated_at=?
                    WHERE session_key=?
                    """,
                    (
                        min(session_start, float(row["session_start"])),
                        merged_identity,
                        score if replace_best else current_score,
                        best_ts if replace_best else float(row["best_ts"]),
                        max(last_event_at, float(row["last_event_at"])),
                        payload_json if replace_best else str(row["payload_json"]),
                        merged_status,
                        now,
                        session_key,
                    ),
                )

            conn.execute(
                """
                INSERT INTO board_events(event_key, session_key, event_state, received_at)
                VALUES (?, ?, ?, ?)
                """,
                (event_key, session_key, event_state, now),
            )
            merged = conn.execute(
                "SELECT * FROM board_sessions WHERE session_key=?", (session_key,)
            ).fetchone()
            assert merged is not None
            return self._decode_board_session(merged), True

    def get_board_session(self, session_key: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM board_sessions WHERE session_key=?", (session_key,)
            ).fetchone()
        return self._decode_board_session(row) if row else None

    def pending_board_sessions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM board_sessions
                WHERE status='ready'
                ORDER BY updated_at ASC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        return [self._decode_board_session(row) for row in rows]

    def count_pending_board_sessions(self) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM board_sessions WHERE status IN ('active', 'ready')"
            ).fetchone()
        return int(row[0]) if row else 0

    def skipped_probable_board_sessions(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM board_sessions
                WHERE status='skipped'
                  AND identity='probable'
                  AND requeue_tag IS NULL
                ORDER BY best_ts ASC
                """
            ).fetchall()
        return [self._decode_board_session(row) for row in rows]

    def requeue_board_sessions(
        self, session_keys: list[str], *, requeue_tag: str
    ) -> int:
        if not session_keys:
            return 0
        updated = 0
        with self.connect() as conn:
            for index, session_key in enumerate(session_keys):
                row = conn.execute(
                    "SELECT best_ts FROM board_sessions WHERE session_key=?",
                    (session_key,),
                ).fetchone()
                if row is None:
                    continue
                ordered_at = (
                    datetime.fromtimestamp(float(row["best_ts"])).astimezone()
                    + timedelta(microseconds=index)
                ).isoformat(timespec="microseconds")
                cursor = conn.execute(
                    """
                    UPDATE board_sessions
                    SET status='ready', attempts=0, last_error=NULL,
                        requeue_tag=?, updated_at=?
                    WHERE session_key=? AND status='skipped'
                      AND requeue_tag IS NULL
                    """,
                    (requeue_tag, ordered_at, session_key),
                )
                updated += max(0, cursor.rowcount)
        return updated

    def finalize_stale_board_sessions(self, cutoff_timestamp: float) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE board_sessions
                SET status='ready', updated_at=?
                WHERE status='active' AND last_event_at <= ?
                """,
                (local_now_iso(), cutoff_timestamp),
            )
            return max(0, cursor.rowcount)

    def mark_board_session_saved(self, session_key: str, moment_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE board_sessions
                SET status='saved', moment_id=?, last_error=NULL, updated_at=?
                WHERE session_key=?
                """,
                (moment_id, local_now_iso(), session_key),
            )

    def mark_board_session_skipped(self, session_key: str, reason: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE board_sessions
                SET status='skipped', last_error=?, updated_at=?
                WHERE session_key=?
                """,
                (reason[:2000], local_now_iso(), session_key),
            )

    def record_board_session_error(self, session_key: str, error: str) -> int:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE board_sessions
                SET attempts=attempts+1, last_error=?, updated_at=?
                WHERE session_key=?
                """,
                (error[:2000], local_now_iso(), session_key),
            )
            row = conn.execute(
                "SELECT attempts FROM board_sessions WHERE session_key=?",
                (session_key,),
            ).fetchone()
        return int(row["attempts"]) if row else 0

    def upsert_segment(
        self,
        *,
        camera_name: str,
        stream_role: str,
        path: Path,
        started_at: str,
        ended_at: str,
        duration_seconds: float,
        size_bytes: int,
    ) -> int:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO segments(
                    camera_name, stream_role, path, started_at, ended_at,
                    duration_seconds, size_bytes, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    ended_at=excluded.ended_at,
                    duration_seconds=excluded.duration_seconds,
                    size_bytes=excluded.size_bytes
                """,
                (
                    camera_name,
                    stream_role,
                    str(path),
                    started_at,
                    ended_at,
                    duration_seconds,
                    size_bytes,
                    local_now_iso(),
                ),
            )
            row = conn.execute("SELECT id FROM segments WHERE path = ?", (str(path),)).fetchone()
            return int(row["id"])

    def upsert_segments(self, segments: list[dict[str, Any]]) -> None:
        """Insert/update one scanner pass in a single short transaction."""
        if not segments:
            return
        values = [
            (
                segment["camera_name"],
                segment["stream_role"],
                str(segment["path"]),
                segment["started_at"],
                segment["ended_at"],
                segment["duration_seconds"],
                segment["size_bytes"],
                local_now_iso(),
            )
            for segment in segments
        ]
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO segments(
                    camera_name, stream_role, path, started_at, ended_at,
                    duration_seconds, size_bytes, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    ended_at=excluded.ended_at,
                    duration_seconds=excluded.duration_seconds,
                    size_bytes=excluded.size_bytes
                WHERE segments.ended_at != excluded.ended_at
                   OR segments.duration_seconds != excluded.duration_seconds
                   OR segments.size_bytes != excluded.size_bytes
                """,
                values,
            )

    def get_pending_segments(
        self,
        *,
        stream_role: str,
        ready_before: str,
        max_attempts: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM segments
                WHERE stream_role = ?
                  AND deleted_at IS NULL
                  AND processed_at IS NULL
                  AND analysis_attempts < ?
                  AND ended_at <= ?
                ORDER BY started_at ASC
                LIMIT ?
                """,
                (stream_role, max_attempts, ready_before, limit),
            ).fetchall()
            return [row_to_dict(row) for row in rows]

    def get_pending_low_segments(
        self,
        *,
        ready_before: str,
        max_attempts: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Backward-compatible wrapper for callers that only analyze low stream."""
        return self.get_pending_segments(
            stream_role="low",
            ready_before=ready_before,
            max_attempts=max_attempts,
            limit=limit,
        )

    def mark_segment_processed(self, segment_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE segments
                SET processed_at = ?, last_error = NULL
                WHERE id = ?
                """,
                (utc_now_iso(), segment_id),
            )

    def record_analysis_error(self, segment_id: int, error: str, *, final: bool) -> None:
        processed_at = utc_now_iso() if final else None
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE segments
                SET analysis_attempts = analysis_attempts + 1,
                    last_error = ?,
                    processed_at = COALESCE(?, processed_at)
                WHERE id = ?
                """,
                (error[:2000], processed_at, segment_id),
            )

    def find_segments_between(
        self,
        *,
        stream_role: str,
        started_before: str,
        ended_after: str,
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM segments
                WHERE stream_role = ?
                  AND deleted_at IS NULL
                  AND started_at < ?
                  AND ended_at > ?
                ORDER BY started_at ASC
                """,
                (stream_role, started_before, ended_after),
            ).fetchall()
            return [row_to_dict(row) for row in rows]

    def find_segment_at(
        self, *, stream_role: str, timestamp: str
    ) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM segments
                WHERE stream_role=? AND deleted_at IS NULL
                  AND started_at <= ? AND ended_at > ?
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (stream_role, timestamp, timestamp),
            ).fetchone()
        return row_to_dict(row) if row else None

    def find_segment_near(
        self,
        *,
        stream_role: str,
        timestamp: str,
        tolerance_seconds: float,
    ) -> dict[str, Any] | None:
        """Find a containing segment, then the nearest boundary within tolerance."""
        exact = self.find_segment_at(stream_role=stream_role, timestamp=timestamp)
        if exact is not None:
            return exact
        target = datetime.fromisoformat(timestamp)
        tolerance = timedelta(seconds=max(0.0, tolerance_seconds))
        lower = (target - tolerance).isoformat(timespec="microseconds")
        upper = (target + tolerance).isoformat(timespec="microseconds")
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM segments
                WHERE stream_role=? AND deleted_at IS NULL
                  AND started_at <= ? AND ended_at >= ?
                ORDER BY started_at ASC
                """,
                (stream_role, upper, lower),
            ).fetchall()
        candidates = [row_to_dict(row) for row in rows]
        if not candidates:
            return None

        def distance(row: dict[str, Any]) -> tuple[float, datetime]:
            started = datetime.fromisoformat(str(row["started_at"]))
            ended = datetime.fromisoformat(str(row["ended_at"]))
            if target < started:
                seconds = (started - target).total_seconds()
            elif target >= ended:
                seconds = (target - ended).total_seconds()
            else:
                seconds = 0.0
            return seconds, started

        nearest = min(candidates, key=distance)
        if distance(nearest)[0] > max(0.0, tolerance_seconds):
            return None
        return nearest

    def create_moment(
        self,
        *,
        camera_name: str,
        title: str,
        summary: str,
        tags: list[str],
        confidence: float,
        source_low_segment_id: int | None,
        source_started_at: str,
        source_ended_at: str,
        clip_path: Path,
        metadata_path: Path,
        analysis_backend: str = "vlm",
        category: str = "semantic",
        selection_score: float | None = None,
        clip_started_at: str | None = None,
        clip_ended_at: str | None = None,
        trigger_key: str | None = None,
        source_segment_id: int | None = None,
        source_stream_role: str | None = None,
    ) -> int:
        resolved_source_segment_id = (
            source_low_segment_id
            if source_segment_id is None
            else source_segment_id
        )
        resolved_source_stream_role = source_stream_role or (
            "low" if source_low_segment_id is not None else None
        )
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO moments(
                    camera_name, title, summary, tags_json, confidence,
                    source_low_segment_id, source_segment_id, source_stream_role,
                    source_started_at, source_ended_at,
                    clip_path, metadata_path, analysis_backend, category,
                    selection_score, clip_started_at, clip_ended_at, trigger_key,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_name,
                    title,
                    summary,
                    json.dumps(tags, ensure_ascii=True),
                    confidence,
                    source_low_segment_id,
                    resolved_source_segment_id,
                    resolved_source_stream_role,
                    source_started_at,
                    source_ended_at,
                    str(clip_path),
                    str(metadata_path),
                    analysis_backend,
                    category,
                    confidence if selection_score is None else selection_score,
                    clip_started_at or source_started_at,
                    clip_ended_at or source_ended_at,
                    trigger_key,
                    local_now_iso(),
                ),
            )
            return int(cursor.lastrowid)

    def count_moments_on_day(self, day: str) -> int:
        """Number of moments whose source started on ``day`` (``YYYY-MM-DD``)."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM moments WHERE clip_started_at LIKE ?",
                (f"{day}%",),
            ).fetchone()
            return int(row[0]) if row else 0

    def min_confidence_on_day(self, day: str) -> float:
        """Return the weakest confidence, or 1.0 when the day is empty."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT MIN(selection_score) FROM moments WHERE clip_started_at LIKE ?",
                (f"{day}%",),
            ).fetchone()
            return float(row[0]) if row and row[0] is not None else 1.0

    def weakest_moment_on_day(self, day: str) -> dict[str, Any] | None:
        """The lowest-confidence moment starting on ``day`` (for daily-cap eviction)."""
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id, title, confidence, selection_score, category,
                       clip_path, metadata_path
                FROM moments
                WHERE clip_started_at LIKE ?
                ORDER BY selection_score ASC
                LIMIT 1
                """,
                (f"{day}%",),
            ).fetchone()
            if not row:
                return None
            return row_to_dict(row)

    def count_moments_between(self, start_iso: str, end_iso: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM moments
                WHERE clip_started_at >= ? AND clip_started_at < ?
                """,
                (start_iso, end_iso),
            ).fetchone()
            return int(row[0]) if row else 0

    def moments_between(self, start_iso: str, end_iso: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, confidence, selection_score, category,
                       clip_path, metadata_path
                FROM moments
                WHERE clip_started_at >= ? AND clip_started_at < ?
                ORDER BY selection_score ASC
                """,
                (start_iso, end_iso),
            ).fetchall()
        return [row_to_dict(row) for row in rows]

    def weakest_moment_between(
        self, start_iso: str, end_iso: str
    ) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id, title, confidence, selection_score, category,
                       clip_path, metadata_path
                FROM moments
                WHERE clip_started_at >= ? AND clip_started_at < ?
                ORDER BY selection_score ASC
                LIMIT 1
                """,
                (start_iso, end_iso),
            ).fetchone()
            if not row:
                return None
            return row_to_dict(row)

    def moments_on_day(self, day: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM moments
                WHERE clip_started_at LIKE ?
                ORDER BY clip_started_at ASC, id ASC
                """,
                (f"{day}%",),
            ).fetchall()
        return [row_to_dict(row) for row in rows]

    def nearest_moment_before(self, clip_started_at: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM moments
                WHERE clip_started_at <= ?
                ORDER BY clip_started_at DESC
                LIMIT 1
                """,
                (clip_started_at,),
            ).fetchone()
        return row_to_dict(row) if row else None

    def delete_moment_by_clip(self, clip_path: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM moments WHERE clip_path = ?", (str(clip_path),)
            )

    def list_moments(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM moments
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        moments = [row_to_dict(row) for row in rows]
        for moment in moments:
            moment["tags"] = json.loads(moment.pop("tags_json"))
            moment["favorited"] = bool(moment["favorited"])
        return moments

    def get_moment(self, moment_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM moments WHERE id = ?", (moment_id,)).fetchone()
        if row is None:
            return None
        moment = row_to_dict(row)
        moment["tags"] = json.loads(moment.pop("tags_json"))
        moment["favorited"] = bool(moment["favorited"])
        return moment

    def get_moment_by_trigger_key(self, trigger_key: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM moments WHERE trigger_key=?", (trigger_key,)
            ).fetchone()
        if row is None:
            return None
        moment = row_to_dict(row)
        moment["tags"] = json.loads(moment.pop("tags_json"))
        moment["favorited"] = bool(moment["favorited"])
        return moment

    def set_favorite(self, moment_id: int, favorited: bool) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE moments SET favorited = ? WHERE id = ?",
                (1 if favorited else 0, moment_id),
            )

    def delete_moment_record(self, moment_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM moments WHERE id = ?", (moment_id,))

    def count_pending_segments(self, *, stream_role: str = "low") -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM segments
                WHERE stream_role = ?
                  AND deleted_at IS NULL
                  AND processed_at IS NULL
                """,
                (stream_role,),
            ).fetchone()
            return int(row["count"])

    def segment_status_between(
        self, *, stream_role: str, start_iso: str, end_iso: str
    ) -> tuple[int, int, int]:
        """Return pending, final-error, and total counts for a time window."""
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    SUM(processed_at IS NULL) AS pending,
                    SUM(processed_at IS NOT NULL AND last_error IS NOT NULL) AS errors,
                    COUNT(*) AS total
                FROM segments
                WHERE stream_role = ? AND started_at >= ? AND started_at < ?
                """,
                (stream_role, start_iso, end_iso),
            ).fetchone()
        return (
            int(row["pending"] or 0),
            int(row["errors"] or 0),
            int(row["total"] or 0),
        )

    def latest_segment(self, stream_role: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM segments
                WHERE stream_role = ? AND deleted_at IS NULL
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (stream_role,),
            ).fetchone()
        return row_to_dict(row) if row else None

    def recent_segments(
        self, stream_role: str, *, limit: int = 10
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM segments
                WHERE stream_role = ? AND deleted_at IS NULL
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (stream_role, max(1, limit)),
            ).fetchall()
            return [row_to_dict(row) for row in rows]

    def expired_segments(self, cutoff_iso: str, *, limit: int = 500) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM segments
                WHERE deleted_at IS NULL
                  AND ended_at < ?
                ORDER BY ended_at ASC
                LIMIT ?
                """,
                (cutoff_iso, limit),
            ).fetchall()
            return [row_to_dict(row) for row in rows]

    def mark_segment_deleted(self, segment_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE segments SET deleted_at = ? WHERE id = ?",
                (utc_now_iso(), segment_id),
            )

    def recent_events(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM events
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [row_to_dict(row) for row in rows]
