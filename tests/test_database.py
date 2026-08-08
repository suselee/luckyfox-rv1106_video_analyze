from datetime import datetime, timedelta, timezone
from pathlib import Path

from nas_video_summarizer.database import Database


def _insert_segment(db: Database, *, stream_role: str, path: str, started_at: str) -> None:
    ended = (datetime.fromisoformat(started_at) + timedelta(seconds=120)).isoformat(timespec="seconds")
    db.upsert_segment(
        camera_name="test-camera",
        stream_role=stream_role,
        path=Path(path),
        started_at=started_at,
        ended_at=ended,
        duration_seconds=120,
        size_bytes=1024,
    )


def test_get_pending_segments_filters_by_role(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    db.migrate()

    ready_before = "2026-07-08T12:00:00+00:00"

    _insert_segment(db, stream_role="low", path="/buffer/low/seg1.mp4", started_at="2026-07-08T10:00:00+00:00")
    _insert_segment(db, stream_role="high", path="/buffer/high/seg1.mp4", started_at="2026-07-08T10:00:00+00:00")
    _insert_segment(db, stream_role="high", path="/buffer/high/seg2.mp4", started_at="2026-07-08T10:02:00+00:00")

    low_only = db.get_pending_segments(stream_role="low", ready_before=ready_before, max_attempts=3, limit=10)
    high_only = db.get_pending_segments(stream_role="high", ready_before=ready_before, max_attempts=3, limit=10)

    assert len(low_only) == 1
    assert low_only[0]["stream_role"] == "low"
    assert len(high_only) == 2
    assert all(row["stream_role"] == "high" for row in high_only)

    assert db.count_pending_segments(stream_role="low") == 1
    assert db.count_pending_segments(stream_role="high") == 2


def test_get_pending_low_segments_still_works(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    db.migrate()
    ready_before = "2026-07-08T12:00:00+00:00"
    _insert_segment(
        db,
        stream_role="low",
        path="/buffer/low/seg1.mp4",
        started_at="2026-07-08T10:00:00+00:00",
    )

    rows = db.get_pending_low_segments(
        ready_before=ready_before, max_attempts=3, limit=10
    )

    assert len(rows) == 1
    assert rows[0]["stream_role"] == "low"


def test_batch_upsert_inserts_and_updates_segments(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    db.migrate()
    segment = {
        "camera_name": "test-camera",
        "stream_role": "low",
        "path": Path("/buffer/low/seg1.mp4"),
        "started_at": "2026-07-08T10:00:00+00:00",
        "ended_at": "2026-07-08T10:02:00+00:00",
        "duration_seconds": 120,
        "size_bytes": 1024,
    }

    db.upsert_segments([segment])
    db.upsert_segments([{**segment, "size_bytes": 2048}])

    latest = db.latest_segment("low")
    assert latest is not None
    assert latest["size_bytes"] == 2048


def test_connections_configure_busy_timeout(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    db.migrate()

    with db.connect() as conn:
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

    assert busy_timeout == 30_000
    assert journal_mode == "wal"


def test_recent_segments_returns_newest_first(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    db.migrate()
    _insert_segment(
        db,
        stream_role="low",
        path="/buffer/low/older.mp4",
        started_at="2026-07-08T10:00:00+00:00",
    )
    _insert_segment(
        db,
        stream_role="low",
        path="/buffer/low/newer.mp4",
        started_at="2026-07-08T10:02:00+00:00",
    )

    rows = db.recent_segments("low", limit=1)

    assert len(rows) == 1
    assert rows[0]["path"] == "/buffer/low/newer.mp4"


def test_find_segment_near_prefers_earlier_segment_across_small_gap(tmp_path):
    database = Database(tmp_path / "segment-gap.sqlite3")
    database.migrate()
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    first_id = database.upsert_segment(
        camera_name="home-camera",
        stream_role="high",
        path=first,
        started_at="2026-07-23T07:56:00+08:00",
        ended_at="2026-07-23T07:58:00+08:00",
        duration_seconds=120,
        size_bytes=5,
    )
    database.upsert_segment(
        camera_name="home-camera",
        stream_role="high",
        path=second,
        started_at="2026-07-23T07:58:01+08:00",
        ended_at="2026-07-23T08:00:01+08:00",
        duration_seconds=120,
        size_bytes=6,
    )

    segment = database.find_segment_near(
        stream_role="high",
        timestamp="2026-07-23T07:58:00.078+08:00",
        tolerance_seconds=2.0,
    )

    assert segment is not None
    assert segment["id"] == first_id


def test_board_session_requeue_is_idempotent(tmp_path):
    database = Database(tmp_path / "board-requeue.sqlite3")
    database.migrate()
    best_ts = datetime.fromisoformat("2026-07-23T10:00:00+08:00").timestamp()
    database.record_board_event(
        event_key="home-camera:session:s1:end:1",
        session_key="home-camera:session:s1",
        session_id="s1",
        camera_id="home-camera",
        session_start=best_ts - 10,
        identity="probable",
        score=0.8,
        best_ts=best_ts,
        last_event_at=best_ts + 5,
        payload={"activity_score": 0.4},
        event_state="end",
    )
    database.mark_board_session_skipped(
        "home-camera:session:s1",
        "probable event rejected by NAS event-level verification",
    )

    assert database.requeue_board_sessions(
        ["home-camera:session:s1"], requeue_tag="roi-v1"
    ) == 1
    assert database.requeue_board_sessions(
        ["home-camera:session:s1"], requeue_tag="roi-v1"
    ) == 0
    session = database.get_board_session("home-camera:session:s1")
    assert session is not None
    assert session["status"] == "ready"
    assert session["requeue_tag"] == "roi-v1"


def test_migrate_adds_detector_columns_to_existing_moments_table(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE moments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_name TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0,
                source_low_segment_id INTEGER,
                source_started_at TEXT NOT NULL,
                source_ended_at TEXT NOT NULL,
                clip_path TEXT NOT NULL UNIQUE,
                metadata_path TEXT NOT NULL,
                favorited INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
    database = Database(path)
    database.migrate()

    with database.connect() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(moments)")}

    assert {
        "analysis_backend",
        "category",
        "selection_score",
        "clip_started_at",
        "source_segment_id",
        "source_stream_role",
    } <= columns


def test_migrate_backfills_generic_source_segment(tmp_path):
    database = Database(tmp_path / "source.sqlite3")
    database.migrate()
    segment_id = database.upsert_segment(
        camera_name="home-camera",
        stream_role="low",
        path=tmp_path / "low.mp4",
        started_at="2026-07-17T09:00:00+08:00",
        ended_at="2026-07-17T09:02:00+08:00",
        duration_seconds=120,
        size_bytes=1,
    )
    with database.connect() as conn:
        conn.execute(
            """
            INSERT INTO moments(
                camera_name, title, summary, tags_json, confidence,
                source_low_segment_id, source_started_at, source_ended_at,
                clip_path, metadata_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "home-camera",
                "legacy",
                "",
                "[]",
                0.5,
                segment_id,
                "2026-07-17T09:00:00+08:00",
                "2026-07-17T09:02:00+08:00",
                str(tmp_path / "clip.mp4"),
                str(tmp_path / "clip.json"),
                "2026-07-17T09:03:00+08:00",
            ),
        )
        conn.execute(
            "UPDATE moments SET source_segment_id=NULL, source_stream_role=NULL"
        )

    database.migrate()

    moment = database.list_moments(limit=1)[0]
    assert moment["source_segment_id"] == segment_id
    assert moment["source_stream_role"] == "low"


def test_migrate_converts_legacy_utc_created_at_to_local_iso(tmp_path):
    import sqlite3

    path = tmp_path / "legacy-time.sqlite3"
    database = Database(path)
    database.migrate()
    with database.connect() as conn:
        conn.execute(
            "INSERT INTO events(event_type, message, created_at) VALUES (?, ?, ?)",
            ("legacy", "old", "2026-07-17 01:02:03"),
        )

    database.migrate()

    event = database.recent_events(limit=1)[0]
    expected = datetime(2026, 7, 17, 1, 2, 3, tzinfo=timezone.utc).astimezone()
    assert event["created_at"] == expected.isoformat(timespec="seconds")


def test_new_database_timestamps_are_local_iso(tmp_path):
    database = Database(tmp_path / "local-time.sqlite3")
    database.migrate()
    database.add_event("test", "message")
    _insert_segment(
        database,
        stream_role="low",
        path="/buffer/low/local-time.mp4",
        started_at="2026-07-17T10:00:00+08:00",
    )

    event_time = datetime.fromisoformat(database.recent_events(limit=1)[0]["created_at"])
    segment_time = datetime.fromisoformat(database.latest_segment("low")["created_at"])
    assert event_time.tzinfo is not None
    assert segment_time.tzinfo is not None


def test_migrate_preserves_retired_comparison_data(tmp_path):
    import sqlite3

    path = tmp_path / "comparison-history.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE detector_events (
                id INTEGER PRIMARY KEY,
                event_key TEXT NOT NULL
            );
            CREATE TABLE comparison_cases (
                id INTEGER PRIMARY KEY,
                review_label TEXT NOT NULL
            );
            INSERT INTO detector_events(id, event_key) VALUES (1, 'edge:1');
            INSERT INTO comparison_cases(id, review_label) VALUES (1, 'present');
            """
        )

    database = Database(path)
    database.migrate()

    with database.connect() as conn:
        event = conn.execute("SELECT event_key FROM detector_events").fetchone()
        review = conn.execute("SELECT review_label FROM comparison_cases").fetchone()
    assert event[0] == "edge:1"
    assert review[0] == "present"


def test_board_events_are_durable_and_deduplicated(tmp_path):
    path = tmp_path / "board-events.sqlite3"
    database = Database(path)
    database.migrate()
    base = datetime.fromisoformat("2026-07-19T10:00:00+08:00").timestamp()
    common = {
        "session_key": "home-camera:session:s1",
        "session_id": "s1",
        "camera_id": "home-camera",
        "session_start": base,
        "identity": "confirmed",
        "score": 0.8,
        "best_ts": base + 5,
        "last_event_at": base + 10,
        "payload": {"session_id": "s1", "seq": 2},
        "event_state": "end",
    }

    session, inserted = database.record_board_event(
        event_key="home-camera:session:s1:end:2", **common
    )
    duplicate, duplicate_inserted = database.record_board_event(
        event_key="home-camera:session:s1:end:2", **common
    )

    assert inserted is True
    assert duplicate_inserted is False
    assert session["status"] == duplicate["status"] == "ready"
    reopened = Database(path)
    assert reopened.pending_board_sessions()[0]["key"] == "home-camera:session:s1"


def test_stale_board_session_is_finalized_with_configurable_cutoff(tmp_path):
    database = Database(tmp_path / "stale-board.sqlite3")
    database.migrate()
    database.record_board_event(
        event_key="home-camera:session:s1:start:1",
        session_key="home-camera:session:s1",
        session_id="s1",
        camera_id="home-camera",
        session_start=100.0,
        identity="probable",
        score=0.6,
        best_ts=100.0,
        last_event_at=105.0,
        payload={"session_id": "s1", "seq": 1},
        event_state="start",
    )

    assert database.finalize_stale_board_sessions(104.0) == 0
    assert database.finalize_stale_board_sessions(105.0) == 1
    assert database.pending_board_sessions()[0]["status"] == "ready"
