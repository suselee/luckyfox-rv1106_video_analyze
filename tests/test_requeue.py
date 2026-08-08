from dataclasses import replace
from datetime import datetime

from nas_video_summarizer.config import load_settings
from nas_video_summarizer.database import Database
from nas_video_summarizer.requeue import select_requeue_candidates


def _add_skipped(
    database: Database,
    *,
    session_id: str,
    best_at: str,
    score: float,
    activity_score: float,
    reason: str = "probable event rejected by NAS event-level verification",
) -> None:
    best_ts = datetime.fromisoformat(best_at).timestamp()
    key = f"home-camera:session:{session_id}"
    database.record_board_event(
        event_key=f"{key}:end:1",
        session_key=key,
        session_id=session_id,
        camera_id="home-camera",
        session_start=best_ts - 10,
        identity="probable",
        score=score,
        best_ts=best_ts,
        last_event_at=best_ts + 5,
        payload={"activity_score": activity_score},
        event_state="end",
    )
    database.mark_board_session_skipped(key, reason)


def test_requeue_selects_best_candidate_per_five_minute_bucket(tmp_path):
    settings = replace(
        load_settings("/nonexistent.env"),
        context_before_seconds=5,
        context_after_seconds=10,
        stream_alignment_tolerance_seconds=2.0,
    )
    database = Database(tmp_path / "requeue.sqlite3")
    database.migrate()
    high_path = tmp_path / "high.mp4"
    high_path.write_bytes(b"high")
    database.upsert_segment(
        camera_name="home-camera",
        stream_role="high",
        path=high_path,
        started_at="2026-07-23T09:59:00+08:00",
        ended_at="2026-07-23T10:12:00+08:00",
        duration_seconds=780,
        size_bytes=4,
    )
    _add_skipped(
        database,
        session_id="low-score",
        best_at="2026-07-23T10:00:10+08:00",
        score=0.70,
        activity_score=0.95,
    )
    _add_skipped(
        database,
        session_id="best-score",
        best_at="2026-07-23T10:03:00+08:00",
        score=0.80,
        activity_score=0.20,
    )
    _add_skipped(
        database,
        session_id="next-bucket",
        best_at="2026-07-23T10:05:10+08:00",
        score=0.75,
        activity_score=0.50,
    )
    _add_skipped(
        database,
        session_id="legacy-low",
        best_at="2026-07-23T10:06:00+08:00",
        score=0.99,
        activity_score=0.99,
        reason="no low segment for board session legacy-low",
    )

    selected = select_requeue_candidates(
        settings,
        database,
        day="2026-07-23",
        bucket_seconds=300,
    )

    assert [item["session_id"] for item in selected] == [
        "best-score",
        "next-bucket",
    ]
