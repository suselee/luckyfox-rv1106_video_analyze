from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import Settings, load_settings
from .database import Database
from .workers import _segments_cover_window


_REQUEUE_REASON_PREFIXES = (
    "probable event rejected by NAS event-level verification",
    "no 4K segment for board session",
    "4K coverage incomplete for board session",
)


def _number(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _has_high_coverage(
    settings: Settings, database: Database, event_time: datetime
) -> bool:
    segment = database.find_segment_near(
        stream_role="high",
        timestamp=event_time.isoformat(timespec="milliseconds"),
        tolerance_seconds=settings.stream_alignment_tolerance_seconds,
    )
    if segment is None or not Path(str(segment["path"])).exists():
        return False
    wanted_start = event_time - timedelta(seconds=settings.context_before_seconds)
    wanted_end = event_time + timedelta(seconds=settings.context_after_seconds)
    rows = database.find_segments_between(
        stream_role="high",
        started_before=wanted_end.isoformat(timespec="milliseconds"),
        ended_after=wanted_start.isoformat(timespec="milliseconds"),
    )
    available = [row for row in rows if Path(str(row["path"])).exists()]
    return bool(available) and _segments_cover_window(
        available,
        wanted_start,
        wanted_end,
        gap_tolerance_seconds=settings.stream_alignment_tolerance_seconds,
    )


def select_requeue_candidates(
    settings: Settings,
    database: Database,
    *,
    day: str,
    bucket_seconds: int = 300,
) -> list[dict[str, Any]]:
    target_day = datetime.strptime(day, "%Y-%m-%d").date()
    bucket_size = max(1, bucket_seconds)
    best_by_bucket: dict[int, dict[str, Any]] = {}
    for session in database.skipped_probable_board_sessions():
        reason = str(session.get("last_error") or "")
        if not reason.startswith(_REQUEUE_REASON_PREFIXES):
            continue
        event_time = datetime.fromtimestamp(float(session["best_ts"])).astimezone()
        if event_time.date() != target_day:
            continue
        if not _has_high_coverage(settings, database, event_time):
            continue
        payload_value = session.get("payload") or {}
        payload = payload_value if isinstance(payload_value, dict) else {}
        candidate = {
            **session,
            "activity_score": _number(payload.get("activity_score")),
            "event_time": event_time.isoformat(timespec="milliseconds"),
        }
        bucket = int(float(session["best_ts"])) // bucket_size
        current = best_by_bucket.get(bucket)
        rank = (
            _number(candidate.get("score")),
            _number(candidate.get("activity_score")),
            -float(candidate["best_ts"]),
        )
        current_rank = (
            (
                _number(current.get("score")),
                _number(current.get("activity_score")),
                -float(current["best_ts"]),
            )
            if current
            else None
        )
        if current is None or rank > current_rank:
            best_by_bucket[bucket] = candidate
    return sorted(best_by_bucket.values(), key=lambda item: float(item["best_ts"]))


def requeue_main() -> None:
    parser = argparse.ArgumentParser(
        description="Requeue the best skipped RV1106 probable session per time bucket."
    )
    parser.add_argument("--day", required=True, help="Local date in YYYY-MM-DD form")
    parser.add_argument("--bucket-seconds", type=int, default=300)
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the requeue; without this flag the command is a dry run.",
    )
    args = parser.parse_args()
    if args.bucket_seconds <= 0:
        parser.error("--bucket-seconds must be positive")
    try:
        datetime.strptime(args.day, "%Y-%m-%d")
    except ValueError:
        parser.error("--day must use YYYY-MM-DD")

    settings = load_settings()
    database = Database(settings.database_path)
    database.migrate()
    candidates = select_requeue_candidates(
        settings,
        database,
        day=args.day,
        bucket_seconds=args.bucket_seconds,
    )
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry-run",
                "day": args.day,
                "bucket_seconds": args.bucket_seconds,
                "selected": [
                    {
                        "session_key": item["key"],
                        "event_time": item["event_time"],
                        "score": item["score"],
                        "activity_score": item["activity_score"],
                    }
                    for item in candidates
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not args.apply or not candidates:
        return

    run_id = args.run_id.strip() or f"roi-v1-{args.day}"
    selected_keys = [str(item["key"]) for item in candidates]
    updated = database.requeue_board_sessions(
        selected_keys,
        requeue_tag=run_id,
    )
    for item in candidates[:updated]:
        database.add_event(
            "board-session-requeued",
            json.dumps(
                {
                    "run_id": run_id,
                    "session_key": item["key"],
                    "event_time": item["event_time"],
                    "score": item["score"],
                    "activity_score": item["activity_score"],
                },
                ensure_ascii=False,
            ),
        )
    print(f"requeued {updated} board session(s) with run-id {run_id}")


if __name__ == "__main__":
    requeue_main()
