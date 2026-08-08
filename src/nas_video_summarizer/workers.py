from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

from .analysis import ClipCandidate
from .archive import rebuild_day_archive
from .config import Settings
from .database import Database, utc_now_iso
from .daughter_detector import DaughterDetector, ProbableVerification
from .ffmpeg_tools import (
    PersonFilterDecision,
    SampledFrame,
    _extract_frame,
    build_contact_sheet_from_frames,
    build_recorder_command,
    extract_cropped_frame,
    extract_clip,
    ffmpeg_available,
    ffprobe_available,
    filter_frames_by_person_detection,
    filter_out_blank_frames,
    parse_segment_filename,
    sample_frames_with_offsets,
    sample_frames_at_fps,
    segment_time_window,
)
from .llm import AnalysisResult, DaughterVerification, LlamaAnalyzer
from .mqtt import MQTTSubscriber, decode_json_payload


class PersonFilterSkip(Exception):
    """Raised when person filter detects no person in any sampled frame."""


class BoardEventPersistenceError(Exception):
    """Leave a QoS-1 event unacknowledged so the broker will redeliver it."""


# How often the board-event worker checks for stale sessions.
_BOARD_SESSION_SCAN_SECONDS = 5.0
# Extraction failures are retried from the durable queue across worker loops.
_BOARD_SESSION_MAX_PROCESS_ATTEMPTS = 3


def _is_timeout_error(exc: Exception) -> bool:
    return "timed out" in str(exc).lower() or isinstance(exc, asyncio.TimeoutError)

def _now() -> datetime:
    return datetime.now().astimezone()


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        if process.returncode is None:
            process.kill()
        await process.wait()


def _in_time_window(now: datetime, start: str, end: str) -> bool:
    """Check whether `now` falls inside the [start, end) time window.

    start/end are "HH:MM" strings. If either is empty, the window is
    disabled and the function returns True (always active). Handles
    windows that cross midnight (e.g. 21:15 -> 06:00).
    """
    if not start or not end:
        return True

    def _to_minutes(value: str) -> int:
        parts = value.split(":")
        return int(parts[0]) * 60 + int(parts[1])

    now_min = now.hour * 60 + now.minute
    start_min = _to_minutes(start)
    end_min = _to_minutes(end)

    if start_min <= end_min:
        return start_min <= now_min < end_min
    # Crosses midnight: e.g. 21:15 -> 06:00.
    return now_min >= start_min or now_min < end_min


def _in_analysis_window(now: datetime, start: str, end: str) -> bool:
    return _in_time_window(now, start, end)


def _in_record_window(now: datetime, start: str, end: str) -> bool:
    return _in_time_window(now, start, end)


def _moment_period(
    started_at: datetime, boundaries_value: str
) -> tuple[str, datetime, datetime] | None:
    parts = [part.strip() for part in boundaries_value.split(",")]
    if len(parts) != 4:
        return None
    try:
        minutes = []
        for part in parts:
            hour_text, minute_text = part.split(":", 1)
            hour, minute = int(hour_text), int(minute_text)
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                return None
            minutes.append(hour * 60 + minute)
    except (TypeError, ValueError):
        return None
    if minutes != sorted(set(minutes)):
        return None

    current = started_at.hour * 60 + started_at.minute
    labels = ("morning", "afternoon", "evening")
    for index, label in enumerate(labels):
        if minutes[index] <= current < minutes[index + 1]:
            start = started_at.replace(
                hour=minutes[index] // 60,
                minute=minutes[index] % 60,
                second=0,
                microsecond=0,
            )
            end = started_at.replace(
                hour=minutes[index + 1] // 60,
                minute=minutes[index + 1] % 60,
                second=0,
                microsecond=0,
            )
            return label, start, end
    return None


def _parse_category_targets(value: str) -> dict[str, int]:
    targets: dict[str, int] = {}
    for item in value.split(","):
        if ":" not in item:
            continue
        name, count = item.split(":", 1)
        try:
            targets[name.strip()] = max(0, int(count.strip()))
        except ValueError:
            continue
    return targets


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _normalized_board_roi(
    payload: dict[str, Any],
    *,
    width_scale: float,
    height_scale: float,
) -> tuple[float, float, float, float] | None:
    """Map a board pixel box to a clamped normalized expanded ROI."""
    raw_box = payload.get("best_box") or payload.get("box")
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return None
    try:
        x, y, width, height = (float(value) for value in raw_box)
        frame_width = float(payload.get("frame_width") or 640)
        frame_height = float(payload.get("frame_height") or 360)
    except (TypeError, ValueError):
        return None
    if frame_width <= 0 or frame_height <= 0 or width <= 0 or height <= 0:
        return None

    left = x / frame_width
    top = y / frame_height
    box_width = width / frame_width
    box_height = height / frame_height
    center_x = left + box_width / 2
    center_y = top + box_height / 2
    expanded_width = min(1.0, box_width * max(1.0, width_scale))
    expanded_height = min(1.0, box_height * max(1.0, height_scale))
    roi_left = max(0.0, center_x - expanded_width / 2)
    roi_top = max(0.0, center_y - expanded_height / 2)
    roi_right = min(1.0, center_x + expanded_width / 2)
    roi_bottom = min(1.0, center_y + expanded_height / 2)
    if roi_right <= roi_left or roi_bottom <= roi_top:
        return None
    return roi_left, roi_top, roi_right - roi_left, roi_bottom - roi_top


def _segments_cover_window(
    segments: list[dict[str, Any]],
    start: datetime,
    end: datetime,
    *,
    gap_tolerance_seconds: float = 2.0,
) -> bool:
    """Return whether ordered segment metadata continuously covers a window."""
    cursor = start
    tolerance = timedelta(seconds=max(0.0, gap_tolerance_seconds))
    for segment in sorted(segments, key=lambda row: _parse_iso(str(row["started_at"]))):
        segment_start = _parse_iso(str(segment["started_at"]))
        segment_end = _parse_iso(str(segment["ended_at"]))
        if segment_end <= cursor:
            continue
        if segment_start > cursor + tolerance:
            return False
        cursor = max(cursor, segment_end)
        if cursor >= end:
            return True
    return cursor >= end


def _stream_alignment_snapshot(
    low_segments: list[dict[str, Any]],
    high_segments: list[dict[str, Any]],
    *,
    tolerance_seconds: float,
    required_samples: int,
    segment_seconds: int,
) -> dict[str, Any]:
    required = max(1, required_samples)
    result: dict[str, Any] = {
        "status": "insufficient",
        "offset_seconds": None,
        "tolerance_seconds": tolerance_seconds,
        "paired_segments": 0,
        "required_samples": required,
        "updated_at": utc_now_iso(),
    }
    if low_segments:
        result["latest_low"] = low_segments[0]["path"]
        result["latest_low_started_at"] = low_segments[0]["started_at"]
    if high_segments:
        result["latest_high"] = high_segments[0]["path"]
        result["latest_high_started_at"] = high_segments[0]["started_at"]
    if not low_segments or not high_segments:
        return result

    available_high = list(high_segments)
    offsets: list[float] = []
    max_pair_distance = max(float(segment_seconds) / 2, tolerance_seconds * 4, 5.0)
    for low in low_segments:
        if not available_high or len(offsets) >= required:
            break
        low_started = _parse_iso(low["started_at"])
        nearest = min(
            available_high,
            key=lambda row: abs(
                (_parse_iso(row["started_at"]) - low_started).total_seconds()
            ),
        )
        offset = (_parse_iso(nearest["started_at"]) - low_started).total_seconds()
        if abs(offset) <= max_pair_distance:
            offsets.append(offset)
            available_high.remove(nearest)

    result["paired_segments"] = len(offsets)
    if offsets:
        result["offset_seconds"] = round(float(median(offsets)), 3)
        result["max_abs_offset_seconds"] = round(max(abs(v) for v in offsets), 3)
    if len(offsets) < required:
        return result
    result["status"] = (
        "stable"
        if abs(float(result["offset_seconds"])) <= tolerance_seconds
        else "drifted"
    )
    return result


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug[:64] or "family-moment"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate unique path for {path}")


def _append_daily_summary(
    *,
    day_dir: Path,
    clip_path: Path,
    result: AnalysisResult,
    clip_start: datetime,
    clip_end: datetime,
) -> Path:
    summary_path = day_dir / "summary.md"
    if not summary_path.exists():
        summary_path.write_text(
            f"# Family Moments - {clip_start.strftime('%Y-%m-%d')}\n\n",
            encoding="utf-8",
        )

    tags = ", ".join(result.tags) if result.tags else "untagged"
    entry = (
        f"## {clip_start.strftime('%H:%M:%S')} - {result.title}\n\n"
        f"- Clip: [{clip_path.name}]({clip_path.name})\n"
        f"- Time: {clip_start.isoformat(timespec='seconds')} to {clip_end.isoformat(timespec='seconds')}\n"
        f"- Confidence: {result.confidence:.2f}\n"
        f"- Tags: {tags}\n\n"
        f"{result.summary}\n\n"
    )
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    return summary_path


@dataclass(frozen=True)
class CapDecision:
    """Outcome of a per-day or per-period moment-cap check.

    ``outcome`` is one of:
      * ``"ok"``      — under the cap; save normally, no eviction.
      * ``"evict"``   — at the cap but the new clip is stronger; ``weakest``
                        should be evicted before saving.
      * ``"blocked"`` — at the cap and the new clip is not stronger; skip it.
    """

    outcome: str
    scope: str
    weakest: dict[str, Any] | None = None


@dataclass(frozen=True)
class MomentSkip:
    """A decision to not save the current candidate, with the event to log."""

    event_type: str
    message: str


@dataclass(frozen=True)
class CapPlan:
    skip: MomentSkip | None = None
    evictions: tuple[tuple[dict[str, Any], str, str], ...] = ()


@dataclass(frozen=True)
class MomentSaveOutcome:
    moment_id: int = -1
    skip: MomentSkip | None = None


class Supervisor:
    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.database = database
        self.stop_event = asyncio.Event()
        self.tasks: list[asyncio.Task[None]] = []
        self._llama_timeout_count = 0
        self._llama_circuit_open_until: datetime | None = None
        self._last_board_saved_at: str | None = None
        self._last_board_moment_id: int | None = None
        self._board_probable_detector: DaughterDetector | None = None
        self._probable_verified = 0
        self._probable_rejected = 0
        self._probable_verify_errors = 0
        self._probable_face_verified = 0
        self._probable_faceless_accepted = 0
        self._probable_adult_rejected = 0
        self._probable_face_without_child_rejected = 0
        self._probable_no_person_rejected = 0
        self._probable_inconclusive_rejected = 0
        # Board sessions and MQTT event deduplication live in SQLite so restarts
        # cannot lose a trigger that has not yet produced a moment.
        # NAS analysis and board-triggered saves run as separate tasks.
        # Serialize cap checks with clip publication so both cannot observe
        # count=23 and independently publish the 24th/25th moments.
        self._moment_save_lock = asyncio.Lock()
        self.state: dict[str, Any] = {
            "started_at": utc_now_iso(),
            "recorders": {
                "low": {"status": "not-started"},
                "high": {"status": "not-started"},
            },
            "scanner": {"status": "not-started"},
            "prefilter": {"status": "not-started"},
            "stream_alignment": {"status": "unknown"},
            "analyzer": {"status": "not-started"},
            "mqtt": {"status": "not-started"},
            "rv1106": {"status": "not-started"},
            "cleanup": {"status": "not-started"},
            "day_archive": {"status": "not-started"},
        }

    async def start(self) -> None:
        if not self.settings.workers_enabled:
            self.state["workers"] = "disabled"
            return

        self.tasks.append(asyncio.create_task(self._scan_loop(), name="segment-scanner"))
        self.tasks.append(asyncio.create_task(self._cleanup_loop(), name="buffer-cleanup"))
        self.tasks.append(asyncio.create_task(self._analyzer_loop(), name="segment-analyzer"))
        self.tasks.append(asyncio.create_task(self._day_archive_loop(), name="day-archive"))
        if self.settings.mqtt_enabled:
            self.tasks.append(asyncio.create_task(self._mqtt_loop(), name="mqtt-subscriber"))
            self.tasks.append(
                asyncio.create_task(self._board_events_loop(), name="board-events")
            )
        else:
            self.state["mqtt"] = {"status": "disabled", "reason": "MQTT_ENABLED=false"}

        low_rtsp_url = self.settings.rtsp_low_url_for_ffmpeg
        high_rtsp_url = self.settings.rtsp_high_url_for_ffmpeg

        if low_rtsp_url:
            self.tasks.append(
                asyncio.create_task(
                    self._recorder_loop("low", low_rtsp_url),
                    name="recorder-low",
                )
            )
        else:
            self.state["recorders"]["low"] = {"status": "disabled", "reason": "RTSP_LOW_URL is empty"}

        if high_rtsp_url:
            self.tasks.append(
                asyncio.create_task(
                    self._recorder_loop("high", high_rtsp_url),
                    name="recorder-high",
                )
            )
        else:
            self.state["recorders"]["high"] = {"status": "disabled", "reason": "RTSP_HIGH_URL is empty"}

    async def stop(self) -> None:
        self.stop_event.set()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def snapshot(self) -> dict[str, Any]:
        return self.state

    def _safe_add_event(self, event_type: str, message: str) -> None:
        try:
            self.database.add_event(event_type, message)
        except Exception:
            # Worker error reporting must never terminate the worker itself.
            pass

    def _set_mqtt_state(self, state: dict[str, Any]) -> None:
        self.state["mqtt"] = {**state, "updated_at": utc_now_iso()}

    def _rv1106_local_state(self) -> dict[str, Any]:
        return {
            "probable_policy": self.settings.rv1106_probable_policy,
            "probable_verified": self._probable_verified,
            "probable_rejected": self._probable_rejected,
            "probable_verify_errors": self._probable_verify_errors,
            "probable_face_verified": self._probable_face_verified,
            "probable_faceless_accepted": self._probable_faceless_accepted,
            "probable_adult_rejected": self._probable_adult_rejected,
            "probable_face_without_child_rejected": (
                self._probable_face_without_child_rejected
            ),
            "probable_no_person_rejected": self._probable_no_person_rejected,
            "probable_inconclusive_rejected": (
                self._probable_inconclusive_rejected
            ),
        }

    async def _mqtt_loop(self) -> None:
        subscriber = MQTTSubscriber(
            host=self.settings.mqtt_host,
            port=self.settings.mqtt_port,
            client_id=self.settings.mqtt_client_id,
            topic=tuple(
                topic
                for topic in (
                    self.settings.mqtt_daughter_topic,
                    self.settings.mqtt_status_topic,
                )
                if topic
            ),
            username=self.settings.mqtt_username,
            password=self.settings.mqtt_password,
            keepalive_seconds=self.settings.mqtt_keepalive_seconds,
        )
        await subscriber.run(
            self._handle_mqtt_message, self._set_mqtt_state, self.stop_event
        )

    async def _handle_mqtt_message(self, topic: str, raw_payload: bytes) -> None:
        if topic == self.settings.mqtt_status_topic:
            try:
                payload = decode_json_payload(raw_payload)
                self.state["rv1106"] = {
                    "status": "online",
                    **payload,
                    **self._rv1106_local_state(),
                    "pending_sessions": self.database.count_pending_board_sessions(),
                    "last_saved_at": self._last_board_saved_at,
                    "last_moment_id": self._last_board_moment_id,
                    "updated_at": utc_now_iso(),
                }
            except Exception as exc:
                self._safe_add_event("mqtt-status-error", str(exc))
            return
        if topic != self.settings.mqtt_daughter_topic:
            return
        try:
            payload = decode_json_payload(raw_payload)
            timestamp = float(payload["ts"])
            score = max(0.0, min(1.0, float(payload["score"])))
            camera_id = str(payload.get("camera_id") or self.settings.camera_name)
            event_state = str(payload.get("event") or "hit").strip().lower()
            identity = str(payload.get("identity") or "confirmed").strip().lower()
            if identity == "probable" and self.settings.rv1106_probable_policy == "reject":
                self.state["mqtt"] = {
                    **self.state.get("mqtt", {}),
                    "status": "connected",
                    "last_ignored_identity": identity,
                    "updated_at": utc_now_iso(),
                }
                return
            session_key, event_key = self._board_event_keys(
                camera_id, payload, timestamp, event_state
            )
            try:
                session_start = float(payload.get("session_start_ts", timestamp))
            except (TypeError, ValueError):
                session_start = timestamp
            try:
                best_ts = float(payload.get("best_ts", timestamp))
            except (TypeError, ValueError):
                best_ts = timestamp
            try:
                session, inserted = await asyncio.to_thread(
                    self.database.record_board_event,
                    event_key=event_key,
                    session_key=session_key,
                    session_id=str(payload.get("session_id") or "").strip(),
                    camera_id=camera_id,
                    session_start=session_start,
                    identity=identity,
                    score=score,
                    best_ts=best_ts,
                    last_event_at=timestamp,
                    payload=payload,
                    event_state=event_state,
                )
            except Exception as exc:
                raise BoardEventPersistenceError(str(exc)) from exc
            now = _now()
            event_end = datetime.fromtimestamp(timestamp).astimezone()
            self.state["mqtt"] = {
                "status": "connected",
                "host": self.settings.mqtt_host,
                "port": self.settings.mqtt_port,
                "last_topic": topic,
                "last_hit_at": event_end.isoformat(timespec="seconds"),
                "last_event": event_state,
                "last_identity": identity,
                "duplicate": not inserted,
                "event_lag_seconds": round(max(0.0, (now - event_end).total_seconds()), 3),
                "updated_at": utc_now_iso(),
            }
            if inserted and event_state == "start":
                self._safe_add_event(
                    "edge-daughter-hit",
                    json.dumps(
                        {
                            "camera_id": camera_id,
                            "score": score,
                            "ts": timestamp,
                            "seq": payload.get("seq"),
                            "event": event_state,
                            "identity": identity,
                            "session_id": session["session_id"],
                        },
                        ensure_ascii=False,
                    ),
                )
        except BoardEventPersistenceError as exc:
            self._safe_add_event("mqtt-persistence-error", str(exc))
            self.state["mqtt"] = {
                **self.state.get("mqtt", {}),
                "status": "persistence-error",
                "message": str(exc),
                "updated_at": utc_now_iso(),
            }
            raise
        except Exception as exc:
            self._safe_add_event("mqtt-message-error", str(exc))
            self.state["mqtt"] = {
                **self.state.get("mqtt", {}),
                "status": "message-error",
                "message": str(exc),
                "updated_at": utc_now_iso(),
            }

    @staticmethod
    def _board_event_keys(
        camera_id: str,
        payload: dict[str, Any],
        timestamp: float,
        event_state: str,
    ) -> tuple[str, str]:
        """Return stable session/event keys for persistence and QoS-1 dedupe."""
        session_id = str(payload.get("session_id") or "").strip()
        sequence = payload.get("seq")
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        event_token = str(sequence) if sequence is not None else digest
        if session_id:
            try:
                session_start = float(payload.get("session_start_ts", timestamp))
            except (TypeError, ValueError):
                session_start = timestamp
            # Track/session counters reset when the board process restarts, so
            # session_id alone is not globally unique. The wall-clock start
            # keeps a later boot's "1-1" session distinct from old history.
            session_key = f"{camera_id}:session:{session_id}:{session_start:.3f}"
        else:
            # Legacy face-only messages are one-shot hits. Giving each unique
            # message its own session prevents a completed legacy row from
            # swallowing later, unrelated detections.
            session_key = f"{camera_id}:legacy:{timestamp:.3f}:{event_token}"
        return session_key, f"{session_key}:{event_state}:{event_token}"

    async def _expire_stale_board_sessions(self) -> int:
        cutoff = _now().timestamp() - max(
            1.0, self.settings.rv1106_session_timeout_seconds
        )
        return await asyncio.to_thread(
            self.database.finalize_stale_board_sessions, cutoff
        )

    async def _board_events_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                expired = await self._expire_stale_board_sessions()
                if expired:
                    self._safe_add_event(
                        "edge-session-expired", f"finalized {expired} stale board session(s)"
                    )
                sessions = await asyncio.to_thread(
                    self.database.pending_board_sessions, limit=20
                )
                for session in sessions:
                    try:
                        await self._save_board_session(session)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        attempts = await asyncio.to_thread(
                            self.database.record_board_session_error,
                            str(session["key"]),
                            str(exc),
                        )
                        self._safe_add_event(
                            "board-event-error",
                            f"{session['key']} attempt {attempts}: {exc}",
                        )
                        if attempts >= _BOARD_SESSION_MAX_PROCESS_ATTEMPTS:
                            await asyncio.to_thread(
                                self.database.mark_board_session_skipped,
                                str(session["key"]),
                                f"processing failed after {attempts} attempts: {exc}",
                            )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._safe_add_event("board-event-error", str(exc))
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(), timeout=_BOARD_SESSION_SCAN_SECONDS
                )
            except asyncio.TimeoutError:
                pass

    async def _save_board_session(self, session: dict[str, Any]) -> None:
        session_key = str(session["key"])
        existing = await asyncio.to_thread(
            self.database.get_moment_by_trigger_key, session_key
        )
        if existing is not None:
            moment_id = int(existing["id"])
            await asyncio.to_thread(
                self.database.mark_board_session_saved, session_key, moment_id
            )
            self._last_board_saved_at = str(existing["created_at"])
            self._last_board_moment_id = moment_id
            self.state["rv1106"] = {
                **self.state.get("rv1106", {}),
                **self._rv1106_local_state(),
                "last_saved_at": self._last_board_saved_at,
                "last_moment_id": self._last_board_moment_id,
                "pending_sessions": self.database.count_pending_board_sessions(),
            }
            return

        identity = str(session["identity"]).lower()
        score = float(session["score"])
        event_time = datetime.fromtimestamp(float(session["best_ts"])).astimezone()
        wanted_start = event_time - timedelta(seconds=self.settings.context_before_seconds)
        wanted_end = event_time + timedelta(seconds=self.settings.context_after_seconds)
        segment = await asyncio.to_thread(
            self.database.find_segment_near,
            stream_role="high",
            timestamp=event_time.isoformat(timespec="milliseconds"),
            tolerance_seconds=self.settings.stream_alignment_tolerance_seconds,
        )
        high_ready = False
        if segment and Path(str(segment["path"])).exists():
            high_segments = await asyncio.to_thread(
                self.database.find_segments_between,
                stream_role="high",
                started_before=wanted_end.isoformat(timespec="seconds"),
                ended_after=wanted_start.isoformat(timespec="seconds"),
            )
            available_high = [
                row for row in high_segments if Path(str(row["path"])).exists()
            ]
            high_ready = bool(available_high) and _segments_cover_window(
                available_high,
                wanted_start,
                wanted_end,
                gap_tolerance_seconds=self.settings.stream_alignment_tolerance_seconds,
            )

        wait_age = max(0.0, _now().timestamp() - float(session["last_event_at"]))
        save_wait_seconds = max(1.0, self.settings.rv1106_save_wait_seconds)
        if not segment or not Path(str(segment["path"])).exists():
            if wait_age < save_wait_seconds:
                return
            reason = f"no 4K segment for board session {session_key}"
            self._safe_add_event(
                "board-session-dropped",
                reason,
            )
            await asyncio.to_thread(
                self.database.mark_board_session_skipped, session_key, reason
            )
            return
        if not high_ready:
            if wait_age < save_wait_seconds:
                return
            reason = f"4K coverage incomplete for board session {session_key}"
            self._safe_add_event(
                "board-session-dropped",
                reason,
            )
            await asyncio.to_thread(
                self.database.mark_board_session_skipped, session_key, reason
            )
            return

        source_start = _parse_iso(str(segment["started_at"]))
        start_offset = max(0, int((event_time - source_start).total_seconds()))
        payload_value = session.get("payload") or {}
        payload = payload_value if isinstance(payload_value, dict) else {}
        activity_score = max(0.0, min(1.0, float(payload.get("activity_score") or 0.0)))
        confirmed = identity == "confirmed"
        verification_details: dict[str, Any] | None = None
        if identity == "probable" and self.settings.rv1106_probable_policy == "verify":
            verification, verification_details = await self._verify_board_probable_event(
                session, event_time
            )
            if not verification.accepted:
                reason = (
                    "probable verification rejected: "
                    f"{verification.decision}: {verification.reason}"
                )
                await asyncio.to_thread(
                    self.database.mark_board_session_skipped, session_key, reason
                )
                self._update_rv1106_runtime_state()
                return

        faceless = bool(
            verification_details
            and verification_details.get("decision") == "faceless_person"
        )
        face_verified = bool(
            verification_details
            and verification_details.get("decision")
            in {"child_face", "daughter_onnx"}
        )
        base_selection_score = min(1.0, score * 0.75 + activity_score * 0.25)
        selection_score = (
            base_selection_score
            * max(0.0, self.settings.rv1106_verify_faceless_score_multiplier)
            if faceless
            else base_selection_score
        )
        result = ClipCandidate(
            keep=True,
            title=(
                "Daughter confirmed by RV1106"
                if confirmed
                else "Probable daughter activity (faceless)"
                if faceless
                else "Probable daughter activity"
            ),
            summary=(
                "开发板通过人脸特征与人体轨迹融合确认女儿出现在画面中。"
                if confirmed
                else "开发板儿童轨迹与 4K ROI 内持续人体证据一致，未要求正脸出现。"
                if faceless
                else "开发板检测到持续稳定的儿童体型活动轨迹，作为高召回候选保存。"
            ),
            tags=[
                "daughter",
                "rv1106",
                identity,
                "person_tracking",
                *(["faceless"] if faceless else []),
                *(["child_face"] if face_verified else []),
            ],
            confidence=score,
            start_offset_seconds=start_offset,
            end_offset_seconds=start_offset + 1,
            raw={
                "board": payload,
                "session_id": session.get("session_id"),
                "board_session_key": session_key,
                "identity": identity,
                "activity_score": activity_score,
                "base_selection_score": base_selection_score,
                **(
                    {"probable_verification": verification_details}
                    if verification_details
                    else {}
                ),
            },
            local_child_confirmed=confirmed or face_verified,
            local_child_score=score,
            analysis_backend="rv1106_edge",
            category=(
                "rv1106_probable_faceless"
                if faceless
                else "rv1106_probable_face"
                if face_verified
                else f"rv1106_{identity}"
            ),
            selection_score=min(1.0, selection_score),
        )
        outcome = await self._save_capped_moment(
            segment, result, analyzer=None, detector=None
        )
        if outcome.skip is not None:
            await asyncio.to_thread(
                self.database.mark_board_session_skipped,
                session_key,
                outcome.skip.event_type,
            )
        if outcome.moment_id >= 0:
            await asyncio.to_thread(
                self.database.mark_board_session_saved,
                session_key,
                outcome.moment_id,
            )
            self._last_board_saved_at = utc_now_iso()
            self._last_board_moment_id = outcome.moment_id
            self._safe_add_event(
                "moment", f"saved edge-triggered moment {outcome.moment_id}"
            )
        self._update_rv1106_runtime_state()

    def _update_rv1106_runtime_state(self) -> None:
        self.state["rv1106"] = {
            **self.state.get("rv1106", {}),
            **self._rv1106_local_state(),
            "last_saved_at": self._last_board_saved_at,
            "last_moment_id": self._last_board_moment_id,
            "pending_sessions": self.database.count_pending_board_sessions(),
        }

    def _get_board_probable_detector(self) -> DaughterDetector:
        if self._board_probable_detector is None:
            self._board_probable_detector = DaughterDetector(
                replace(
                    self.settings,
                    daughter_age_check_every=1,
                    daughter_body_fallback_enabled=False,
                )
            )
        return self._board_probable_detector

    @staticmethod
    def _probable_verification_failure(
        decision: str,
        reason: str,
        *,
        sampled_frames: int = 0,
    ) -> ProbableVerification:
        return ProbableVerification(
            accepted=False,
            positive_frames=0,
            required_frames=1,
            evidence=(),
            reason=reason,
            decision=decision,
            sampled_frames=sampled_frames,
        )

    def _record_probable_verification(
        self,
        verification: ProbableVerification,
        details: dict[str, Any],
    ) -> None:
        if verification.accepted:
            self._probable_verified += 1
            if verification.decision in {"child_face", "daughter_onnx"}:
                self._probable_face_verified += 1
            elif verification.decision == "faceless_person":
                self._probable_faceless_accepted += 1
            event_type = "rv1106-probable-verified"
        else:
            self._probable_rejected += 1
            if verification.decision == "adult_face":
                self._probable_adult_rejected += 1
            elif verification.decision == "face_without_child":
                self._probable_face_without_child_rejected += 1
            elif verification.decision == "no_person":
                self._probable_no_person_rejected += 1
            else:
                self._probable_inconclusive_rejected += 1
            if verification.decision == "verification_error":
                self._probable_verify_errors += 1
                event_type = "rv1106-probable-verify-error"
            else:
                event_type = "rv1106-probable-rejected"
        self._safe_add_event(
            event_type, json.dumps(details, ensure_ascii=False, sort_keys=True)
        )

    async def _verify_board_probable_event(
        self, session: dict[str, Any], event_time: datetime
    ) -> tuple[ProbableVerification, dict[str, Any]]:
        payload_value = session.get("payload") or {}
        payload = payload_value if isinstance(payload_value, dict) else {}
        session_key = str(session.get("key") or "")
        roi = _normalized_board_roi(
            payload,
            width_scale=self.settings.rv1106_verify_roi_width_scale,
            height_scale=self.settings.rv1106_verify_roi_height_scale,
        )
        base_details: dict[str, Any] = {
            "session_key": session_key,
            "board_score": float(session.get("score") or 0.0),
            "board_person_score": float(payload.get("person_score") or 0.0),
            "board_box": payload.get("best_box") or payload.get("box"),
            "frame_width": payload.get("frame_width") or 640,
            "frame_height": payload.get("frame_height") or 360,
            "sample_offsets_seconds": [-2, -1, 0, 1, 2],
        }
        if roi is None:
            verification = self._probable_verification_failure(
                "invalid_roi", "missing or invalid board person box"
            )
            details = {**base_details, **asdict(verification), "roi": None}
            self._record_probable_verification(verification, details)
            return verification, details

        base_details["roi"] = [round(value, 6) for value in roi]
        offsets = (-2.0, -1.0, 0.0, 1.0, 2.0)
        try:
            with tempfile.TemporaryDirectory(
                prefix="nas-video-board-roi-verify-"
            ) as temp_dir:
                frames: list[SampledFrame] = []
                for index, relative_offset in enumerate(offsets, start=1):
                    sample_time = event_time + timedelta(seconds=relative_offset)
                    segment = await asyncio.to_thread(
                        self.database.find_segment_near,
                        stream_role="high",
                        timestamp=sample_time.isoformat(timespec="milliseconds"),
                        tolerance_seconds=(
                            self.settings.stream_alignment_tolerance_seconds
                        ),
                    )
                    if segment is None or not Path(str(segment["path"])).exists():
                        break
                    segment_start = _parse_iso(str(segment["started_at"]))
                    segment_end = _parse_iso(str(segment["ended_at"]))
                    segment_duration = max(
                        0.1, (segment_end - segment_start).total_seconds()
                    )
                    seek_offset = (sample_time - segment_start).total_seconds()
                    seek_offset = max(
                        0.0, min(seek_offset, max(0.0, segment_duration - 0.1))
                    )
                    frame_path = Path(temp_dir) / f"board-roi-{index}.jpg"
                    await extract_cropped_frame(
                        self.settings,
                        Path(str(segment["path"])),
                        frame_path,
                        seek_offset,
                        roi=roi,
                        output_width=self.settings.rv1106_verify_frame_width,
                    )
                    if not frame_path.exists():
                        break
                    frames.append(
                        SampledFrame(
                            path=frame_path, offset_seconds=relative_offset
                        )
                    )

                if len(frames) != len(offsets):
                    verification = self._probable_verification_failure(
                        "incomplete_frames",
                        f"expected {len(offsets)} ROI frames, got {len(frames)}",
                        sampled_frames=len(frames),
                    )
                else:
                    detector = self._get_board_probable_detector()
                    verification = await asyncio.to_thread(
                        detector.verify_board_probable_paths,
                        [frame.path for frame in frames],
                        required_frames=2,
                        required_person_frames=(
                            self.settings.rv1106_verify_person_frames
                        ),
                        board_score=float(session.get("score") or 0.0),
                        board_person_score=float(
                            payload.get("person_score") or 0.0
                        ),
                        board_score_threshold=(
                            self.settings.rv1106_verify_board_score
                        ),
                        board_person_score_threshold=(
                            self.settings.rv1106_verify_board_person_score
                        ),
                    )
            details = {**base_details, **asdict(verification)}
            self._record_probable_verification(verification, details)
            return verification, details
        except Exception as exc:
            verification = self._probable_verification_failure(
                "verification_error", str(exc)
            )
            details = {**base_details, **asdict(verification)}
            self._record_probable_verification(verification, details)
            return verification, details

    async def _recorder_loop(self, role: str, rtsp_url: str) -> None:
        while not self.stop_event.is_set():
            if not ffmpeg_available(self.settings):
                self.state["recorders"][role] = {
                    "status": "error",
                    "message": f"{self.settings.ffmpeg_bin} not found on PATH",
                    "updated_at": utc_now_iso(),
                }
                await asyncio.sleep(30)
                continue
            if not _in_record_window(
                _now(),
                self.settings.record_window_start,
                self.settings.record_window_end,
            ):
                self.state["recorders"][role] = {
                    "status": "waiting_for_record_window",
                    "window_start": self.settings.record_window_start,
                    "window_end": self.settings.record_window_end,
                    "updated_at": utc_now_iso(),
                }
                await asyncio.sleep(60)
                continue

            command = build_recorder_command(self.settings, role, rtsp_url)
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self.state["recorders"][role] = {
                "status": "running",
                "pid": process.pid,
                "updated_at": utc_now_iso(),
            }
            try:
                window_closed = False
                while process.returncode is None and not self.stop_event.is_set():
                    await asyncio.sleep(1)
                    if not _in_record_window(
                        _now(),
                        self.settings.record_window_start,
                        self.settings.record_window_end,
                    ):
                        window_closed = True
                        await _stop_process(process)
                        break
                if self.stop_event.is_set() and process.returncode is None:
                    await _stop_process(process)
                    return
                if window_closed:
                    self.state["recorders"][role] = {
                        "status": "waiting_for_record_window",
                        "window_start": self.settings.record_window_start,
                        "window_end": self.settings.record_window_end,
                        "updated_at": utc_now_iso(),
                    }
                    self.database.add_event(
                        "recorder-window",
                        f"{role} recorder stopped at window end",
                    )
                    continue
                return_code = await process.wait()
                self.state["recorders"][role] = {
                    "status": "restarting",
                    "return_code": return_code,
                    "updated_at": utc_now_iso(),
                }
                self.database.add_event("recorder", f"{role} recorder exited with code {return_code}")
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await _stop_process(process)
                raise

    async def _scan_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                count = await asyncio.to_thread(self._scan_once)
                self.state["scanner"] = {
                    "status": "ok",
                    "last_scan_at": utc_now_iso(),
                    "segments_seen": count,
                }
                self._update_stream_alignment()
            except Exception as exc:
                self.state["scanner"] = {
                    "status": "error",
                    "message": str(exc),
                    "updated_at": utc_now_iso(),
                }
                self._safe_add_event("scanner-error", str(exc))
            await asyncio.sleep(10)

    def _update_stream_alignment(self) -> None:
        if not self.settings.rtsp_low_url and self.settings.rtsp_high_url:
            alignment = {
                "status": "not_applicable",
                "reason": "high-only recording mode",
                "updated_at": utc_now_iso(),
            }
            self.state["stream_alignment"] = alignment
            return
        sample_count = max(1, self.settings.stream_alignment_sample_count)
        fetch_limit = max(sample_count * 2, sample_count)
        alignment = _stream_alignment_snapshot(
            self.database.recent_segments("low", limit=fetch_limit),
            self.database.recent_segments("high", limit=fetch_limit),
            tolerance_seconds=self.settings.stream_alignment_tolerance_seconds,
            required_samples=sample_count,
            segment_seconds=self.settings.segment_seconds,
        )
        previous = self.state.get("stream_alignment", {}).get("status")
        self.state["stream_alignment"] = alignment
        status = alignment["status"]
        if status != previous and status in {"stable", "drifted", "insufficient"}:
            self._safe_add_event(
                f"stream-alignment-{status}",
                json.dumps(alignment, ensure_ascii=False),
            )

    def _scan_once(self) -> int:
        now = _now()
        segments: list[dict[str, Any]] = []
        for directory, role in (
            (self.settings.low_buffer_dir, "low"),
            (self.settings.high_buffer_dir, "high"),
        ):
            directory.mkdir(parents=True, exist_ok=True)
            for path in sorted(directory.glob("*.mp4")):
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                file_age = now.timestamp() - stat.st_mtime
                if file_age < self.settings.segment_stable_seconds:
                    continue

                parsed = parse_segment_filename(path)
                if parsed is None:
                    started_at = datetime.fromtimestamp(stat.st_mtime).astimezone()
                    camera_name = self.settings.camera_name
                else:
                    camera_name, parsed_role, started_at = parsed
                    if parsed_role != role:
                        continue

                started, ended = segment_time_window(started_at, self.settings.segment_seconds)
                segments.append(
                    {
                        "camera_name": camera_name,
                        "stream_role": role,
                        "path": path,
                        "started_at": started.isoformat(timespec="seconds"),
                        "ended_at": ended.isoformat(timespec="seconds"),
                        "duration_seconds": self.settings.segment_seconds,
                        "size_bytes": stat.st_size,
                    }
                )
        self.database.upsert_segments(segments)
        return len(segments)

    def _moment_cooldown_active(
        self, segment: dict[str, Any] | None = None, result: ClipCandidate | None = None
    ) -> bool:
        if self.settings.moment_cooldown_seconds <= 0:
            return False
        if segment is None or result is None or not hasattr(self.database, "nearest_moment_before"):
            return False
        candidate = _parse_iso(segment["started_at"]) + timedelta(
            seconds=result.start_offset_seconds + self.settings.camera_time_offset_seconds
        )
        previous = self.database.nearest_moment_before(candidate.isoformat(timespec="seconds"))
        if not previous or not previous.get("clip_started_at"):
            return False
        gap = (candidate - _parse_iso(str(previous["clip_started_at"]))).total_seconds()
        return 0 <= gap < self.settings.moment_cooldown_seconds

    def _display_started_at(self, segment: dict[str, Any]) -> datetime | None:
        value = str(segment.get("started_at", ""))
        if not value:
            return None
        return _parse_iso(value) + timedelta(
            seconds=self.settings.camera_time_offset_seconds
        )

    def _daily_cap_decision(
        self, segment: dict[str, Any], result: AnalysisResult
    ) -> CapDecision:
        """Decide the daily keep-best-N cap for this candidate (see CapDecision)."""
        cap = self.settings.max_moments_per_day
        if cap <= 0:
            return CapDecision("ok", "daily")
        started_at = self._display_started_at(segment)
        if started_at is None:
            return CapDecision("ok", "daily")
        day = started_at.strftime("%Y-%m-%d")
        if self.database.count_moments_on_day(day) < cap:
            return CapDecision("ok", "daily")
        weakest = self.database.weakest_moment_on_day(day)
        if weakest is None:
            return CapDecision("ok", "daily")
        if result.effective_selection_score > float(weakest.get("selection_score", weakest["confidence"])):
            return CapDecision("evict", "daily", weakest)
        return CapDecision("blocked", "daily")

    def _period_cap_decision(
        self, segment: dict[str, Any], result: AnalysisResult
    ) -> CapDecision:
        """Decide the per-period keep-best-N cap for this candidate.

        ``scope`` carries the period label ("morning"/"afternoon"/"evening"),
        or "" when the period cap does not apply (disabled, no timestamp, or
        the timestamp falls outside every configured period).
        """
        cap = self.settings.max_moments_per_period
        if cap <= 0:
            return CapDecision("ok", "")
        started_at = self._display_started_at(segment)
        if started_at is None:
            return CapDecision("ok", "")
        period = _moment_period(started_at, self.settings.moment_period_boundaries)
        if period is None:
            return CapDecision("ok", "")
        label, start, end = period
        start_iso = start.isoformat(timespec="seconds")
        end_iso = end.isoformat(timespec="seconds")
        if self.database.count_moments_between(start_iso, end_iso) < cap:
            return CapDecision("ok", label)
        weakest = self.database.weakest_moment_between(start_iso, end_iso)
        if weakest is None:
            return CapDecision("ok", label)
        if result.analysis_backend == "daughter_detector" and hasattr(
            self.database, "moments_between"
        ):
            target_map = _parse_category_targets(self.settings.moment_category_targets)
            target = target_map.get(result.category, 0)
            existing = self.database.moments_between(start_iso, end_iso)
            category_counts = {
                category: sum(item.get("category") == category for item in existing)
                for category in target_map
            }
            if target > 0 and category_counts.get(result.category, 0) < target:
                overrepresented = [
                    item for item in existing
                    if category_counts.get(str(item.get("category")), 0)
                    > target_map.get(str(item.get("category")), 0)
                ]
                if overrepresented:
                    return CapDecision(
                        "evict", label,
                        min(overrepresented, key=lambda item: float(item.get("selection_score", item["confidence"])))
                    )
        if result.effective_selection_score > float(weakest.get("selection_score", weakest["confidence"])):
            return CapDecision("evict", label, weakest)
        return CapDecision("blocked", label)

    def _evict_moment(
        self,
        moment: dict[str, Any],
        *,
        event_type: str = "daily-cap-evict",
        scope: str = "daily",
    ) -> None:
        for key in ("clip_path", "metadata_path"):
            path = Path(str(moment[key]))
            path.unlink(missing_ok=True)
        self.database.delete_moment_by_clip(str(moment["clip_path"]))
        day = Path(str(moment["clip_path"])).parent.name
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            rebuild_day_archive(self.settings, self.database, day)
        self.database.add_event(
            event_type,
            f"replaced weakest {scope} moment '{moment['title']}' "
            f"(score {float(moment.get('selection_score', moment['confidence'])):.2f})",
        )

    def _apply_moment_caps(
        self, segment: dict[str, Any], result: AnalysisResult
    ) -> CapPlan:
        """Apply cooldown + period + daily caps for a keep-worthy candidate.

        Return a plan without deleting anything. Evictions are applied only
        after the new clip has passed final verification and been registered.
        This prevents a false positive or ffmpeg failure from deleting an old
        good moment.
        """
        if self._moment_cooldown_active(segment, result):
            return CapPlan(
                skip=MomentSkip(
                    "moment-cooldown", f"skipped '{result.title}' due to cooldown"
                )
            )

        evictions: list[tuple[dict[str, Any], str, str]] = []
        period = self._period_cap_decision(segment, result)
        if period.outcome == "blocked":
            return CapPlan(
                skip=MomentSkip(
                    "period-cap",
                    f"skipped '{result.title}' ({period.scope} limit "
                    f"{self.settings.max_moments_per_period} reached, "
                    f"confidence {result.confidence:.2f} not above weakest)",
                )
            )
        if period.outcome == "evict":
            assert period.weakest is not None
            evictions.append((period.weakest, "period-cap-evict", period.scope))

        display_started = self._display_started_at(segment)
        day = display_started.strftime("%Y-%m-%d") if display_started else ""
        daily_count = self.database.count_moments_on_day(day) if day else 0
        # A period eviction belongs to this same source day, so account for it
        # before applying the daily cap without mutating the database yet.
        daily_count -= len(evictions)
        daily = self._daily_cap_decision_after_count(
            segment, result, daily_count
        )
        if daily.outcome == "blocked":
            return CapPlan(
                skip=MomentSkip(
                    "daily-cap",
                    f"skipped '{result.title}' (daily limit "
                    f"{self.settings.max_moments_per_day} reached, "
                    f"confidence {result.confidence:.2f} not above weakest)",
                )
            )
        if daily.outcome == "evict":
            assert daily.weakest is not None
            if not any(item[0]["id"] == daily.weakest["id"] for item in evictions):
                evictions.append((daily.weakest, "daily-cap-evict", "daily"))
        return CapPlan(evictions=tuple(evictions))

    def _daily_cap_decision_after_count(
        self,
        segment: dict[str, Any],
        result: AnalysisResult,
        count: int,
    ) -> CapDecision:
        cap = self.settings.max_moments_per_day
        if cap <= 0 or count < cap:
            return CapDecision("ok", "daily")
        started_at = self._display_started_at(segment)
        if started_at is None:
            return CapDecision("ok", "daily")
        day = started_at.strftime("%Y-%m-%d")
        weakest = self.database.weakest_moment_on_day(day)
        if weakest is None:
            return CapDecision("ok", "daily")
        if result.effective_selection_score > float(weakest.get("selection_score", weakest["confidence"])):
            return CapDecision("evict", "daily", weakest)
        return CapDecision("blocked", "daily")

    async def _save_capped_moment(
        self,
        segment: dict[str, Any],
        result: ClipCandidate,
        analyzer: LlamaAnalyzer | None,
        detector: DaughterDetector | None = None,
    ) -> MomentSaveOutcome:
        """Apply every clip limit atomically across all producer tasks."""
        async with self._moment_save_lock:
            cap_plan = self._apply_moment_caps(segment, result)
            if cap_plan.skip is not None:
                self.database.add_event(
                    cap_plan.skip.event_type, cap_plan.skip.message
                )
                return MomentSaveOutcome(skip=cap_plan.skip)
            moment_id = await self._save_moment(
                segment, result, analyzer=analyzer, detector=detector
            )
            if moment_id >= 0:
                for moment, event_type, scope in cap_plan.evictions:
                    self._evict_moment(
                        moment, event_type=event_type, scope=scope
                    )
            return MomentSaveOutcome(moment_id=moment_id)

    async def _analyzer_loop(self) -> None:
        backend = self.settings.analysis_backend
        analyzer = LlamaAnalyzer(self.settings) if backend == "vlm" else None
        detector = DaughterDetector(self.settings) if backend == "daughter_detector" else None
        while not self.stop_event.is_set():
            if not self.settings.analysis_enabled:
                self.state["analyzer"] = {"status": "disabled", "reason": "ANALYSIS_ENABLED=false"}
                await asyncio.sleep(self.settings.analysis_interval_seconds)
                continue
            if backend not in {"daughter_detector", "rv1106"} and not _in_analysis_window(
                _now(),
                self.settings.analysis_window_start,
                self.settings.analysis_window_end,
            ):
                self.state["analyzer"] = {
                    "status": "waiting_for_window",
                    "window_start": self.settings.analysis_window_start,
                    "window_end": self.settings.analysis_window_end,
                    "updated_at": utc_now_iso(),
                }
                await asyncio.sleep(60)
                continue
            if not ffmpeg_available(self.settings):
                self.state["analyzer"] = {
                    "status": "waiting",
                    "message": f"{self.settings.ffmpeg_bin} not found on PATH",
                    "updated_at": utc_now_iso(),
                }
                await asyncio.sleep(self.settings.analysis_interval_seconds)
                continue
            if backend == "vlm" and (
                self._llama_circuit_open_until is not None
                and _now() < self._llama_circuit_open_until
            ):
                self.state["analyzer"] = {
                    "status": "circuit-open",
                    "resume_at": self._llama_circuit_open_until.isoformat(timespec="seconds"),
                    "consecutive_timeouts": self._llama_timeout_count,
                    "updated_at": utc_now_iso(),
                }
                await asyncio.sleep(min(30, self.settings.analysis_interval_seconds))
                continue

            ready_before = (_now() - timedelta(seconds=self.settings.analysis_delay_seconds)).isoformat(
                timespec="seconds"
            )
            segments = self.database.get_pending_segments(
                stream_role=self.settings.analysis_stream_role,
                ready_before=ready_before,
                max_attempts=self.settings.analysis_max_attempts,
                limit=1,
            )
            if not segments:
                self.state["analyzer"] = {
                    "status": "idle",
                    "updated_at": utc_now_iso(),
                }
                await asyncio.sleep(self.settings.analysis_interval_seconds)
                continue

            segment = segments[0]
            if backend == "daughter_detector" and not _in_analysis_window(
                _parse_iso(segment["started_at"]),
                self.settings.analysis_window_start,
                self.settings.analysis_window_end,
            ):
                self.database.add_event(
                    "analysis-window-skip",
                    f"detector skipped segment outside source-time window: {segment['path']}",
                )
                self.database.mark_segment_processed(int(segment["id"]))
                continue
            self.state["analyzer"] = {
                "status": "analyzing",
                "segment": segment["path"],
                "updated_at": utc_now_iso(),
            }
            if backend == "rv1106":
                self.database.mark_segment_processed(int(segment["id"]))
                self.state["analyzer"] = {
                    "status": "edge-only",
                    "last_segment": segment["path"],
                    "last_keep": False,
                    "backend": backend,
                    "updated_at": utc_now_iso(),
                }
                continue
            try:
                if backend == "vlm":
                    assert analyzer is not None
                    results = [await self._analyze_segment(analyzer, segment)]
                elif backend == "daughter_detector":
                    assert detector is not None
                    results = await self._analyze_daughter_segment(detector, segment)
                else:
                    raise ValueError(f"unsupported ANALYSIS_BACKEND: {backend}")
                self._llama_timeout_count = 0
                self._llama_circuit_open_until = None
                saved_any = False
                for result in results:
                    keep_threshold = (
                        self.settings.daughter_detector_threshold
                        if result.analysis_backend == "daughter_detector"
                        else self.settings.moment_keep_threshold
                    )
                    should_save = result.should_save(keep_threshold)
                    if result.keep_consistency_repaired(
                        self.settings.moment_keep_threshold
                    ):
                        self.database.add_event(
                            "keep-consistency-repair",
                            "corrected keep=false with local child evidence "
                            f"({result.local_child_score:.2f}): {result.title}",
                        )
                    if should_save:
                        outcome = await self._save_capped_moment(
                            segment, result, analyzer=analyzer, detector=detector
                        )
                        if outcome.moment_id >= 0:
                            saved_any = True
                            self.database.add_event(
                                "moment",
                                f"saved moment {outcome.moment_id}: {result.title}",
                            )
                    if not should_save:
                        self.database.add_event(
                            "analysis-skip",
                            json.dumps(
                                {
                                    "backend": result.analysis_backend,
                                    "keep": result.keep,
                                    "confidence": result.confidence,
                                    "title": result.title,
                                    "tags": result.tags,
                                    "start_offset": result.start_offset_seconds,
                                    "end_offset": result.end_offset_seconds,
                                    "raw_text": result.raw_text[:1000],
                                },
                                ensure_ascii=False,
                            ),
                        )
                if not results and backend == "daughter_detector":
                    self.database.add_event(
                        "daughter-detector-skip", "no qualifying daughter event"
                    )
                self.database.mark_segment_processed(int(segment["id"]))
                self.state["analyzer"] = {
                    "status": "ok",
                    "last_segment": segment["path"],
                    "last_keep": saved_any,
                    "backend": backend,
                    "updated_at": utc_now_iso(),
                }
            except PersonFilterSkip:
                self.state["analyzer"] = {
                    "status": "ok",
                    "last_segment": segment["path"],
                    "last_keep": False,
                    "updated_at": utc_now_iso(),
                }
                if self.settings.analysis_cooldown_seconds > 0:
                    await asyncio.sleep(self.settings.analysis_cooldown_seconds)
                continue
            except Exception as exc:
                if backend == "vlm" and _is_timeout_error(exc):
                    self._llama_timeout_count += 1
                    if (
                        self.settings.llama_circuit_breaker_failures > 0
                        and self._llama_timeout_count
                        >= self.settings.llama_circuit_breaker_failures
                    ):
                        self._llama_circuit_open_until = _now() + timedelta(
                            seconds=self.settings.llama_circuit_breaker_seconds
                        )
                        self.database.add_event(
                            "llama-circuit-open",
                            f"paused analysis for {self.settings.llama_circuit_breaker_seconds}s "
                            f"after {self._llama_timeout_count} consecutive timeouts",
                        )
                elif backend == "vlm":
                    self._llama_timeout_count = 0
                attempt = int(segment["analysis_attempts"]) + 1
                final = attempt >= self.settings.analysis_max_attempts
                try:
                    self.database.record_analysis_error(
                        int(segment["id"]), str(exc), final=final
                    )
                except Exception:
                    pass
                self.state["analyzer"] = {
                    "status": "error",
                    "message": str(exc),
                    "final": final,
                    "updated_at": utc_now_iso(),
                }
                self._safe_add_event("analysis-error", str(exc))
                await asyncio.sleep(self.settings.analysis_interval_seconds)
            finally:
                if self.settings.analysis_cooldown_seconds > 0:
                    await asyncio.sleep(self.settings.analysis_cooldown_seconds)

    async def _analyze_segment(self, analyzer: LlamaAnalyzer, segment: dict[str, Any]) -> AnalysisResult:
        with tempfile.TemporaryDirectory(prefix="nas-video-frames-") as temp_dir:
            prefilter_started = time.monotonic()
            duration_seconds = int(segment["duration_seconds"])
            sample_count = (
                self.settings.person_filter_sample_count
                if self.settings.person_filter_enabled
                else self.settings.sample_frame_count
            )
            sampled_frames = await sample_frames_with_offsets(
                self.settings,
                Path(segment["path"]),
                Path(temp_dir) / "frames",
                duration_seconds=duration_seconds,
                sample_count=sample_count,
            )
            extracted_frame_count = len(sampled_frames)
            sampled_frames = await filter_out_blank_frames(
                self.settings, sampled_frames
            )
            if not sampled_frames:
                self._record_prefilter(
                    prefilter_started, "blank", extracted_frame_count, 0
                )
                self.database.add_event(
                    "blank-frame-skip",
                    "all sampled frames are near-black (camera masked/off?)",
                )
                self.database.mark_segment_processed(int(segment["id"]))
                raise PersonFilterSkip()

            decision = await filter_frames_by_person_detection(
                self.settings,
                sampled_frames,
            )
            sampled_frames = decision.frames
            if not sampled_frames:
                outcome = decision.skip_reason or "no-person"
                self._record_prefilter(
                    prefilter_started, outcome, extracted_frame_count, 0
                )
                if decision.skip_reason == "adult-only":
                    self.database.add_event(
                        "adult-only-filter-skip",
                        "all visible people were confidently classified as adults",
                    )
                else:
                    self.database.add_event(
                        "person-filter-skip",
                        "no person detected in any frame",
                    )
                self.database.mark_segment_processed(int(segment["id"]))
                raise PersonFilterSkip()

            if self.settings.analysis_image_mode == "frames":
                image_paths = [frame.path for frame in sampled_frames]
                frame_offsets_seconds = [frame.offset_seconds for frame in sampled_frames]
            else:
                sheet = await build_contact_sheet_from_frames(
                    self.settings,
                    sampled_frames,
                    Path(temp_dir),
                )
                image_paths = [sheet.path]
                frame_offsets_seconds = sheet.frame_offsets_seconds
            self._record_prefilter(
                prefilter_started,
                "ready",
                extracted_frame_count,
                len(sampled_frames),
            )
            try:
                result = await analyzer.analyze(
                    video_path=Path(segment["path"]),
                    image_paths=image_paths,
                    duration_seconds=duration_seconds,
                    frame_offsets_seconds=frame_offsets_seconds,
                )
            except Exception as exc:
                if (
                    not _is_timeout_error(exc)
                    or not self.settings.llama_timeout_fallback
                    or len(sampled_frames) <= 1
                ):
                    raise
                fallback_dir = Path(temp_dir) / "fallback"
                fallback_dir.mkdir(parents=True, exist_ok=True)
                fallback_sheet = await build_contact_sheet_from_frames(
                    self.settings,
                    sampled_frames,
                    fallback_dir,
                )
                self.database.add_event(
                    "analysis-timeout-fallback",
                    f"retrying {len(sampled_frames)} frames as one contact sheet",
                )
                result = await analyzer.analyze(
                    video_path=Path(segment["path"]),
                    image_paths=[fallback_sheet.path],
                    duration_seconds=duration_seconds,
                    frame_offsets_seconds=fallback_sheet.frame_offsets_seconds,
                    image_mode="contact_sheet",
                )
            return replace(
                result,
                local_child_confirmed=decision.child_confirmed,
                local_child_score=decision.max_child_score,
            )

    async def _analyze_daughter_segment(
        self, detector: DaughterDetector, segment: dict[str, Any]
    ) -> list[ClipCandidate]:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="nas-video-daughter-") as temp_dir:
            detector.reset_segment()
            duration = int(segment["duration_seconds"])
            frames = await sample_frames_at_fps(
                self.settings,
                Path(segment["path"]),
                Path(temp_dir) / "frames",
                duration_seconds=duration,
                fps=self.settings.daughter_scan_fps,
                width=self.settings.daughter_detector_input_size,
            )
            observations = []
            for frame in frames:
                observations.append(await asyncio.to_thread(detector.detect_path, frame))
            candidates = detector.candidates(observations)
        elapsed = time.monotonic() - started
        self.state["prefilter"] = {
            "status": "daughter-detector",
            "mode": detector.mode,
            "elapsed_seconds": round(elapsed, 3),
            "realtime_ratio": round(elapsed / max(1, duration), 3),
            "input_frame_count": len(frames),
            "candidate_count": len(candidates),
            "updated_at": utc_now_iso(),
        }
        return candidates

    def _record_prefilter(
        self,
        started_at: float,
        outcome: str,
        input_frame_count: int,
        output_frame_count: int,
    ) -> None:
        self.state["prefilter"] = {
            "status": outcome,
            "seconds": round(time.monotonic() - started_at, 3),
            "input_frames": input_frame_count,
            "output_frames": output_frame_count,
            "updated_at": utc_now_iso(),
        }

    async def _save_moment(
        self,
        segment: dict[str, Any],
        result: ClipCandidate,
        analyzer: LlamaAnalyzer | None,
        detector: DaughterDetector | None = None,
    ) -> int:
        source_started = _parse_iso(segment["started_at"])
        source_ended = _parse_iso(segment["ended_at"])
        display_offset = timedelta(seconds=self.settings.camera_time_offset_seconds)
        display_source_started = source_started + display_offset
        display_source_ended = source_ended + display_offset

        wanted_start = source_started + timedelta(
            seconds=result.start_offset_seconds - self.settings.context_before_seconds
        )
        wanted_end = source_started + timedelta(
            seconds=result.end_offset_seconds + self.settings.context_after_seconds
        )
        max_end = wanted_start + timedelta(seconds=self.settings.max_moment_seconds)
        wanted_end = min(wanted_end, max_end)

        segment_role = str(
            segment.get("stream_role") or self.settings.analysis_stream_role
        )
        high_segments = self.database.find_segments_between(
            stream_role="high",
            started_before=wanted_end.isoformat(timespec="seconds"),
            ended_after=wanted_start.isoformat(timespec="seconds"),
        )
        source_rows = sorted(
            (row for row in high_segments if Path(row["path"]).exists()),
            key=lambda row: _parse_iso(row["started_at"]),
        )
        if source_rows:
            source_paths = [Path(row["path"]) for row in source_rows]
            first_started = _parse_iso(source_rows[0]["started_at"])
            last_ended = _parse_iso(source_rows[-1]["ended_at"])
        else:
            source_paths = [Path(segment["path"])]
            first_started = source_started
            last_ended = source_ended

        clip_start = max(wanted_start, first_started)
        clip_end = min(wanted_end, last_ended)
        if clip_end <= clip_start:
            clip_start = first_started
            clip_end = min(last_ended, first_started + timedelta(seconds=self.settings.segment_seconds))

        start_offset_seconds = (clip_start - first_started).total_seconds()
        duration_seconds = max(1.0, (clip_end - clip_start).total_seconds())
        if duration_seconds > self.settings.max_moment_seconds:
            clip_end = clip_start + timedelta(seconds=self.settings.max_moment_seconds)
            duration_seconds = float(self.settings.max_moment_seconds)

        display_clip_start = clip_start + display_offset
        display_clip_end = clip_end + display_offset
        day_dir = self.settings.output_dir / display_clip_start.strftime("%Y-%m-%d")
        title_slug = _slugify(result.title)
        clip_path = _unique_path(day_dir / f"{display_clip_start.strftime('%H%M%S')}_{title_slug}.mp4")
        metadata_path = clip_path.with_suffix(".json")

        # Confirm the candidate on the analysis stream before doing expensive 4K
        # extraction. Three higher-resolution frames refine the positive decision.
        verified = True
        if result.analysis_backend == "vlm":
            assert analyzer is not None
            verified = await self._verify_candidate(
                analyzer,
                Path(segment["path"]),
                result.start_offset_seconds,
                result.end_offset_seconds,
            )
        if not verified:
            self.database.add_event(
                "moment-verify-failed",
                f"rejected false positive before clip extraction: {result.title}",
            )
            return -1

        day_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".nas-video-staged-", dir=day_dir
        ) as staging_dir:
            staged_clip_path = Path(staging_dir) / clip_path.name
            await extract_clip(
                self.settings,
                source_paths,
                staged_clip_path,
                start_offset_seconds=start_offset_seconds,
                duration_seconds=duration_seconds,
            )
            if result.analysis_backend == "daughter_detector":
                assert detector is not None
                verified_final = await self._verify_detector_saved_clip(
                    detector, staged_clip_path, duration_seconds
                )
            elif result.analysis_backend in {"rv1106_face", "rv1106_edge"}:
                identity = str(result.raw.get("identity") or "confirmed").lower()
                if identity == "probable" and self.settings.rv1106_probable_policy == "verify":
                    probable_verification = result.raw.get("probable_verification")
                    verified_final = bool(
                        isinstance(probable_verification, dict)
                        and probable_verification.get("accepted")
                    )
                else:
                    verified_final = (
                        self.settings.rv1106_probable_policy != "reject"
                        or identity != "probable"
                    )
            else:
                assert analyzer is not None
                verified_final = await self._verify_saved_clip(
                    analyzer, staged_clip_path, duration_seconds
                )
            if not verified_final:
                self.database.add_event(
                    "moment-verify-failed",
                    f"rejected final source clip before publishing: {result.title}",
                )
                return -1
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_clip_path, clip_path)

        metadata = {
            "schema_version": 3,
            "owner": "nas",
            "camera_name": segment["camera_name"],
            "analysis_backend": result.analysis_backend,
            "category": result.category,
            "title": result.title,
            "summary": result.summary,
            "tags": result.tags,
            "confidence": result.confidence,
            "selection_score": result.effective_selection_score,
            "keep_consistency_repaired": result.keep_consistency_repaired(
                self.settings.moment_keep_threshold
            ),
            "local_child_confirmed": result.local_child_confirmed,
            "local_child_score": result.local_child_score,
            "source_segment": segment["path"],
            "source_stream_role": segment_role,
            "source_started_at": display_source_started.isoformat(timespec="seconds"),
            "source_ended_at": display_source_ended.isoformat(timespec="seconds"),
            "wanted_start": (wanted_start + display_offset).isoformat(timespec="seconds"),
            "wanted_end": (wanted_end + display_offset).isoformat(timespec="seconds"),
            "clip_start": display_clip_start.isoformat(timespec="seconds"),
            "clip_end": display_clip_end.isoformat(timespec="seconds"),
            "clip_duration_seconds": duration_seconds,
            "source_paths": [str(path) for path in source_paths],
            "model_raw": result.raw,
        }
        if segment_role == "low":
            metadata["source_low_segment"] = segment["path"]
        trigger_key = str(result.raw.get("board_session_key") or "").strip() or None
        if trigger_key:
            metadata["trigger_key"] = trigger_key
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        moment_id = self.database.create_moment(
            camera_name=segment["camera_name"],
            title=result.title,
            summary=result.summary,
            tags=result.tags,
            confidence=result.confidence,
            source_low_segment_id=(
                int(segment["id"]) if segment_role == "low" else None
            ),
            source_started_at=display_source_started.isoformat(timespec="seconds"),
            source_ended_at=display_source_ended.isoformat(timespec="seconds"),
            clip_path=clip_path,
            metadata_path=metadata_path,
            analysis_backend=result.analysis_backend,
            category=result.category,
            selection_score=result.effective_selection_score,
            clip_started_at=display_clip_start.isoformat(timespec="seconds"),
            clip_ended_at=display_clip_end.isoformat(timespec="seconds"),
            trigger_key=trigger_key,
            source_segment_id=int(segment["id"]),
            source_stream_role=segment_role,
        )
        metadata["event_id"] = moment_id
        metadata["daily_summary_path"] = str(day_dir / "summary.md")
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        rebuild_day_archive(
            self.settings, self.database, display_clip_start.strftime("%Y-%m-%d")
        )
        return moment_id

    async def _verify_detector_saved_clip(
        self, detector: DaughterDetector, clip_path: Path, duration_seconds: float
    ) -> bool:
        offsets = [
            max(0.0, min(duration_seconds - 0.1, duration_seconds * fraction))
            for fraction in (0.1, 0.3, 0.5, 0.7, 0.9)
        ]
        try:
            with tempfile.TemporaryDirectory(prefix="nas-video-detector-verify-") as temp_dir:
                frames = await self._extract_verification_frames(
                    replace(self.settings, analysis_frame_width=self.settings.daughter_detector_input_size),
                    clip_path,
                    offsets,
                    temp_dir,
                    prefix="daughter",
                )
                return await asyncio.to_thread(
                    detector.verify_paths, [frame.path for frame in frames]
                )
        except Exception as exc:
            self.database.add_event("detector-verify-error", str(exc))
            return False

    async def _extract_verification_frames(
        self,
        settings: Settings,
        video_path: Path,
        offsets: list[float],
        temp_dir: str,
        *,
        prefix: str,
    ) -> list[SampledFrame]:
        """Extract one frame per offset; return only the frames that were written."""
        frames: list[SampledFrame] = []
        for index, offset in enumerate(offsets, start=1):
            frame_path = Path(temp_dir) / f"{prefix}-{index}.jpg"
            await _extract_frame(settings, video_path, frame_path, offset)
            if frame_path.exists():
                frames.append(SampledFrame(path=frame_path, offset_seconds=offset))
        return frames

    @staticmethod
    def _dual_evidence_keep(
        local_decision: PersonFilterDecision,
        verification: DaughterVerification,
    ) -> bool:
        """Keep an otherwise-unverified clip when local child evidence is strong
        and the VLM is only weakly negative."""
        return local_decision.child_confirmed and verification.confidence < 0.75

    def _log_verification_outcome(
        self,
        kept: bool,
        local_decision: PersonFilterDecision,
        verification: DaughterVerification,
    ) -> None:
        """Record the shared dual-evidence keep / verify-detail reject event."""
        if kept:
            self.database.add_event(
                "verify-local-child-keep",
                json.dumps(
                    {
                        "child_score": local_decision.max_child_score,
                        "confidence": verification.confidence,
                        "description": verification.description,
                    },
                    ensure_ascii=False,
                ),
            )
        else:
            self.database.add_event(
                "verify-detail",
                json.dumps(
                    {
                        "confidence": verification.confidence,
                        "description": verification.description,
                        "raw_text": verification.raw_text[:1000],
                    },
                    ensure_ascii=False,
                ),
            )

    async def _verify_saved_clip(
        self,
        analyzer: LlamaAnalyzer,
        clip_path: Path,
        duration_seconds: float,
    ) -> bool:
        """Verify the saved clip with frames from its beginning, middle, and end."""
        try:
            with tempfile.TemporaryDirectory(prefix="nas-video-verify-") as temp_dir:
                last_offset = max(0.0, duration_seconds - 0.5)
                offsets = sorted(
                    {
                        min(last_offset, max(0.0, duration_seconds * fraction))
                        for fraction in (0.2, 0.5, 0.8)
                    }
                )
                frames = await self._extract_verification_frames(
                    self.settings, clip_path, offsets, temp_dir, prefix="verify"
                )
                if not frames:
                    return False
                local_decision = await filter_frames_by_person_detection(
                    self.settings, frames
                )
                if not local_decision.frames:
                    self.database.add_event(
                        "verify-local-reject",
                        local_decision.skip_reason or "no local child evidence",
                    )
                    return False
                verification = await analyzer.verify_daughter_visible(
                    [frame.path for frame in frames]
                )
                verified = verification.visible and (
                    not verification.repaired or local_decision.child_confirmed
                )
                if verification.repaired and local_decision.child_confirmed:
                    self.database.add_event(
                        "verify-consistency-repair",
                        "corrected has_daughter=false with local child evidence",
                    )
                if not verified and self._dual_evidence_keep(local_decision, verification):
                    self._log_verification_outcome(True, local_decision, verification)
                    return True
                if not verified:
                    self._log_verification_outcome(False, local_decision, verification)
                return verified
        except Exception as exc:
            self.database.add_event("verify-error", str(exc))
            if _is_timeout_error(exc):
                raise
            # Final-source verification is fail-closed: an unreadable or
            # malformed clip must never become a published moment.
            return False

    async def _verify_candidate(
        self,
        analyzer: LlamaAnalyzer,
        video_path: Path,
        start_offset_seconds: float,
        end_offset_seconds: float,
    ) -> bool:
        center = (start_offset_seconds + end_offset_seconds) / 2
        offsets = sorted(
            {
                max(0.0, start_offset_seconds),
                max(0.0, center),
                max(0.0, end_offset_seconds),
            }
        )
        verification_settings = replace(
            self.settings,
            analysis_frame_width=self.settings.verification_frame_width,
        )
        try:
            with tempfile.TemporaryDirectory(prefix="nas-video-candidate-") as temp_dir:
                frames = await self._extract_verification_frames(
                    verification_settings, video_path, offsets, temp_dir, prefix="candidate"
                )
                if not frames:
                    return False
                local_decision = await filter_frames_by_person_detection(
                    self.settings, frames
                )
                if not local_decision.frames:
                    self.database.add_event(
                        "verify-local-reject",
                        local_decision.skip_reason or "no local child evidence",
                    )
                    return False
                verification = await analyzer.verify_daughter_visible(
                    [frame.path for frame in frames]
                )
                if verification.visible:
                    return True
                if self._dual_evidence_keep(local_decision, verification):
                    self._log_verification_outcome(True, local_decision, verification)
                    return True
                self._log_verification_outcome(False, local_decision, verification)
                return False
        except Exception as exc:
            self.database.add_event("verify-error", str(exc))
            if _is_timeout_error(exc):
                raise
            return False

    async def _cleanup_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                cutoff = (_now() - timedelta(hours=self.settings.retention_hours)).isoformat(
                    timespec="seconds"
                )
                deleted = await asyncio.to_thread(self._cleanup_once, cutoff)
                self.state["cleanup"] = {
                    "status": "ok",
                    "deleted": deleted,
                    "updated_at": utc_now_iso(),
                }
            except Exception as exc:
                self.state["cleanup"] = {
                    "status": "error",
                    "message": str(exc),
                    "updated_at": utc_now_iso(),
                }
                self._safe_add_event("cleanup-error", str(exc))
            await asyncio.sleep(300)

    async def _day_archive_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                finalized = self._finalize_ready_days()
                self.state["day_archive"] = {
                    "status": "ok",
                    "finalized": finalized,
                    "updated_at": utc_now_iso(),
                }
            except Exception as exc:
                self.state["day_archive"] = {
                    "status": "error",
                    "message": str(exc),
                    "updated_at": utc_now_iso(),
                }
                self._safe_add_event("day-archive-error", str(exc))
            await asyncio.sleep(60)

    def _finalize_ready_days(self) -> list[str]:
        now = _now()
        candidates = {
            path.name for path in self.settings.output_dir.glob("????-??-??")
            if path.is_dir()
        }
        candidates.add(now.strftime("%Y-%m-%d"))
        finalized: list[str] = []
        for day in sorted(candidates):
            try:
                start = datetime.fromisoformat(f"{day}T00:00:00").astimezone()
            except ValueError:
                continue
            end = start + timedelta(days=1)
            if start.date() > now.date():
                continue
            if start.date() == now.date() and not self._current_day_ready(now):
                continue
            pending, errors, total = self.database.segment_status_between(
                stream_role=self.settings.analysis_stream_role,
                start_iso=start.isoformat(timespec="seconds"),
                end_iso=end.isoformat(timespec="seconds"),
            )
            if total == 0 or pending > 0:
                continue
            day_dir = self.settings.output_dir / day
            ready_path = day_dir / "_READY.json"
            manifest_path = day_dir / "manifest.json"
            if ready_path.exists() and manifest_path.exists():
                try:
                    ready = json.loads(ready_path.read_text(encoding="utf-8"))
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if ready.get("manifest_revision") == manifest.get("revision"):
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            rebuild_day_archive(
                self.settings, self.database, day, ready=True, error_count=errors
            )
            finalized.append(day)
        return finalized

    def _current_day_ready(self, now: datetime) -> bool:
        if not self.settings.record_window_end:
            return False
        try:
            hour, minute = (int(value) for value in self.settings.record_window_end.split(":", 1))
        except (TypeError, ValueError):
            return False
        cutoff = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        cutoff += timedelta(
            seconds=self.settings.analysis_delay_seconds
            + self.settings.segment_stable_seconds
            + self.settings.day_ready_grace_seconds
        )
        return now >= cutoff

    def _cleanup_once(self, cutoff_iso: str) -> int:
        deleted = 0
        for segment in self.database.expired_segments(cutoff_iso):
            path = Path(segment["path"])
            try:
                path.unlink(missing_ok=True)
            finally:
                self.database.mark_segment_deleted(int(segment["id"]))
                deleted += 1
        return deleted


def health_snapshot(settings: Settings, database: Database, supervisor: Supervisor | None) -> dict[str, Any]:
    disk = shutil.disk_usage(settings.data_dir)
    recording_mode = (
        "dual"
        if settings.rtsp_low_url and settings.rtsp_high_url
        else "high_only"
        if settings.rtsp_high_url
        else "low_only"
        if settings.rtsp_low_url
        else "disabled"
    )
    return {
        "configured": {
            "low_rtsp": bool(settings.rtsp_low_url),
            "high_rtsp": bool(settings.rtsp_high_url),
            "recording_mode": recording_mode,
            "rtsp_credentials": bool(settings.rtsp_username or settings.rtsp_password),
            "camera_time_offset_seconds": settings.camera_time_offset_seconds,
            "segment_at_clocktime": settings.segment_at_clocktime,
            "stream_alignment_tolerance_seconds": settings.stream_alignment_tolerance_seconds,
            "stream_alignment_sample_count": settings.stream_alignment_sample_count,
            "analysis_backend": settings.analysis_backend,
            "analysis_enabled": settings.analysis_enabled,
            "daughter_detector_mode": settings.daughter_detector_mode,
            "daughter_detector_model_path": str(settings.daughter_detector_model_path or ""),
            "daughter_detector_input_size": settings.daughter_detector_input_size,
            "daughter_detector_threshold": settings.daughter_detector_threshold,
            "daughter_age_check_every": settings.daughter_age_check_every,
            "daughter_body_fallback_enabled": settings.daughter_body_fallback_enabled,
            "daughter_body_height_ratio": settings.daughter_body_height_ratio,
            "daughter_body_area_ratio": settings.daughter_body_area_ratio,
            "daughter_scan_fps": settings.daughter_scan_fps,
            "daughter_event_min_hits": settings.daughter_event_min_hits,
            "daughter_event_max_gap_seconds": settings.daughter_event_max_gap_seconds,
            "daughter_event_min_seconds": settings.daughter_event_min_seconds,
            "moment_category_targets": settings.moment_category_targets,
            "mqtt_enabled": settings.mqtt_enabled,
            "mqtt_host": settings.mqtt_host,
            "mqtt_port": settings.mqtt_port,
            "mqtt_daughter_topic": settings.mqtt_daughter_topic,
            "mqtt_status_topic": settings.mqtt_status_topic,
            "rv1106_accept_probable": settings.rv1106_accept_probable,
            "rv1106_probable_policy": settings.rv1106_probable_policy,
            "rv1106_save_wait_seconds": settings.rv1106_save_wait_seconds,
            "llama_base_url": settings.llama_base_url,
            "model": settings.llama_model,
            "llama_analysis_temperature": settings.llama_analysis_temperature,
            "llama_verification_temperature": settings.llama_verification_temperature,
            "analysis_image_mode": settings.analysis_image_mode,
            "analysis_frame_width": settings.analysis_frame_width,
            "sample_frame_count": settings.sample_frame_count,
            "verification_frame_width": settings.verification_frame_width,
            "llama_timeout_fallback": settings.llama_timeout_fallback,
            "llama_circuit_breaker_failures": settings.llama_circuit_breaker_failures,
            "llama_circuit_breaker_seconds": settings.llama_circuit_breaker_seconds,
            "max_moments_per_day": settings.max_moments_per_day,
            "max_moments_per_period": settings.max_moments_per_period,
            "moment_period_boundaries": settings.moment_period_boundaries,
            "person_filter_face_threshold": settings.person_filter_face_threshold,
            "person_filter_adult_threshold": settings.person_filter_adult_threshold,
            "person_filter_child_threshold": settings.person_filter_child_threshold,
            "analysis_stream_role": settings.analysis_stream_role,
            "analysis_window_start": settings.analysis_window_start,
            "analysis_window_end": settings.analysis_window_end,
            "output_dir": str(settings.output_dir),
            "buffer_dir": str(settings.buffer_dir),
        },
        "tools": {
            "ffmpeg": ffmpeg_available(settings),
            "ffprobe": ffprobe_available(settings),
        },
        "storage": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "segments": {
            "pending_low": database.count_pending_segments(stream_role="low"),
            "pending_high": database.count_pending_segments(stream_role="high"),
            "latest_low": database.latest_segment("low"),
            "latest_high": database.latest_segment("high"),
        },
        "stream_alignment": (
            supervisor.snapshot().get("stream_alignment", {"status": "unknown"})
            if supervisor
            else {"status": "not-started"}
        ),
        "workers": supervisor.snapshot() if supervisor else {"status": "not-started"},
        "events": database.recent_events(limit=10),
    }
