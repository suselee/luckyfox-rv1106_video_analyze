import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from nas_video_summarizer.config import load_settings
from nas_video_summarizer.database import Database
from nas_video_summarizer.daughter_detector import ProbableVerification
from nas_video_summarizer.ffmpeg_tools import (
    ContactSheet,
    PersonFilterDecision,
    SampledFrame,
)
from nas_video_summarizer.llm import AnalysisResult, DaughterVerification
from nas_video_summarizer.workers import (
    Supervisor,
    _append_daily_summary,
    _in_analysis_window,
    _in_time_window,
    _in_record_window,
    _moment_period,
    _normalized_board_roi,
    _segments_cover_window,
    _stream_alignment_snapshot,
)


def test_append_daily_summary(tmp_path):
    start = datetime.fromisoformat("2026-07-06T10:00:00+08:00")
    end = start + timedelta(seconds=90)
    clip_path = tmp_path / "100000_painting.mp4"
    result = AnalysisResult(
        keep=True,
        title="Painting together",
        summary="A quiet family moment at the table.",
        tags=["art", "family"],
        confidence=0.82,
        start_offset_seconds=0,
        end_offset_seconds=60,
        raw={},
    )

    summary_path = _append_daily_summary(
        day_dir=tmp_path,
        clip_path=clip_path,
        result=result,
        clip_start=start,
        clip_end=end,
    )

    text = summary_path.read_text(encoding="utf-8")
    assert "# Family Moments - 2026-07-06" in text
    assert "## 10:00:00 - Painting together" in text
    assert "[100000_painting.mp4](100000_painting.mp4)" in text
    assert "art, family" in text


def _alignment_segment(role: str, second: int) -> dict:
    return {
        "path": f"/{role}-{second}.mp4",
        "started_at": f"2026-07-15T08:{second // 60:02d}:{second % 60:02d}+08:00",
    }


def test_stream_alignment_is_stable_with_small_offset():
    low = [_alignment_segment("low", value) for value in (8, 6, 4, 2, 0)]
    high = [_alignment_segment("high", value) for value in (9, 7, 5, 3, 1)]

    result = _stream_alignment_snapshot(
        low,
        high,
        tolerance_seconds=2,
        required_samples=5,
        segment_seconds=120,
    )

    assert result["status"] == "stable"
    assert result["offset_seconds"] == 1.0
    assert result["paired_segments"] == 5


def test_stream_alignment_reports_drift():
    low = [_alignment_segment("low", value) for value in (480, 360, 240, 120, 0)]
    high = [_alignment_segment("high", value) for value in (484, 364, 244, 124, 4)]

    result = _stream_alignment_snapshot(
        low,
        high,
        tolerance_seconds=2,
        required_samples=5,
        segment_seconds=120,
    )

    assert result["status"] == "drifted"
    assert result["offset_seconds"] == 4.0


def test_stream_alignment_requires_enough_pairs():
    result = _stream_alignment_snapshot(
        [_alignment_segment("low", 0)],
        [_alignment_segment("high", 1)],
        tolerance_seconds=2,
        required_samples=5,
        segment_seconds=120,
    )

    assert result["status"] == "insufficient"
    assert result["paired_segments"] == 1


def test_segments_cover_window_across_adjacent_high_segments():
    segments = [
        {
            "started_at": "2026-07-15T08:00:00+08:00",
            "ended_at": "2026-07-15T08:02:00+08:00",
        },
        {
            "started_at": "2026-07-15T08:02:01+08:00",
            "ended_at": "2026-07-15T08:04:00+08:00",
        },
    ]
    assert _segments_cover_window(
        segments,
        datetime.fromisoformat("2026-07-15T08:01:55+08:00"),
        datetime.fromisoformat("2026-07-15T08:02:10+08:00"),
        gap_tolerance_seconds=2,
    )


def test_board_roi_maps_best_box_and_legacy_dimensions():
    roi = _normalized_board_roi(
        {"best_box": [202, 190, 62, 88]},
        width_scale=3.0,
        height_scale=2.0,
    )

    assert roi is not None
    assert roi == pytest.approx(
        (0.21875, 0.405556, 0.290625, 0.488889), rel=1e-5
    )


def test_board_roi_clamps_at_frame_edge():
    roi = _normalized_board_roi(
        {
            "box": [0, 0, 100, 100],
            "frame_width": 640,
            "frame_height": 360,
        },
        width_scale=3.0,
        height_scale=2.0,
    )

    assert roi is not None
    assert roi[0] == 0.0
    assert roi[1] == 0.0
    assert roi[2] > 0
    assert roi[3] > 0


def _dt(hour: int, minute: int) -> datetime:
    return datetime(2026, 7, 8, hour, minute, tzinfo=datetime.now().astimezone().tzinfo)


def test_in_analysis_window_empty_is_always_true():
    assert _in_analysis_window(_dt(3, 0), "", "") is True
    assert _in_analysis_window(_dt(15, 0), "", "") is True
    assert _in_analysis_window(_dt(23, 59), "", "") is True


def test_in_analysis_window_crosses_midnight():
    # Window: 21:15 -> 06:00 (next day)
    assert _in_analysis_window(_dt(21, 15), "21:15", "06:00") is True
    assert _in_analysis_window(_dt(23, 30), "21:15", "06:00") is True
    assert _in_analysis_window(_dt(3, 0), "21:15", "06:00") is True
    assert _in_analysis_window(_dt(5, 59), "21:15", "06:00") is True
    # Just before window opens
    assert _in_analysis_window(_dt(21, 14), "21:15", "06:00") is False
    # At window close
    assert _in_analysis_window(_dt(6, 0), "21:15", "06:00") is False
    # Middle of day
    assert _in_analysis_window(_dt(12, 0), "21:15", "06:00") is False


def test_in_analysis_window_same_day():
    # Window: 09:00 -> 17:00
    assert _in_analysis_window(_dt(9, 0), "09:00", "17:00") is True
    assert _in_analysis_window(_dt(12, 30), "09:00", "17:00") is True
    assert _in_analysis_window(_dt(16, 59), "09:00", "17:00") is True
    assert _in_analysis_window(_dt(8, 59), "09:00", "17:00") is False
    assert _in_analysis_window(_dt(17, 0), "09:00", "17:00") is False
    assert _in_analysis_window(_dt(22, 0), "09:00", "17:00") is False


def test_time_window_helpers_and_record_alias():
    # _in_time_window is the generic backend for both analysis and record.
    assert _in_time_window(_dt(3, 0), "21:15", "06:00") is True
    assert _in_time_window(_dt(12, 0), "21:15", "06:00") is False
    assert _in_time_window(_dt(10, 0), "", "") is True
    # record window same-day (07:00 .. 21:00, end exclusive)
    assert _in_record_window(_dt(20, 0), "07:00", "21:00") is True
    assert _in_record_window(_dt(21, 30), "07:00", "21:00") is False
    assert _in_record_window(_dt(7, 0), "07:00", "21:00") is True
    assert _in_record_window(_dt(10, 0), "", "") is True


def test_running_recorder_stops_when_record_window_closes(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        record_window_start="07:00",
        record_window_end="21:00",
    )
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)

    class FakeProcess:
        pid = 123
        returncode = None

        async def wait(self):
            return self.returncode

    process = FakeProcess()
    times = iter((_dt(20, 59), _dt(21, 0)))

    async def fake_spawn(*args, **kwargs):
        return process

    async def fake_stop(proc):
        proc.returncode = 0
        supervisor.stop_event.set()

    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr("nas_video_summarizer.workers.ffmpeg_available", lambda settings: True)
    monkeypatch.setattr(
        "nas_video_summarizer.workers.build_recorder_command",
        lambda settings, role, url: ["ffmpeg"],
    )
    monkeypatch.setattr("nas_video_summarizer.workers._now", lambda: next(times))
    monkeypatch.setattr("nas_video_summarizer.workers._stop_process", fake_stop)
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_spawn)
    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    asyncio.run(supervisor._recorder_loop("low", "rtsp://camera/low"))

    assert supervisor.state["recorders"]["low"]["status"] == "waiting_for_record_window"
    assert database.recent_events(limit=1)[0]["event_type"] == "recorder-window"


class _FakeCapDB:
    def __init__(self, count, weakest, period_count=None):
        self._count = count
        self._period_count = count if period_count is None else period_count
        self._weakest = weakest
        self.evicted = None

    def count_moments_on_day(self, day):
        return self._count

    def weakest_moment_on_day(self, day):
        return self._weakest

    def count_moments_between(self, start_iso, end_iso):
        return self._period_count

    def weakest_moment_between(self, start_iso, end_iso):
        return self._weakest

    def delete_moment_by_clip(self, clip):
        self.evicted = clip

    def add_event(self, *args, **kwargs):
        pass


def _supervisor_with(db, max_per_day=20, max_per_period=0):
    settings = replace(
        load_settings("/nonexistent.env"),
        max_moments_per_day=max_per_day,
        max_moments_per_period=max_per_period,
        moment_keep_threshold=0.5,
    )
    return Supervisor(settings, db)


def _segment(day="2026-07-10T10:00:00+08:00"):
    return {"started_at": day}


def _result(confidence):
    return AnalysisResult(
        keep=True,
        title="x",
        summary="",
        tags=[],
        confidence=confidence,
        start_offset_seconds=0,
        end_offset_seconds=1,
        raw={},
    )


def test_daily_cap_under_limit_saves():
    db = _FakeCapDB(
        count=5,
        weakest={"confidence": 0.3, "clip_path": "/x", "metadata_path": "/y", "title": "t", "id": 1},
    )
    sup = _supervisor_with(db, max_per_day=20)
    assert sup._daily_cap_decision(_segment(), _result(0.6)).outcome == "ok"


def test_daily_cap_at_limit_better_evicts():
    db = _FakeCapDB(
        count=20,
        weakest={"confidence": 0.3, "clip_path": "/x.mp4", "metadata_path": "/x.json", "title": "weak", "id": 1},
    )
    sup = _supervisor_with(db, max_per_day=20)
    decision = sup._daily_cap_decision(_segment(), _result(0.8))
    assert decision.outcome == "evict" and decision.weakest["title"] == "weak"


def test_daily_cap_at_limit_worse_skips():
    db = _FakeCapDB(
        count=20,
        weakest={"confidence": 0.7, "clip_path": "/x.mp4", "metadata_path": "/x.json", "title": "strong", "id": 1},
    )
    sup = _supervisor_with(db, max_per_day=20)
    assert sup._daily_cap_decision(_segment(), _result(0.5)).outcome == "blocked"


def test_moment_period_boundaries():
    boundaries = "07:00,12:00,17:00,21:00"

    assert _moment_period(_dt(11, 59), boundaries)[0] == "morning"
    assert _moment_period(_dt(12, 0), boundaries)[0] == "afternoon"
    assert _moment_period(_dt(16, 59), boundaries)[0] == "afternoon"
    assert _moment_period(_dt(17, 0), boundaries)[0] == "evening"
    assert _moment_period(_dt(21, 0), boundaries) is None
    assert _moment_period(_dt(10, 0), "invalid") is None


def test_period_cap_under_limit_saves():
    db = _FakeCapDB(count=20, period_count=7, weakest=None)
    sup = _supervisor_with(db, max_per_day=24, max_per_period=8)

    decision = sup._period_cap_decision(
        {"started_at": "2026-07-10T10:00:00+08:00"}, _result(0.8)
    )
    assert decision.outcome == "ok" and decision.scope == "morning"


def test_period_cap_replaces_weaker_moment_in_same_period():
    weakest = {
        "confidence": 0.7,
        "clip_path": "/x.mp4",
        "metadata_path": "/x.json",
        "title": "weak",
        "id": 1,
    }
    db = _FakeCapDB(count=20, period_count=8, weakest=weakest)
    sup = _supervisor_with(db, max_per_day=24, max_per_period=8)

    decision = sup._period_cap_decision(
        {"started_at": "2026-07-10T18:00:00+08:00"}, _result(0.85)
    )

    assert decision.outcome == "evict"
    assert decision.scope == "evening"
    assert decision.weakest == weakest


def test_period_cap_skips_equal_or_weaker_moment():
    weakest = {
        "confidence": 0.85,
        "clip_path": "/x.mp4",
        "metadata_path": "/x.json",
        "title": "strong",
        "id": 1,
    }
    db = _FakeCapDB(count=20, period_count=8, weakest=weakest)
    sup = _supervisor_with(db, max_per_day=24, max_per_period=8)

    decision = sup._period_cap_decision(
        {"started_at": "2026-07-10T13:00:00+08:00"}, _result(0.85)
    )
    assert decision.outcome == "blocked" and decision.scope == "afternoon"


def test_evict_moment_removes_files(tmp_path):
    clip = tmp_path / "c.mp4"
    meta = tmp_path / "c.json"
    clip.write_bytes(b"x")
    meta.write_bytes(b"y")
    db = _FakeCapDB(0, None)
    sup = _supervisor_with(db)
    sup._evict_moment(
        {"clip_path": str(clip), "metadata_path": str(meta), "title": "t", "confidence": 0.3}
    )
    assert not clip.exists() and not meta.exists()


def test_detector_cooldown_uses_clip_source_time(tmp_path):
    settings = replace(
        load_settings("/nonexistent.env"),
        output_dir=tmp_path / "out",
        moment_cooldown_seconds=480,
    )
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    clip = tmp_path / "out" / "2026-07-10" / "100000_x.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"x")
    metadata = clip.with_suffix(".json")
    metadata.write_text("{}")
    database.create_moment(
        camera_name="home-camera",
        title="x",
        summary="",
        tags=[],
        confidence=0.9,
        source_low_segment_id=None,
        source_started_at="2026-07-10T10:00:00+08:00",
        source_ended_at="2026-07-10T10:02:00+08:00",
        clip_path=clip,
        metadata_path=metadata,
        clip_started_at="2026-07-10T10:00:00+08:00",
        clip_ended_at="2026-07-10T10:00:30+08:00",
    )
    supervisor = Supervisor(settings, database)
    result = replace(_result(0.9), analysis_backend="daughter_detector")

    assert supervisor._moment_cooldown_active(
        {"started_at": "2026-07-10T10:05:00+08:00"}, result
    )
    assert not supervisor._moment_cooldown_active(
        {"started_at": "2026-07-10T10:09:00+08:00"}, result
    )


def test_apply_moment_caps_period_eviction_lowers_daily_count(tmp_path):
    # The period-weakest is also the day-weakest, and both caps are at limit.
    # Evicting it for the period must lower the day count so the daily check
    # sees the post-eviction state and does NOT evict a second moment.
    weakest = {
        "confidence": 0.3,
        "clip_path": str(tmp_path / "w.mp4"),
        "metadata_path": str(tmp_path / "w.json"),
        "title": "weak",
        "id": 1,
    }

    class _SharedWeakestDB:
        def __init__(self):
            self.day_count = 20
            self.period_count = 8
            self.evictions = 0

        def count_moments_on_day(self, day):
            return self.day_count

        def weakest_moment_on_day(self, day):
            return weakest

        def count_moments_between(self, start_iso, end_iso):
            return self.period_count

        def weakest_moment_between(self, start_iso, end_iso):
            return weakest

        def delete_moment_by_clip(self, clip):
            # The evicted moment leaves both the day and its period.
            self.day_count -= 1
            self.period_count -= 1
            self.evictions += 1

        def add_event(self, *args, **kwargs):
            pass

    db = _SharedWeakestDB()
    sup = _supervisor_with(db, max_per_day=20, max_per_period=8)

    plan = sup._apply_moment_caps(
        {"started_at": "2026-07-10T18:00:00+08:00"}, _result(0.85)
    )

    assert plan.skip is None  # proceed to save
    assert len(plan.evictions) == 1  # schedule once, do not delete yet
    assert db.evictions == 0


def test_analyze_segment_passes_actual_frame_offsets(tmp_path, monkeypatch):
    settings = replace(load_settings("/nonexistent.env"), analysis_image_mode="frames")
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"jpeg")
    captured: dict[str, list[float]] = {}

    async def fake_sample_frames_with_offsets(settings, video_path, output_dir, *, duration_seconds, sample_count=None):
        return [SampledFrame(path=frame_path, offset_seconds=73.5)]

    class FakeAnalyzer:
        async def analyze(self, **kwargs):
            captured["offsets"] = kwargs["frame_offsets_seconds"]
            return AnalysisResult(
                keep=False,
                title="quiet",
                summary="none",
                tags=[],
                confidence=0.1,
                start_offset_seconds=0,
                end_offset_seconds=1,
                raw={},
            )

    monkeypatch.setattr(
        "nas_video_summarizer.workers.sample_frames_with_offsets",
        fake_sample_frames_with_offsets,
    )

    asyncio.run(
        supervisor._analyze_segment(
            FakeAnalyzer(),
            {
                "path": str(tmp_path / "segment.mp4"),
                "duration_seconds": 120,
            },
        )
    )

    assert captured["offsets"] == [73.5]


def test_contact_sheet_uses_prefiltered_frames(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        analysis_image_mode="contact_sheet",
        person_filter_enabled=True,
    )
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)
    frames = []
    for index, offset in enumerate((20.0, 80.0)):
        path = tmp_path / f"frame-{index}.jpg"
        path.write_bytes(b"jpeg")
        frames.append(SampledFrame(path=path, offset_seconds=offset))
    captured = {}

    async def fake_sample(*args, **kwargs):
        return frames

    async def fake_blank(settings, sampled_frames):
        return sampled_frames

    async def fake_filter(settings, sampled_frames):
        return PersonFilterDecision([sampled_frames[1]])

    async def fake_sheet(settings, sampled_frames, output_dir):
        captured["sheet_frames"] = sampled_frames
        path = output_dir / "contact_sheet.jpg"
        path.write_bytes(b"sheet")
        return ContactSheet(path=path, frame_offsets_seconds=[80.0])

    class FakeAnalyzer:
        async def analyze(self, **kwargs):
            captured["image_paths"] = kwargs["image_paths"]
            captured["offsets"] = kwargs["frame_offsets_seconds"]
            return _result(0.1)

    monkeypatch.setattr(
        "nas_video_summarizer.workers.sample_frames_with_offsets", fake_sample
    )
    monkeypatch.setattr(
        "nas_video_summarizer.workers.filter_out_blank_frames", fake_blank
    )
    monkeypatch.setattr(
        "nas_video_summarizer.workers.filter_frames_by_person_detection",
        fake_filter,
    )
    monkeypatch.setattr(
        "nas_video_summarizer.workers.build_contact_sheet_from_frames",
        fake_sheet,
    )

    asyncio.run(
        supervisor._analyze_segment(
            FakeAnalyzer(),
            {"path": str(tmp_path / "segment.mp4"), "duration_seconds": 120},
        )
    )

    assert captured["sheet_frames"] == [frames[1]]
    assert captured["offsets"] == [80.0]
    assert captured["image_paths"][0].name == "contact_sheet.jpg"
    assert supervisor.state["prefilter"]["status"] == "ready"
    assert supervisor.state["prefilter"]["input_frames"] == 2
    assert supervisor.state["prefilter"]["output_frames"] == 1


def test_adult_only_filter_records_distinct_event(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        analysis_image_mode="frames",
        person_filter_enabled=True,
    )
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"jpeg")

    async def fake_sample(*args, **kwargs):
        return [SampledFrame(path=frame_path, offset_seconds=30.0)]

    async def fake_blank(settings, sampled_frames):
        return sampled_frames

    async def fake_filter(settings, sampled_frames):
        return PersonFilterDecision([], "adult-only")

    monkeypatch.setattr(
        "nas_video_summarizer.workers.sample_frames_with_offsets", fake_sample
    )
    monkeypatch.setattr(
        "nas_video_summarizer.workers.filter_out_blank_frames", fake_blank
    )
    monkeypatch.setattr(
        "nas_video_summarizer.workers.filter_frames_by_person_detection",
        fake_filter,
    )

    try:
        asyncio.run(
            supervisor._analyze_segment(
                object(),
                {
                    "id": 1,
                    "path": str(tmp_path / "segment.mp4"),
                    "duration_seconds": 120,
                },
            )
        )
    except Exception as exc:
        assert exc.__class__.__name__ == "PersonFilterSkip"
    else:
        raise AssertionError("adult-only segment should skip analysis")

    events = database.recent_events(limit=1)
    assert events[0]["event_type"] == "adult-only-filter-skip"
    assert supervisor.state["prefilter"]["status"] == "adult-only"


def test_saved_clip_verification_uses_three_frames(tmp_path, monkeypatch):
    settings = load_settings("/nonexistent.env")
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)
    offsets = []

    async def fake_extract(settings, clip_path, frame_path, offset):
        offsets.append(offset)
        frame_path.write_bytes(b"jpeg")

    class FakeAnalyzer:
        async def verify_daughter_visible(self, frame_paths):
            assert len(frame_paths) == 3
            return DaughterVerification(
                visible=True,
                confidence=0.9,
                description="A young girl is visible.",
                repaired=False,
                raw_text="{}",
            )

    monkeypatch.setattr("nas_video_summarizer.workers._extract_frame", fake_extract)

    verified = asyncio.run(
        supervisor._verify_saved_clip(
            FakeAnalyzer(), tmp_path / "clip.mp4", duration_seconds=20
        )
    )

    assert verified is True
    assert offsets == [4.0, 10.0, 16.0]


def test_low_confidence_negative_without_local_child_rejects_clip(
    tmp_path, monkeypatch
):
    settings = load_settings("/nonexistent.env")
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)

    async def fake_extract(settings, clip_path, frame_path, offset):
        frame_path.write_bytes(b"jpeg")

    class FakeAnalyzer:
        async def verify_daughter_visible(self, frame_paths):
            return DaughterVerification(
                visible=False,
                confidence=0.1,
                description="A blurry person, possibly interacting with a child.",
                repaired=False,
                raw_text="{}",
            )

    monkeypatch.setattr("nas_video_summarizer.workers._extract_frame", fake_extract)

    verified = asyncio.run(
        supervisor._verify_saved_clip(
            FakeAnalyzer(), tmp_path / "clip.mp4", duration_seconds=20
        )
    )

    assert verified is False
    assert database.recent_events(limit=1)[0]["event_type"] == "verify-detail"


def test_local_child_evidence_can_keep_uncertain_verification(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"), person_filter_enabled=True
    )
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)

    async def fake_extract(settings, clip_path, frame_path, offset):
        frame_path.write_bytes(b"jpeg")

    async def fake_local_filter(settings, frames):
        return PersonFilterDecision(
            frames,
            max_child_score=0.82,
            child_confirmed=True,
        )

    class FakeAnalyzer:
        async def verify_daughter_visible(self, frame_paths):
            return DaughterVerification(
                visible=False,
                confidence=0.1,
                description="A blurry person, possibly interacting with a child.",
                repaired=False,
                raw_text="{}",
            )

    monkeypatch.setattr("nas_video_summarizer.workers._extract_frame", fake_extract)
    monkeypatch.setattr(
        "nas_video_summarizer.workers.filter_frames_by_person_detection",
        fake_local_filter,
    )

    verified = asyncio.run(
        supervisor._verify_saved_clip(
            FakeAnalyzer(), tmp_path / "clip.mp4", duration_seconds=20
        )
    )

    assert verified is True
    assert database.recent_events(limit=1)[0]["event_type"] == "verify-local-child-keep"


def test_candidate_verification_uses_high_resolution_frames(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"), verification_frame_width=512
    )
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)
    captured = []

    async def fake_extract(frame_settings, video_path, frame_path, offset):
        captured.append((frame_settings.analysis_frame_width, offset))
        frame_path.write_bytes(b"jpeg")

    class FakeAnalyzer:
        async def verify_daughter_visible(self, frame_paths):
            return DaughterVerification(True, 0.9, "daughter visible", False, "{}")

    monkeypatch.setattr("nas_video_summarizer.workers._extract_frame", fake_extract)

    verified = asyncio.run(
        supervisor._verify_candidate(
            FakeAnalyzer(), tmp_path / "segment.mp4", 20, 40
        )
    )

    assert verified is True
    assert captured == [(512, 20), (512, 30.0), (512, 40)]


def test_final_source_verification_fails_closed(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        analysis_stream_role="high",
        output_dir=tmp_path / "output",
    )
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)
    segment_path = tmp_path / "segment.mp4"
    segment_path.write_bytes(b"source")
    segment = {
        "id": 1,
        "camera_name": "home-camera",
        "path": str(segment_path),
        "started_at": "2026-07-15T08:34:15+08:00",
        "ended_at": "2026-07-15T08:36:15+08:00",
    }
    database.upsert_segment(
        camera_name="home-camera",
        stream_role="high",
        path=segment_path,
        started_at=segment["started_at"],
        ended_at=segment["ended_at"],
        duration_seconds=120,
        size_bytes=6,
    )

    async def fake_candidate(*args, **kwargs):
        return True

    async def fake_extract(settings, paths, output_path, **kwargs):
        output_path.write_bytes(b"staged")

    async def fake_verify(*args, **kwargs):
        return False

    monkeypatch.setattr(supervisor, "_verify_candidate", fake_candidate)
    monkeypatch.setattr("nas_video_summarizer.workers.extract_clip", fake_extract)
    monkeypatch.setattr(supervisor, "_verify_saved_clip", fake_verify)

    result = asyncio.run(
        supervisor._save_moment(
            segment,
            _result(0.95),
            object(),
        )
    )

    assert result == -1
    assert not list((tmp_path / "output").rglob("*.mp4"))


def test_rv1106_moment_is_not_rejected_by_nas_detector(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        analysis_stream_role="high",
        output_dir=tmp_path / "output",
    )
    database = Database(tmp_path / "edge.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)
    segment_path = tmp_path / "segment.mp4"
    segment_path.write_bytes(b"source")
    segment_id = database.upsert_segment(
        camera_name="home-camera",
        stream_role="high",
        path=segment_path,
        started_at="2026-07-15T08:34:15+08:00",
        ended_at="2026-07-15T08:36:15+08:00",
        duration_seconds=120,
        size_bytes=6,
    )
    segment = {
        "id": segment_id,
        "camera_name": "home-camera",
        "path": str(segment_path),
        "started_at": "2026-07-15T08:34:15+08:00",
        "ended_at": "2026-07-15T08:36:15+08:00",
    }

    async def fake_extract(settings, paths, output_path, **kwargs):
        output_path.write_bytes(b"edge-source")

    monkeypatch.setattr("nas_video_summarizer.workers.extract_clip", fake_extract)
    result = replace(
        _result(0.8),
        analysis_backend="rv1106_face",
        local_child_confirmed=True,
    )

    moment_id = asyncio.run(
        supervisor._save_moment(segment, result, analyzer=None, detector=None)
    )

    assert moment_id > 0
    assert next((tmp_path / "output").rglob("*.mp4")).read_bytes() == b"edge-source"


def test_published_clip_uses_camera_offset_and_final_source(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        analysis_stream_role="high",
        output_dir=tmp_path / "output",
        camera_time_offset_seconds=-23,
    )
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)
    segment_path = tmp_path / "segment.mp4"
    segment_path.write_bytes(b"source")
    segment_id = database.upsert_segment(
        camera_name="home-camera",
        stream_role="high",
        path=segment_path,
        started_at="2026-07-15T08:34:14+08:00",
        ended_at="2026-07-15T08:36:14+08:00",
        duration_seconds=120,
        size_bytes=6,
    )
    segment = {
        "id": segment_id,
        "camera_name": "home-camera",
        "path": str(segment_path),
        "started_at": "2026-07-15T08:34:14+08:00",
        "ended_at": "2026-07-15T08:36:14+08:00",
    }

    async def fake_candidate(*args, **kwargs):
        return True

    async def fake_extract(settings, paths, output_path, **kwargs):
        output_path.write_bytes(b"verified-source")

    async def fake_verify(*args, **kwargs):
        return True

    monkeypatch.setattr(supervisor, "_verify_candidate", fake_candidate)
    monkeypatch.setattr("nas_video_summarizer.workers.extract_clip", fake_extract)
    monkeypatch.setattr(supervisor, "_verify_saved_clip", fake_verify)

    moment_id = asyncio.run(
        supervisor._save_moment(segment, _result(0.95), object())
    )

    assert moment_id > 0
    clip = next((tmp_path / "output" / "2026-07-15").glob("*.mp4"))
    assert clip.name.startswith("083351_")
    metadata = json.loads(clip.with_suffix(".json").read_text())
    assert metadata["clip_start"] == "2026-07-15T08:33:51+08:00"
    assert clip.read_bytes() == b"verified-source"


def test_mqtt_hit_is_persisted_and_updates_state(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        mqtt_enabled=True,
        mqtt_daughter_topic="homecam/daughter/hit",
    )
    database = Database(tmp_path / "mqtt.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)

    async def immediate_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)
    payload = json.dumps(
        {
            "ts": datetime.fromisoformat("2026-07-19T10:00:00+08:00").timestamp(),
            "score": 0.73,
            "camera_id": "home-camera",
            "box": [1, 2, 3, 4],
            "seq": 9,
            "identity": "confirmed",
        }
    ).encode()

    asyncio.run(supervisor._handle_mqtt_message("homecam/daughter/hit", payload))

    sessions = database.pending_board_sessions()
    assert len(sessions) == 1
    session = sessions[0]
    assert session["score"] == 0.73
    assert session["identity"] == "confirmed"
    assert session["status"] == "ready"
    assert supervisor.state["mqtt"]["status"] == "connected"
    assert supervisor.state["mqtt"]["last_identity"] == "confirmed"


def test_mqtt_status_heartbeat_updates_rv1106_health(tmp_path):
    settings = replace(
        load_settings("/nonexistent.env"),
        mqtt_enabled=True,
        mqtt_status_topic="homecam/daughter/status",
    )
    database = Database(tmp_path / "mqtt-status.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)
    payload = json.dumps(
        {
            "pipeline": "rockiva_fusion_v1",
            "cpu_percent": 48.2,
            "temperature_c": 61.0,
            "detector_p95_ms": 29.6,
        }
    ).encode()

    asyncio.run(
        supervisor._handle_mqtt_message("homecam/daughter/status", payload)
    )

    assert supervisor.state["rv1106"]["status"] == "online"
    assert supervisor.state["rv1106"]["pipeline"] == "rockiva_fusion_v1"


def test_mqtt_reject_policy_ignores_probable_sessions(tmp_path):
    settings = replace(
        load_settings("/nonexistent.env"),
        mqtt_enabled=True,
        mqtt_daughter_topic="homecam/daughter/hit",
        rv1106_probable_policy="reject",
        rv1106_accept_probable=False,
    )
    database = Database(tmp_path / "mqtt-reject.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)
    payload = json.dumps(
        {
            "ts": datetime.fromisoformat("2026-07-19T10:00:00+08:00").timestamp(),
            "score": 0.7,
            "camera_id": "home-camera",
            "event": "start",
            "session_id": "track-1",
            "identity": "probable",
        }
    ).encode()

    asyncio.run(supervisor._handle_mqtt_message("homecam/daughter/hit", payload))

    assert database.count_pending_board_sessions() == 0
    assert supervisor.state["mqtt"]["last_ignored_identity"] == "probable"


def test_mqtt_fusion_session_persists_start_update_end(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        mqtt_enabled=True,
        mqtt_daughter_topic="homecam/daughter/hit",
    )
    database = Database(tmp_path / "mqtt-session.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)

    async def immediate_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)
    base = datetime.fromisoformat("2026-07-19T10:00:00+08:00").timestamp()
    for index, event in enumerate(("start", "update", "end")):
        payload = json.dumps(
            {
                "ts": base + index * 30,
                "session_start_ts": base,
                "best_ts": base + 30,
                "score": 0.55 if index == 0 else 0.72,
                "camera_id": "home-camera",
                "box": [1, 2, 3, 4],
                "seq": index + 1,
                "session_id": "track-7-1",
                "event": event,
                "identity": "probable" if index == 0 else "confirmed",
            }
        ).encode()
        asyncio.run(
            supervisor._handle_mqtt_message("homecam/daughter/hit", payload)
        )

    ready = database.pending_board_sessions()
    assert len(ready) == 1
    finished = ready[0]
    assert finished["session_id"] == "track-7-1"
    assert finished["identity"] == "confirmed"
    assert finished["score"] == 0.72
    assert finished["best_ts"] == base + 30


def test_duplicate_mqtt_end_is_ignored_and_survives_restart(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        mqtt_enabled=True,
        mqtt_daughter_topic="homecam/daughter/hit",
    )
    path = tmp_path / "mqtt-dedupe.sqlite3"
    database = Database(path)
    database.migrate()
    supervisor = Supervisor(settings, database)

    async def immediate_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)
    timestamp = datetime.fromisoformat("2026-07-19T10:00:00+08:00").timestamp()
    payload = json.dumps(
        {
            "ts": timestamp,
            "best_ts": timestamp,
            "score": 0.8,
            "camera_id": "home-camera",
            "seq": 7,
            "session_id": "track-1-1",
            "event": "end",
            "identity": "confirmed",
        }
    ).encode()

    asyncio.run(supervisor._handle_mqtt_message("homecam/daughter/hit", payload))
    asyncio.run(supervisor._handle_mqtt_message("homecam/daughter/hit", payload))

    assert database.count_pending_board_sessions() == 1
    assert supervisor.state["mqtt"]["duplicate"] is True
    reopened = Database(path)
    assert len(reopened.pending_board_sessions()) == 1


def test_board_restart_can_reuse_session_counter_without_key_collision(tmp_path):
    settings = load_settings("/nonexistent.env")
    database = Database(tmp_path / "session-reuse.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)
    first = {
        "session_id": "1-1",
        "session_start_ts": 100.0,
        "seq": 1,
        "event": "start",
    }
    after_restart = {**first, "session_start_ts": 200.0}

    first_session, first_event = supervisor._board_event_keys(
        "home-camera", first, 100.0, "start"
    )
    second_session, second_event = supervisor._board_event_keys(
        "home-camera", after_restart, 200.0, "start"
    )

    assert first_session != second_session
    assert first_event != second_event


def test_board_session_dropped_when_segments_never_indexed(tmp_path, monkeypatch):
    settings = replace(load_settings("/nonexistent.env"))
    database = Database(tmp_path / "pending-board.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)

    async def immediate_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    async def instant_sleep(seconds):
        return None

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setattr(asyncio, "sleep", instant_sleep)
    session = {
        "session_id": "s-1",
        "key": "s-1",
        "camera_id": "home-camera",
        "session_start": datetime.fromisoformat("2026-07-19T10:00:00+08:00").timestamp(),
        "identity": "confirmed",
        "score": 0.8,
        "best_ts": datetime.fromisoformat("2026-07-19T10:00:00+08:00").timestamp(),
        "last_event_at": datetime.fromisoformat("2026-07-19T10:00:01+08:00").timestamp(),
        "payload": {},
    }

    asyncio.run(supervisor._save_board_session(session))

    assert database.list_moments() == []
    events = database.recent_events(limit=5)
    assert any("board-session-dropped" in event["event_type"] for event in events)


def _board_session(best_ts: float, *, identity: str = "confirmed", score: float = 0.8):
    return {
        "session_id": "s-1",
        "key": "s-1",
        "camera_id": "home-camera",
        "session_start": best_ts - 10,
        "identity": identity,
        "score": score,
        "best_ts": best_ts,
        "last_event_at": best_ts + 5,
        "payload": {
            "activity_score": 0.4,
            "person_score": 0.9,
            "box": [202, 190, 62, 88],
            "best_box": [202, 190, 62, 88],
            "frame_width": 640,
            "frame_height": 360,
        },
    }


def _index_low_high_segments(database, tmp_path):
    low_path = tmp_path / "low.mp4"
    high_path = tmp_path / "high.mp4"
    low_path.write_bytes(b"low")
    high_path.write_bytes(b"high")
    low_id = database.upsert_segment(
        camera_name="home-camera",
        stream_role="low",
        path=low_path,
        started_at="2026-07-19T10:00:00+08:00",
        ended_at="2026-07-19T10:02:00+08:00",
        duration_seconds=120,
        size_bytes=3,
    )
    database.upsert_segment(
        camera_name="home-camera",
        stream_role="high",
        path=high_path,
        started_at="2026-07-19T10:00:00+08:00",
        ended_at="2026-07-19T10:02:00+08:00",
        duration_seconds=120,
        size_bytes=4,
    )
    return low_id


def _index_high_segments(database, tmp_path, *, split=False):
    rows = [
        ("high-1.mp4", "2026-07-19T10:00:00+08:00", "2026-07-19T10:02:00+08:00")
    ]
    if split:
        rows.append(
            ("high-2.mp4", "2026-07-19T10:02:00+08:00", "2026-07-19T10:04:00+08:00")
        )
    ids = []
    for name, started_at, ended_at in rows:
        path = tmp_path / name
        path.write_bytes(name.encode())
        ids.append(
            database.upsert_segment(
                camera_name="home-camera",
                stream_role="high",
                path=path,
                started_at=started_at,
                ended_at=ended_at,
                duration_seconds=120,
                size_bytes=path.stat().st_size,
            )
        )
    return ids


def test_board_session_obeys_daily_cap(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        output_dir=tmp_path / "output",
        max_moments_per_day=1,
        max_moments_per_period=0,
        moment_cooldown_seconds=0,
        context_after_seconds=10,
    )
    database = Database(tmp_path / "board-cap.sqlite3")
    database.migrate()
    low_id = _index_low_high_segments(database, tmp_path)
    existing_clip = tmp_path / "output" / "2026-07-19" / "existing.mp4"
    existing_clip.parent.mkdir(parents=True)
    existing_clip.write_bytes(b"existing")
    existing_metadata = existing_clip.with_suffix(".json")
    existing_metadata.write_text("{}")
    database.create_moment(
        camera_name="home-camera",
        title="strong existing moment",
        summary="",
        tags=[],
        confidence=0.95,
        source_low_segment_id=low_id,
        source_started_at="2026-07-19T10:00:00+08:00",
        source_ended_at="2026-07-19T10:02:00+08:00",
        clip_path=existing_clip,
        metadata_path=existing_metadata,
        analysis_backend="daughter_detector",
        selection_score=0.95,
        clip_started_at="2026-07-19T10:00:05+08:00",
        clip_ended_at="2026-07-19T10:00:30+08:00",
    )
    supervisor = Supervisor(settings, database)

    async def immediate_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)
    event_time = datetime.fromisoformat("2026-07-19T10:00:40+08:00")
    asyncio.run(
        supervisor._save_board_session(
            _board_session(event_time.timestamp(), identity="probable", score=0.2)
        )
    )

    assert database.count_moments_on_day("2026-07-19") == 1


def test_board_session_saves_moment_from_buffered_segments(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        output_dir=tmp_path / "output",
        context_after_seconds=10,
    )
    database = Database(tmp_path / "board-save.sqlite3")
    database.migrate()
    _index_low_high_segments(database, tmp_path)
    supervisor = Supervisor(settings, database)

    async def immediate_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    async def fake_extract(settings, paths, output_path, **kwargs):
        output_path.write_bytes(b"edge-clip")

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setattr("nas_video_summarizer.workers.extract_clip", fake_extract)
    event_time = datetime.fromisoformat("2026-07-19T10:00:40+08:00")
    asyncio.run(
        supervisor._save_board_session(_board_session(event_time.timestamp()))
    )
    asyncio.run(
        supervisor._save_board_session(_board_session(event_time.timestamp()))
    )

    moments = database.list_moments()
    assert len(moments) == 1
    assert moments[0]["analysis_backend"] == "rv1106_edge"
    assert moments[0]["category"] == "rv1106_confirmed"
    assert Path(moments[0]["clip_path"]).read_bytes() == b"edge-clip"
    assert moments[0]["trigger_key"] == "s-1"
    assert supervisor._last_board_moment_id == moments[0]["id"]
    assert supervisor._last_board_saved_at is not None


def test_board_session_saves_from_high_only_across_segment_boundary(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        output_dir=tmp_path / "output",
        context_before_seconds=5,
        context_after_seconds=10,
        analysis_stream_role="high",
    )
    database = Database(tmp_path / "board-high-only.sqlite3")
    database.migrate()
    high_ids = _index_high_segments(database, tmp_path, split=True)
    supervisor = Supervisor(settings, database)
    captured_paths = []

    async def immediate_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    async def fake_extract(settings, paths, output_path, **kwargs):
        captured_paths.extend(paths)
        output_path.write_bytes(b"high-only")

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setattr("nas_video_summarizer.workers.extract_clip", fake_extract)
    event_time = datetime.fromisoformat("2026-07-19T10:01:58+08:00")

    asyncio.run(
        supervisor._save_board_session(_board_session(event_time.timestamp()))
    )

    moments = database.list_moments()
    assert len(moments) == 1
    assert moments[0]["source_segment_id"] == high_ids[0]
    assert moments[0]["source_low_segment_id"] is None
    assert moments[0]["source_stream_role"] == "high"
    assert [path.name for path in captured_paths] == ["high-1.mp4", "high-2.mp4"]
    metadata = json.loads(Path(moments[0]["metadata_path"]).read_text())
    assert metadata["source_stream_role"] == "high"
    assert "source_low_segment" not in metadata


def test_board_session_uses_tolerance_for_one_second_segment_gap(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        output_dir=tmp_path / "output",
        context_before_seconds=5,
        context_after_seconds=10,
        stream_alignment_tolerance_seconds=2.0,
    )
    database = Database(tmp_path / "board-gap.sqlite3")
    database.migrate()
    rows = [
        ("gap-1.mp4", "2026-07-19T07:56:00+08:00", "2026-07-19T07:58:00+08:00"),
        ("gap-2.mp4", "2026-07-19T07:58:01+08:00", "2026-07-19T08:00:01+08:00"),
    ]
    ids = []
    for name, started_at, ended_at in rows:
        path = tmp_path / name
        path.write_bytes(b"high")
        ids.append(
            database.upsert_segment(
                camera_name="home-camera",
                stream_role="high",
                path=path,
                started_at=started_at,
                ended_at=ended_at,
                duration_seconds=120,
                size_bytes=4,
            )
        )
    supervisor = Supervisor(settings, database)

    async def immediate_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    async def fake_extract(settings, paths, output_path, **kwargs):
        output_path.write_bytes(b"gap-ok")

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setattr("nas_video_summarizer.workers.extract_clip", fake_extract)
    event_time = datetime.fromisoformat("2026-07-19T07:58:00.078+08:00")

    asyncio.run(
        supervisor._save_board_session(_board_session(event_time.timestamp()))
    )

    moment = database.list_moments()[0]
    assert moment["source_segment_id"] == ids[0]
    assert Path(moment["clip_path"]).read_bytes() == b"gap-ok"


def test_probable_board_clip_is_not_published_when_event_verification_fails(
    tmp_path, monkeypatch
):
    settings = replace(
        load_settings("/nonexistent.env"),
        output_dir=tmp_path / "output",
        context_after_seconds=10,
        rv1106_probable_policy="verify",
        rv1106_accept_probable=True,
    )
    database = Database(tmp_path / "board-probable.sqlite3")
    database.migrate()
    _index_high_segments(database, tmp_path)
    supervisor = Supervisor(settings, database)

    async def immediate_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    async def reject_probable(session, event_time):
        verification = ProbableVerification(
            accepted=False,
            positive_frames=0,
            required_frames=1,
            evidence=("adult_face",),
            reason="repeated adult-only face evidence",
            decision="adult_face",
            sampled_frames=5,
            person_frames=5,
            face_frames=2,
            adult_face_frames=2,
        )
        details = {"accepted": False, "decision": "adult_face"}
        supervisor._record_probable_verification(verification, details)
        return verification, details

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setattr(supervisor, "_verify_board_probable_event", reject_probable)
    event_time = datetime.fromisoformat("2026-07-19T10:00:40+08:00")

    asyncio.run(
        supervisor._save_board_session(
            _board_session(event_time.timestamp(), identity="probable")
        )
    )

    assert database.list_moments() == []
    assert not list((tmp_path / "output").rglob("*.mp4"))
    assert supervisor._probable_rejected == 1
    assert supervisor._probable_adult_rejected == 1


def test_probable_event_samples_five_cropped_frames_around_best_ts(
    tmp_path, monkeypatch
):
    settings = replace(
        load_settings("/nonexistent.env"),
        rv1106_probable_policy="verify",
        rv1106_verify_frame_width=960,
    )
    database = Database(tmp_path / "board-roi.sqlite3")
    database.migrate()
    _index_high_segments(database, tmp_path)
    supervisor = Supervisor(settings, database)
    extracted = []

    async def immediate_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    async def fake_extract(
        settings, video_path, output_path, offset_seconds, *, roi, output_width
    ):
        extracted.append((offset_seconds, roi, output_width))
        output_path.write_bytes(b"jpeg")

    class FakeDetector:
        def verify_board_probable_paths(self, paths, **kwargs):
            assert len(paths) == 5
            return ProbableVerification(
                accepted=True,
                positive_frames=1,
                required_frames=2,
                evidence=("child_face",),
                reason="independent child face-age evidence",
                decision="child_face",
                sampled_frames=5,
                person_frames=5,
                face_frames=1,
                child_face_frames=1,
                max_person_score=0.9,
            )

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setattr(
        "nas_video_summarizer.workers.extract_cropped_frame", fake_extract
    )
    monkeypatch.setattr(supervisor, "_get_board_probable_detector", FakeDetector)
    event_time = datetime.fromisoformat("2026-07-19T10:00:40+08:00")

    verification, details = asyncio.run(
        supervisor._verify_board_probable_event(
            _board_session(event_time.timestamp(), identity="probable"),
            event_time,
        )
    )

    assert verification.accepted is True
    assert verification.decision == "child_face"
    assert len(extracted) == 5
    assert [round(item[0]) for item in extracted] == [38, 39, 40, 41, 42]
    assert all(item[2] == 960 for item in extracted)
    assert details["roi"] == pytest.approx(
        [0.21875, 0.405556, 0.290625, 0.488889], rel=1e-5
    )
    assert supervisor._probable_face_verified == 1


def test_faceless_probable_is_saved_with_selection_penalty(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        output_dir=tmp_path / "output",
        context_after_seconds=10,
        rv1106_probable_policy="verify",
        rv1106_accept_probable=True,
        rv1106_verify_faceless_score_multiplier=0.75,
    )
    database = Database(tmp_path / "board-faceless.sqlite3")
    database.migrate()
    _index_high_segments(database, tmp_path)
    supervisor = Supervisor(settings, database)

    async def immediate_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    async def fake_extract(settings, paths, output_path, **kwargs):
        output_path.write_bytes(b"faceless-probable")

    async def accept_faceless(session, event_time):
        verification = ProbableVerification(
            accepted=True,
            positive_frames=0,
            required_frames=1,
            evidence=("person",),
            reason="stable person ROI with strong board evidence",
            decision="faceless_person",
            sampled_frames=5,
            person_frames=4,
            max_person_score=0.9,
        )
        details = {
            "accepted": True,
            "decision": "faceless_person",
            "person_frames": 4,
        }
        supervisor._record_probable_verification(verification, details)
        return verification, details

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setattr("nas_video_summarizer.workers.extract_clip", fake_extract)
    monkeypatch.setattr(supervisor, "_verify_board_probable_event", accept_faceless)
    event_time = datetime.fromisoformat("2026-07-19T10:00:40+08:00")

    asyncio.run(
        supervisor._save_board_session(
            _board_session(event_time.timestamp(), identity="probable", score=0.8)
        )
    )

    moment = database.list_moments()[0]
    assert moment["category"] == "rv1106_probable_faceless"
    assert moment["selection_score"] == pytest.approx(0.525)
    metadata = json.loads(Path(moment["metadata_path"]).read_text())
    assert metadata["model_raw"]["probable_verification"]["accepted"] is True
    assert supervisor._probable_faceless_accepted == 1


def test_high_only_alignment_is_not_applicable(tmp_path):
    settings = replace(
        load_settings("/nonexistent.env"),
        rtsp_low_url="",
        rtsp_high_url="rtsp://camera/high",
    )
    database = Database(tmp_path / "alignment.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)

    supervisor._update_stream_alignment()

    assert supervisor.state["stream_alignment"]["status"] == "not_applicable"


def test_rv1106_backend_finalizes_segments_without_detector(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        analysis_backend="rv1106",
        analysis_delay_seconds=0,
        analysis_interval_seconds=1,
    )
    database = Database(tmp_path / "edge-only.sqlite3")
    database.migrate()
    segment_id = database.upsert_segment(
        camera_name="home-camera",
        stream_role="low",
        path=tmp_path / "low.mp4",
        started_at="2026-07-18T10:00:00+08:00",
        ended_at="2026-07-18T10:02:00+08:00",
        duration_seconds=120,
        size_bytes=100,
    )
    supervisor = Supervisor(settings, database)

    async def stop_after_idle(seconds):
        supervisor.stop_event.set()

    monkeypatch.setattr("nas_video_summarizer.workers.ffmpeg_available", lambda settings: True)
    monkeypatch.setattr(asyncio, "sleep", stop_after_idle)

    asyncio.run(supervisor._analyzer_loop())

    with database.connect() as conn:
        row = conn.execute(
            "SELECT processed_at, analysis_attempts FROM segments WHERE id=?",
            (segment_id,),
        ).fetchone()
    assert row["processed_at"] is not None
    assert row["analysis_attempts"] == 0
    assert supervisor.state["analyzer"]["status"] == "idle"
