from pathlib import Path

import asyncio
from dataclasses import replace

from nas_video_summarizer.ffmpeg_tools import (
    SampledFrame,
    _select_person_filtered_frames,
    _hwaccel_args,
    _motion_aware_offsets,
    build_recorder_command,
    contact_sheet_layout,
    extract_cropped_frame,
    filter_frames_by_person_detection,
    filter_out_blank_frames,
    sample_offsets,
)
from nas_video_summarizer.config import Settings, load_settings


def test_sample_offsets_are_evenly_distributed():
    offsets = sample_offsets(duration_seconds=120, frame_count=4, minimum_spacing_seconds=30)

    assert offsets == [24.0, 48.0, 72.0, 96.0]


def test_sample_offsets_respect_minimum_spacing_limit():
    offsets = sample_offsets(duration_seconds=60, frame_count=8, minimum_spacing_seconds=30)

    assert offsets == [15.0, 30.0, 45.0]


def test_contact_sheet_layout_caps_columns_to_frame_count():
    assert contact_sheet_layout(frame_count=4, preferred_columns=2) == (2, 2)
    assert contact_sheet_layout(frame_count=1, preferred_columns=2) == (1, 1)


def test_recorder_segments_at_wall_clock_by_default():
    settings = load_settings("/nonexistent.env")

    command = build_recorder_command(settings, "low", "rtsp://camera/low")

    index = command.index("-segment_atclocktime")
    assert command[index + 1] == "1"
    assert command[command.index("-reset_timestamps") + 1] == "1"


def test_recorder_wall_clock_alignment_can_be_disabled():
    settings = replace(
        load_settings("/nonexistent.env"), segment_at_clocktime=False
    )

    command = build_recorder_command(settings, "low", "rtsp://camera/low")

    assert "-segment_atclocktime" not in command


def test_extract_cropped_frame_builds_normalized_crop_filter(tmp_path, monkeypatch):
    settings = load_settings("/nonexistent.env")
    captured = []

    async def fake_run(command, error_prefix):
        captured.append((command, error_prefix))

    monkeypatch.setattr("nas_video_summarizer.ffmpeg_tools._run_command", fake_run)
    asyncio.run(
        extract_cropped_frame(
            settings,
            tmp_path / "segment.mp4",
            tmp_path / "frame.jpg",
            12.345,
            roi=(0.1, 0.2, 0.3, 0.4),
            output_width=960,
        )
    )

    command, error_prefix = captured[0]
    filter_value = command[command.index("-vf") + 1]
    assert filter_value == (
        "crop=iw*0.30000000:ih*0.40000000:"
        "iw*0.10000000:ih*0.20000000,scale=960:-2"
    )
    assert command[command.index("-ss") + 1] == "12.345"
    assert error_prefix == "ffmpeg ROI frame extraction failed"


def test_motion_aware_offsets_keeps_static_baseline():
    # 120s segment, motion only in seconds 30-60, 6 frames.
    # At least one sampled offset must land in a static region (0-25 or
    # 65-120) so a child sitting still outside the motion window is still
    # captured instead of being dropped.
    motion_ts = [32.0, 35.0, 40.0, 50.0, 55.0]
    offsets = _motion_aware_offsets(120, motion_ts, 6, bucket_count=12)

    assert len(offsets) == 6
    assert any(o < 25 or o > 65 for o in offsets), offsets


def test_motion_aware_offsets_prefers_high_motion_buckets():
    # Motion concentrated in buckets 3 (30-40s) and 4 (40-50s).
    # Every motion timestamp should become a candidate frame, so the
    # 30-50s region gets at least 3 frames (one per timestamp), while
    # pure even sampling over 120s with 6 frames would put only ~1
    # frame in that region.
    motion_ts = [35.0, 40.0, 45.0]
    offsets = _motion_aware_offsets(120, motion_ts, 6, bucket_count=12)

    assert any(30 <= o < 40 for o in offsets), offsets
    assert any(40 <= o < 50 for o in offsets), offsets
    motion_region_count = sum(1 for o in offsets if 30 <= o < 50)
    assert motion_region_count >= len(motion_ts), offsets


def test_motion_aware_offsets_all_static_falls_back_to_even():
    # No motion timestamps at all: still produce frame_count offsets
    # spread across the segment (every bucket gets 1 frame).
    offsets = _motion_aware_offsets(120, [], 6, bucket_count=12)

    assert len(offsets) == 6
    assert offsets[0] > 0
    assert offsets[-1] < 120


def test_person_filter_selection_preserves_temporal_coverage(tmp_path):
    settings = replace(load_settings("/nonexistent.env"), sample_frame_count=4)
    frames = [
        SampledFrame(tmp_path / f"frame-{index}.jpg", offset)
        for index, offset in enumerate((5.0, 10.0, 35.0, 65.0, 95.0))
    ]
    scores = [
        {"idx": index, "person_score": score, "child_score": 0.0, "adult_only": False}
        for index, score in enumerate((0.99, 0.98, 0.6, 0.6, 0.6))
    ]

    decision = _select_person_filtered_frames(settings, frames, scores)

    assert [frame.offset_seconds for frame in decision.frames] == [5.0, 35.0, 65.0, 95.0]


def _settings_with_hwaccel(hwaccel: str) -> Settings:
    import os
    os.environ["FFMPEG_HWACCEL"] = hwaccel
    return load_settings("/nonexistent.env")


def test_hwaccel_args_empty_when_disabled():
    settings = _settings_with_hwaccel("")
    assert _hwaccel_args(settings) == []


def test_hwaccel_args_vaapi():
    import os
    os.environ["FFMPEG_HWACCEL"] = "vaapi"
    settings = load_settings("/nonexistent.env")
    assert _hwaccel_args(settings) == ["-hwaccel", "vaapi"]


class _FakeProc:
    def __init__(self, stderr):
        self._stderr = stderr.encode() if isinstance(stderr, str) else stderr

    async def communicate(self):
        return (b"", self._stderr)


def _fake_spawn(monkeypatch, luma):
    async def _spawn(*args, **kwargs):
        return _FakeProc(f"lavfi.signalstats.1.luma_average={luma}")
    monkeypatch.setattr("asyncio.create_subprocess_exec", _spawn)


def test_is_blank_frame_black(monkeypatch):
    from nas_video_summarizer import ffmpeg_tools as ft

    _fake_spawn(monkeypatch, 2.34)
    settings = load_settings("/nonexistent.env")
    assert asyncio.run(ft._is_blank_frame(settings, Path("/nope.jpg"))) is True


def test_is_blank_frame_normal(monkeypatch):
    from nas_video_summarizer import ffmpeg_tools as ft

    _fake_spawn(monkeypatch, 120.5)
    settings = load_settings("/nonexistent.env")
    assert asyncio.run(ft._is_blank_frame(settings, Path("/nope.jpg"))) is False


def test_filter_out_blank_frames_drops_all_blank(monkeypatch):
    from nas_video_summarizer import ffmpeg_tools as ft

    _fake_spawn(monkeypatch, 1.0)
    settings = load_settings("/nonexistent.env")
    frames = [SampledFrame(path=Path(f"/f{i}.jpg"), offset_seconds=i) for i in range(3)]
    out = asyncio.run(filter_out_blank_frames(settings, frames))
    assert out == []


def test_filter_out_blank_frames_keeps_mixed(monkeypatch):
    from nas_video_summarizer import ffmpeg_tools as ft

    _fake_spawn(monkeypatch, 100.0)
    settings = load_settings("/nonexistent.env")
    frames = [SampledFrame(path=Path("/f.jpg"), offset_seconds=0)]
    out = asyncio.run(filter_out_blank_frames(settings, frames))
    assert len(out) == 1


def _person_filter_frames(tmp_path, count=3):
    frames = []
    for index in range(count):
        path = tmp_path / f"frame-{index}.jpg"
        path.write_bytes(b"jpeg")
        frames.append(SampledFrame(path=path, offset_seconds=float(index * 10)))
    return frames


def test_person_filter_skips_when_every_person_frame_is_adult(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"), person_filter_enabled=True
    )
    scores = [
        {"person_score": 0.9, "adult_only": True, "adult_score": 0.98},
        {"person_score": 0.8, "adult_only": True, "adult_score": 0.95},
    ]
    decision = _select_person_filtered_frames(
        settings, _person_filter_frames(tmp_path, count=2), scores
    )

    assert decision.frames == []
    assert decision.skip_reason == "adult-only"


def test_person_filter_keeps_and_prioritizes_child_or_uncertain_frames(
    tmp_path, monkeypatch
):
    settings = replace(
        load_settings("/nonexistent.env"),
        person_filter_enabled=True,
        sample_frame_count=2,
    )
    scores = [
        {"person_score": 0.95, "adult_only": True, "child_score": 0.01},
        {"person_score": 0.75, "adult_only": False, "child_score": 0.2},
        {"person_score": 0.8, "adult_only": False, "child_score": 0.85},
    ]
    frames = _person_filter_frames(tmp_path)

    decision = _select_person_filtered_frames(settings, frames, scores)

    assert decision.skip_reason is None
    assert decision.frames == [frames[1], frames[2]]
    assert decision.child_confirmed is True
    assert decision.max_child_score == 0.85


def test_person_filter_age_failure_is_fail_open(tmp_path, monkeypatch):
    from nas_video_summarizer import ffmpeg_tools as ft

    settings = replace(
        load_settings("/nonexistent.env"), person_filter_enabled=True
    )
    frames = _person_filter_frames(tmp_path, count=1)

    async def fail(*args, **kwargs):
        raise RuntimeError("age model failed")

    monkeypatch.setattr(ft.asyncio, "to_thread", fail)

    decision = asyncio.run(filter_frames_by_person_detection(settings, frames))

    assert decision.frames == frames
    assert decision.skip_reason is None
