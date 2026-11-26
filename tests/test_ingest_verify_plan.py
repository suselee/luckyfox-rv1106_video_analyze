import asyncio
from dataclasses import replace

import pytest

from nas_video_summarizer import ingest as ingest_module
from nas_video_summarizer.config import load_settings


BASE_META = {
    "session_id": "1-9",
    "event": "end",
    "identity": "probable",
    "track_id": 9,
    "ts": 1750000025.0,
    "session_start": 1750000018.0,
    "best_ts": 1750000019.0,
    "activity_peak_ts": 1750000020.0,
    "last_active_ts": 1750000020.0,
    "score": 0.72,
    "face_score": 0.0,
    "person_score": 0.85,
    "activity_score": 0.35,
    "box": [0.40, 0.40, 0.52, 0.65],
    "best_box": [0.40, 0.40, 0.52, 0.65],
    "people_count": 1,
    "camera_id": "home-camera",
    "clip_start": 1750000000.0,
    "clip_end": 1750000022.0,
    "clip_bytes": 100,
    "codec": "H264",
    "source": "board_high_ring",
}


def test_plan_verify_targets_anchors_near_best_ts_and_shifts_cleanly():
    """旧板端无 track_samples 时, 锚定 best_ts 附近 ±3s 窗口, 贴尾整体左移不产生重复帧。"""
    meta = dict(BASE_META, clip_start=1000.0, clip_end=1022.0, best_ts=1021.0)
    targets, err = ingest_module._plan_verify_targets(
        meta, width_scale=1.5, height_scale=1.5
    )
    assert err is None
    assert len(targets) == 5
    offsets = [t.offset_seconds for t in targets]
    # duration=22, margin=0.5 -> max_off=21.5; 6s 窗平移到 [15.5, 21.5]
    assert offsets[0] == pytest.approx(15.5)
    assert offsets[-1] == pytest.approx(21.5)
    assert len(set(round(o, 4) for o in offsets)) == 5  # 无重复帧
    # 全部集中在事件活跃尾段, 不再抽 context_before 的早期空画面
    assert all(o >= 15.0 for o in offsets)


def test_plan_verify_targets_falls_back_to_activity_peak_when_best_ts_out_of_clip():
    """best_ts 漂移到 clip_end 之后 (实盘尾裁剪常见), 退回锚定 activity_peak_ts。"""
    meta = dict(
        BASE_META,
        clip_start=1000.0,
        clip_end=1022.0,
        best_ts=1028.4,
        activity_peak_ts=1020.0,
    )
    targets, err = ingest_module._plan_verify_targets(
        meta, width_scale=1.5, height_scale=1.5
    )
    assert err is None
    offsets = [t.offset_seconds for t in targets]
    assert offsets[0] == pytest.approx(15.5)
    assert offsets[-1] == pytest.approx(21.5)


def test_plan_verify_targets_falls_back_to_uniform_when_no_in_clip_anchor():
    meta = dict(
        BASE_META,
        clip_start=1000.0,
        clip_end=1020.0,
        best_ts=900.0,
        activity_peak_ts=0.0,
        last_active_ts=0.0,
    )
    targets, err = ingest_module._plan_verify_targets(
        meta, width_scale=1.5, height_scale=1.5
    )
    assert err is None
    offsets = [round(t.offset_seconds, 2) for t in targets]
    assert offsets == [3.0, 6.0, 10.0, 14.0, 17.0]


def test_plan_verify_targets_uses_eff_start_and_per_frame_track_samples():
    """新板端携带 eff_start/eff_end + track_samples 时按实际关键帧原点对齐并逐帧跟踪 ROI。"""
    # 实际关键帧比 clip_start 早 2.0s (eff_start=998.0), 孩子从画面左侧一路走到右侧
    samples = [
        [1010.0, 0.10, 0.40, 0.20, 0.65],
        [1012.0, 0.20, 0.40, 0.30, 0.65],
        [1014.0, 0.30, 0.40, 0.40, 0.65],
        [1016.0, 0.40, 0.40, 0.50, 0.65],
        [1018.0, 0.50, 0.40, 0.60, 0.65],
        [1020.0, 0.60, 0.40, 0.70, 0.65],
        [1022.0, 0.70, 0.40, 0.80, 0.65],
    ]
    meta = dict(
        BASE_META,
        clip_start=1000.0,
        clip_end=1022.0,
        eff_start=998.0,
        eff_end=1024.0,
        track_samples=samples,
    )
    targets, err = ingest_module._plan_verify_targets(
        meta, width_scale=1.0, height_scale=1.0
    )
    assert err is None
    assert len(targets) == 5
    # 偏移应以 eff_start=998.0 为 0s 原点: 1010->12.0s, 1022->24.0s (夹紧至 max_off=25.5)
    assert targets[0].offset_seconds == pytest.approx(12.0)
    assert targets[-1].offset_seconds == pytest.approx(24.0)
    # ROI 随时间从左向右单调移动, 不再共用同一个静止框
    roi_lefts = [t.roi[0] for t in targets]
    assert roi_lefts == sorted(roi_lefts)
    assert roi_lefts[0] < 0.15 and roi_lefts[-1] > 0.65


def test_plan_verify_targets_sparse_track_samples_binds_nearest_roi():
    """只有 2 个轨迹点时, 围绕两点展开 5 帧并各自绑定最近时刻的 ROI。"""
    meta = dict(
        BASE_META,
        clip_start=1000.0,
        clip_end=1022.0,
        eff_start=999.0,
        eff_end=1023.0,
        track_samples=[
            [1015.0, 0.10, 0.40, 0.20, 0.65],
            [1019.0, 0.60, 0.40, 0.70, 0.65],
        ],
    )
    targets, err = ingest_module._plan_verify_targets(
        meta, width_scale=1.0, height_scale=1.0
    )
    assert err is None
    assert len(targets) == 5
    # 前半段绑定左侧框 (x1=0.10), 后半段绑定右侧框 (x1=0.60)
    assert targets[0].roi[0] == pytest.approx(0.10)
    assert targets[-1].roi[0] == pytest.approx(0.60)


def test_verify_probable_clip_passes_aligned_rois_and_median_height(tmp_path, monkeypatch):
    """端到端校验: extract_roi_and_full_frame 收到逐帧 ROI, detector 收到轨迹框中位高度。"""
    settings = replace(
        load_settings("/nonexistent.env"),
        rv1106_probable_policy="verify",
        rv1106_verify_roi_width_scale=1.0,
        rv1106_verify_roi_height_scale=1.0,
    )
    clip_mp4 = tmp_path / "clip.mp4"
    clip_mp4.write_bytes(b"fake-mp4")

    # 4 帧正常儿童高度 0.25 + 1 帧异常大框 0.70 -> 中位数仍为 0.25 (不被单帧噪声误杀)
    meta = dict(
        BASE_META,
        clip_start=1000.0,
        clip_end=1022.0,
        eff_start=999.0,
        eff_end=1023.0,
        best_box=[0.10, 0.10, 0.30, 0.80],  # 故意设很高 0.70
        track_samples=[
            [1010.0, 0.10, 0.40, 0.20, 0.65],  # h=0.25
            [1012.0, 0.25, 0.40, 0.35, 0.65],  # h=0.25
            [1014.0, 0.40, 0.10, 0.55, 0.80],  # h=0.70 (异常单帧)
            [1016.0, 0.55, 0.40, 0.65, 0.65],  # h=0.25
            [1018.0, 0.70, 0.40, 0.80, 0.65],  # h=0.25
        ],
    )

    extracted: list[tuple[float, tuple[float, float, float, float]]] = []

    async def fake_dual_extract(
        settings_, video_path_, roi_path, full_path, offset, *, roi, output_width
    ):
        extracted.append((offset, roi))
        roi_path.write_bytes(b"roi")
        full_path.write_bytes(b"full")

    seen_heights: list[float] = []

    class _Verifier:
        def verify_board_probable_paths(self, paths, **kwargs):
            seen_heights.append(kwargs["board_box_height"])
            return type("_Res", (), {"accepted": True, "decision": "faceless_person"})()

    monkeypatch.setattr(ingest_module, "extract_roi_and_full_frame", fake_dual_extract)
    accepted, decision, _ = asyncio.run(
        ingest_module._verify_probable_clip(settings, clip_mp4, meta, _Verifier())
    )
    assert accepted is True
    assert decision == "faceless_person"
    assert [round(off, 1) for off, _ in extracted] == [11.0, 13.0, 15.0, 17.0, 19.0]
    assert [round(roi[0], 2) for _, roi in extracted] == [0.10, 0.25, 0.40, 0.55, 0.70]
    assert seen_heights[0] == pytest.approx(0.25)
