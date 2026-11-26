import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from nas_video_summarizer import ingest as ingest_module
from nas_video_summarizer.config import load_settings
from nas_video_summarizer.database import Database
from nas_video_summarizer.daughter_detector import DaughterDetector
from nas_video_summarizer.face_verifier import DaughterFaceVerifier, FaceVerifyResult


BASE_META = {
    "session_id": "1-42",
    "event": "end",
    "identity": "probable",
    "track_id": 42,
    "ts": 1750000025.0,
    "session_start": 1750000018.0,
    "best_ts": 1750000019.0,
    "activity_peak_ts": 1750000020.0,
    "last_active_ts": 1750000020.0,
    "score": 0.76,
    "face_score": 0.0,
    "person_score": 0.85,
    "activity_score": 0.40,
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


def test_face_verifier_readiness_and_graceful_fallback(tmp_path):
    model = tmp_path / "w600k_r50.onnx"
    gallery = tmp_path / "daughter_r50.npz"
    yunet = tmp_path / "face_detection_yunet_2023mar.onnx"

    verifier = DaughterFaceVerifier(
        model_path=model, gallery_path=gallery, yunet_path=yunet, threshold=0.36
    )
    assert verifier.is_ready() is False
    res = verifier.verify_paths([tmp_path / "a.jpg"])
    assert res == FaceVerifyResult(
        matched=False,
        best_similarity=0.0,
        hit_frames=0,
        faces_checked=0,
        best_face_px=0,
    )

    model.write_bytes(b"m")
    gallery.write_bytes(b"g")
    assert verifier.is_ready() is False
    yunet.write_bytes(b"y")
    assert verifier.is_ready() is True


def test_daughter_detector_face_verifier_toggle(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = model_dir / "w600k_r50.onnx"
    gallery = model_dir / "daughter_r50.npz"
    yunet = model_dir / "face_detection_yunet_2023mar.onnx"
    for path in (model, gallery, yunet):
        path.write_bytes(b"x")

    base = load_settings("/nonexistent.env")
    disabled_settings = replace(
        base,
        person_filter_model_dir=model_dir,
        rv1106_verify_face_recog_enabled=False,
        rv1106_verify_face_recog_model_path=model,
        rv1106_verify_face_recog_gallery_path=gallery,
    )
    assert DaughterDetector(disabled_settings).get_face_verifier() is None

    enabled_settings = replace(disabled_settings, rv1106_verify_face_recog_enabled=True)
    verifier = DaughterDetector(enabled_settings).get_face_verifier()
    assert verifier is not None
    assert verifier.threshold == pytest.approx(enabled_settings.rv1106_verify_face_recog_threshold)


def test_verify_probable_clip_accepts_4k_daughter_face_early(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        rv1106_probable_policy="verify",
        rv1106_verify_face_recog_enabled=True,
        rv1106_verify_face_recog_full_width=1920,
    )
    clip_mp4 = tmp_path / "clip.mp4"
    clip_mp4.write_bytes(b"fake-mp4")

    seen_full_widths: list[int | None] = []

    async def fake_dual_extract(
        settings_,
        video_path_,
        roi_path,
        full_path,
        offset,
        *,
        roi,
        output_width,
        full_output_width=None,
    ):
        seen_full_widths.append(full_output_width)
        roi_path.write_bytes(b"roi")
        full_path.write_bytes(b"full")

    heuristic_called = []

    class _Detector:
        def verify_daughter_face_paths(self, paths):
            assert len(paths) == 5
            assert all("clip-full-" in p.name for p in paths)
            return FaceVerifyResult(
                matched=True,
                best_similarity=0.642,
                hit_frames=2,
                faces_checked=3,
                best_face_px=74,
            )

        def verify_board_probable_paths(self, paths, **kwargs):
            heuristic_called.append(True)
            raise AssertionError("heuristic should not run after daughter_face match")

    monkeypatch.setattr(ingest_module, "extract_roi_and_full_frame", fake_dual_extract)
    accepted, decision, verification = asyncio.run(
        ingest_module._verify_probable_clip(settings, clip_mp4, BASE_META, _Detector())
    )
    assert accepted is True
    assert decision == "daughter_face"
    assert verification.evidence == ("daughter_face",)
    assert verification.face_frames == 2
    assert seen_full_widths == [1920] * 5
    assert heuristic_called == []


def test_verify_probable_clip_falls_through_when_4k_face_misses(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        rv1106_probable_policy="verify",
        rv1106_verify_face_recog_enabled=True,
    )
    clip_mp4 = tmp_path / "clip.mp4"
    clip_mp4.write_bytes(b"fake-mp4")

    async def fake_dual_extract(
        settings_,
        video_path_,
        roi_path,
        full_path,
        offset,
        *,
        roi,
        output_width,
        full_output_width=None,
    ):
        roi_path.write_bytes(b"roi")
        full_path.write_bytes(b"full")

    calls = []

    class _Detector:
        def verify_daughter_face_paths(self, paths):
            return FaceVerifyResult(
                matched=False,
                best_similarity=0.19,
                hit_frames=0,
                faces_checked=2,
                best_face_px=80,
            )

        def verify_board_probable_paths(self, paths, **kwargs):
            calls.append(len(paths))
            return type(
                "_Res",
                (),
                {
                    "accepted": False,
                    "decision": "face_without_child",
                    "person_frames": 3,
                    "max_person_height": 0.55,
                },
            )()

    monkeypatch.setattr(ingest_module, "extract_roi_and_full_frame", fake_dual_extract)
    accepted, decision, _ = asyncio.run(
        ingest_module._verify_probable_clip(settings, clip_mp4, BASE_META, _Detector())
    )
    assert accepted is False
    assert decision == "face_without_child"
    assert len(calls) == 2  # ROI + full-frame fallback


def test_save_ingested_clip_marks_daughter_face_as_confirmed_child(tmp_path, monkeypatch):
    base = load_settings("/nonexistent.env")
    settings = replace(
        base,
        data_dir=tmp_path / "var",
        output_dir=tmp_path / "out",
        board_ingest_dir=tmp_path / "board_ingest",
        rv1106_probable_policy="verify",
        rv1106_ingest_cooldown_seconds=0.0,
    )
    db = Database(tmp_path / "app.sqlite3")
    db.migrate()
    spool = ingest_module.IngestSpool(settings)
    job_dir = spool.spool(BASE_META, b"fake-h264")

    async def fake_remux(settings_, video_path, audio_path, meta, output_path, **kwargs):
        output_path.write_bytes(b"muxed-mp4")

    async def fake_verify(settings_, video_path, meta, detector, **kwargs):
        return True, "daughter_face", None

    monkeypatch.setattr(ingest_module, "_remux_with_audio_fallback", fake_remux)
    monkeypatch.setattr(ingest_module, "_verify_probable_clip", fake_verify)

    moment_id = asyncio.run(
        ingest_module.save_ingested_clip(
            settings, db, spool, job_dir, probable_verifier=object()
        )
    )
    assert moment_id is not None
    moment = db.get_moment(moment_id)
    assert moment is not None
    assert "NAS安检通过(daughter_face)" in moment["summary"]
    meta_json = json.loads(Path(moment["metadata_path"]).read_text(encoding="utf-8"))
    assert meta_json["local_child_confirmed"] is True
    assert meta_json["local_child_score"] == pytest.approx(1.0)
