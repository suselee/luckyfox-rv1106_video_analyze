import asyncio
import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest

from nas_video_summarizer import ingest as ingest_module
from nas_video_summarizer.app import AppState, RequestHandler
from nas_video_summarizer.config import load_settings
from nas_video_summarizer.database import Database
from nas_video_summarizer.ingest import (
    IngestSpool,
    ingest_event_key,
    parse_multipart,
    save_ingested_clip,
    validate_meta,
)
from nas_video_summarizer.workers import Supervisor


BOUNDARY = "dwclip-boundary"
SAMPLE_META = {
    "session_id": "1-3",
    "event": "confirmed",
    "identity": "confirmed",
    "track_id": 3,
    "ts": 1750000000.0,
    "session_start": 1749999950.0,
    "best_ts": 1750000005.5,
    "score": 0.91,
    "face_score": 0.88,
    "person_score": 0.93,
    "activity_score": 0.4,
    "box": [0.1, 0.1, 0.2, 0.2],
    "best_box": [0.1, 0.1, 0.2, 0.2],
    "people_count": 1,
    "camera_id": "home-camera",
    "clip_start": 1750000000.0,
    "clip_end": 1750000015.5,
    "clip_bytes": 100,
    "codec": "H264",
    "source": "board_high_ring",
}
VIDEO_BYTES = b"\x00\x00\x00\x01\x67" + bytes(range(200)) + b"\r\n"


def build_multipart(meta, video=VIDEO_BYTES, filename="clip.h264") -> bytes:
    head_meta = (
        f"--{BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="meta"\r\n'
        "Content-Type: application/json\r\n\r\n"
    )
    meta_body = json.dumps(meta).encode("utf-8")
    head_video = (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="video"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    )
    tail = f"\r\n--{BOUNDARY}--\r\n".encode("utf-8")
    return (
        head_meta.encode("utf-8")
        + meta_body
        + b"\r\n"
        + head_video.encode("utf-8")
        + video
        + tail
    )


def test_parse_multipart_roundtrip():
    body = build_multipart(SAMPLE_META)
    parts = parse_multipart(body, BOUNDARY)
    assert [part.name for part in parts] == ["meta", "video"]
    meta_part, video_part = parts
    assert meta_part.filename is None
    assert video_part.filename == "clip.h264"
    assert video_part.data == VIDEO_BYTES


def test_parse_multipart_preserves_crlf_tail():
    """文件部分以 \r\n 结尾的二进制不会被误截。"""
    video_trailing = VIDEO_BYTES + b"\r\n"
    body = build_multipart(SAMPLE_META, video=video_trailing)
    parts = parse_multipart(body, BOUNDARY)
    assert parts[1].data == video_trailing


def test_parse_multipart_rejects_bad_body():
    with pytest.raises(ValueError):
        parse_multipart(b"garbage", BOUNDARY)
    with pytest.raises(ValueError):
        parse_multipart(b"--dwclip-boundary\r\n", "other-boundary")


def test_validate_meta():
    ok, err = validate_meta(dict(SAMPLE_META))
    assert ok is not None and err is None

    missing = dict(SAMPLE_META)
    del missing["clip_start"]
    assert validate_meta(missing)[1] is not None

    bad_ts = dict(SAMPLE_META, clip_start="nan")
    assert validate_meta(bad_ts)[1] is not None

    assert validate_meta("string")[1] is not None


def test_event_key_stability():
    a = ingest_event_key(dict(SAMPLE_META))
    b = ingest_event_key(dict(SAMPLE_META))
    assert a == b
    c = ingest_event_key(dict(SAMPLE_META, clip_end=16.0))
    assert c != a


def test_spool_roundtrip(tmp_path):
    settings = replace(load_settings("/nonexistent.env"), board_ingest_dir=tmp_path / "spool")
    spool = IngestSpool(settings)
    job_dir = spool.spool(dict(SAMPLE_META), VIDEO_BYTES)
    assert job_dir.is_dir()
    assert spool.job_meta(job_dir)["identity"] == "confirmed"
    assert spool.job_video_path(job_dir, dict(SAMPLE_META)).read_bytes() == VIDEO_BYTES
    assert spool.pending_count() == 1
    spool.remove(job_dir)
    assert spool.pending_count() == 0


def test_spool_fail_moves_to_failed(tmp_path):
    settings = replace(load_settings("/nonexistent.env"), board_ingest_dir=tmp_path / "spool")
    spool = IngestSpool(settings)
    job_dir = spool.spool(dict(SAMPLE_META), VIDEO_BYTES)
    spool.fail(job_dir)
    assert (spool.failed_dir / job_dir.name).is_dir()
    assert spool.pending_count() == 0


def _make_settings(tmp_path, ingest=True):
    return replace(
        load_settings("/nonexistent.env"),
        data_dir=tmp_path / "var",
        output_dir=tmp_path / "out",
        board_ingest_enabled=ingest,
        board_ingest_dir=tmp_path / "spool",
        database_path=tmp_path / "app.sqlite3",
    )


def test_save_ingested_clip_publishes_moment(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    database = Database(settings.database_path)
    database.migrate()

    spool = IngestSpool(settings)
    job_dir = spool.spool(dict(SAMPLE_META), VIDEO_BYTES)

    async def fake_remux(settings_, input_path, output_path, **kwargs):
        output_path.write_bytes(input_path.read_bytes())

    monkeypatch.setattr(ingest_module, "remux_elementary_stream", fake_remux)

    async def run_once():
        return await save_ingested_clip(settings, database, spool, job_dir)

    moment_id = asyncio.run(run_once())
    assert moment_id is not None
    assert not job_dir.exists()  # 处理后清理 spool

    moment = database.get_moment(moment_id)
    assert moment is not None
    clip_path = Path(moment["clip_path"])
    assert clip_path.exists()
    assert clip_path.read_bytes() == VIDEO_BYTES
    metadata = json.loads(Path(moment["metadata_path"]).read_text())
    assert metadata["source"] == "board_high_ring"
    assert metadata["analysis_backend"] == "rv1106_edge"
    assert metadata["event_key"] == ingest_event_key(SAMPLE_META)

    # 每日 manifest 已重建
    day = clip_path.parent.name
    manifest = json.loads((settings.output_dir / day / "manifest.json").read_text())
    assert manifest["clip_count"] == 1

    # 幂等: DB 已有 event_key -> 直接复用 moment, 不再重复入库
    second_job = spool.spool(dict(SAMPLE_META), VIDEO_BYTES)
    moment_id_again = asyncio.run(save_ingested_clip(settings, database, spool, second_job))
    assert moment_id_again == moment_id
    assert database.count_ingested_done() == 1


def test_save_ingested_clip_records_error(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    database = Database(settings.database_path)
    database.migrate()

    spool = IngestSpool(settings)
    job_dir = spool.spool(dict(SAMPLE_META), VIDEO_BYTES)

    async def failing_remux(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ingest_module, "remux_elementary_stream", failing_remux)

    async def run():
        with pytest.raises(RuntimeError, match="boom"):
            await save_ingested_clip(settings, database, spool, job_dir)

    asyncio.run(run())
    assert job_dir.exists()  # 失败留队
    assert database.record_ingested_error(job_dir.name, "boom") == 1
    assert database.count_pending_ingested_clips() == 1


def _ingest_handler(settings, database):
    handler = object.__new__(RequestHandler)
    handler.state = AppState(settings, database, Supervisor(settings, database))
    handler.headers = {}
    handler.rfile = BytesIO()
    handler.wfile = BytesIO()
    status = []
    headers = {}
    handler.send_response = lambda value: status.append(value)
    handler.send_header = lambda name, value: headers.__setitem__(name, value)
    handler.end_headers = lambda: None
    stored = {}
    handler._send_json = lambda payload, status=None, **kw: stored.update(
        {"payload": payload, "status": status}
    )
    handler._send_error_json = lambda status, message, **kw: stored.update(
        {"error": (status, message)}
    )
    return handler, stored


def test_ingest_endpoint_disabled(tmp_path):
    settings = load_settings("/nonexistent.env")
    database = Database(tmp_path / "db.sqlite3")
    database.migrate()
    handler, stored = _ingest_handler(settings, database)
    handler._handle_board_ingest()
    assert stored["error"][0].value == 404


def test_ingest_endpoint_rejects_bad_multipart(tmp_path):
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.migrate()
    handler, stored = _ingest_handler(settings, database)
    handler.headers = {
        "Content-Type": "multipart/form-data; boundary=xx",
        "Content-Length": "5",
    }
    handler.rfile = BytesIO(b"garbage")
    handler._handle_board_ingest()
    assert stored["error"][0].value == 400


def test_ingest_endpoint_accepts_and_dedupes(tmp_path):
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.migrate()
    body = build_multipart(SAMPLE_META)
    headers = {
        "Content-Type": f"multipart/form-data; boundary={BOUNDARY}",
        "Content-Length": str(len(body)),
    }

    handler, stored = _ingest_handler(settings, database)
    handler.headers = headers
    handler.rfile = BytesIO(body)
    handler._handle_board_ingest()
    assert stored["status"].value == 202
    assert stored["payload"]["accepted"] is True
    jobs = list((settings.board_ingest_dir / "pending").iterdir())
    assert len(jobs) == 1

    # 同一片段重复上传 -> 只落盘一次
    handler2, stored2 = _ingest_handler(settings, database)
    handler2.headers = headers
    handler2.rfile = BytesIO(body)
    handler2._handle_board_ingest()
    assert stored2["payload"]["duplicate"] is True
    assert len(list((settings.board_ingest_dir / "pending").iterdir())) == 1


def test_ingest_endpoint_too_large(tmp_path):
    settings = replace(
        _make_settings(tmp_path), board_ingest_max_bytes=len(VIDEO_BYTES) - 10
    )
    database = Database(settings.database_path)
    database.migrate()
    body = build_multipart(SAMPLE_META)
    handler, stored = _ingest_handler(settings, database)
    handler.headers = {
        "Content-Type": f"multipart/form-data; boundary={BOUNDARY}",
        "Content-Length": str(len(body)),
    }
    handler.rfile = BytesIO(body)
    handler._handle_board_ingest()
    assert stored["error"][0].value == 413


def _save_clip(tmp_path, monkeypatch, meta, settings=None, database=None):
    settings = settings or _make_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    database = database or Database(settings.database_path)
    database.migrate()
    spool = IngestSpool(settings)
    job_dir = spool.spool(dict(meta), VIDEO_BYTES)

    async def fake_remux(settings_, input_path, output_path, **kwargs):
        output_path.write_bytes(input_path.read_bytes())

    monkeypatch.setattr(ingest_module, "remux_elementary_stream", fake_remux)

    moment_id = asyncio.run(save_ingested_clip(settings, database, spool, job_dir))
    return moment_id, database, spool, job_dir


def test_save_ingested_clip_skips_probable_identity(tmp_path, monkeypatch):
    # reject 策略: probable 上传直接清理, 不保存 (当前默认 accept,
    # verify 走抽帧校验, 各有独立用例)。
    settings = replace(_make_settings(tmp_path), rv1106_probable_policy="reject")
    meta = dict(SAMPLE_META, session_id="9-9", identity="probable")
    moment_id, database, spool, job_dir = _save_clip(
        tmp_path, monkeypatch, meta, settings
    )
    assert moment_id is None
    assert not job_dir.exists()  # probable 上传直接清理, 不保存
    assert database.count_moments_on_day("2025-06-15") == 0
    assert database.count_ingested_done() == 0


def test_save_ingested_clip_daily_cap_replaces_weakest(tmp_path, monkeypatch):
    # 冷却闸门会拦截同刻片段, 本用例只验证每日上限的弱换强, 显式关闭。
    settings = replace(
        _make_settings(tmp_path),
        max_moments_per_day=1,
        rv1106_ingest_cooldown_seconds=0,
    )
    database = Database(settings.database_path)
    database.migrate()

    weak_meta = dict(SAMPLE_META, session_id="1-1", score=0.5, face_score=0.1)
    strong_meta = dict(SAMPLE_META, session_id="2-2", score=0.9, face_score=0.8)

    weak_id, database, _, _ = _save_clip(tmp_path, monkeypatch, weak_meta, settings, database)
    assert weak_id is not None
    weak = database.get_moment(weak_id)
    weak_clip = Path(weak["clip_path"])

    # 更强的新片段 -> 覆盖当天最弱片段 (文件与记录一起移除)
    strong_id, database, _, _ = _save_clip(
        tmp_path, monkeypatch, strong_meta, settings, database
    )
    assert strong_id is not None and strong_id != weak_id
    assert database.get_moment(weak_id) is None
    assert not weak_clip.exists()
    assert database.count_moments_on_day("2025-06-15") == 1
    weak_sel = float(weak["selection_score"])
    assert database.get_moment(strong_id)["selection_score"] > weak_sel

    # 更弱的新片段 -> 直接跳过
    weaker_meta = dict(SAMPLE_META, session_id="3-3", score=0.4, face_score=0.1)
    skip_id, database, _, _ = _save_clip(
        tmp_path, monkeypatch, weaker_meta, settings, database
    )
    assert skip_id is None
    assert database.get_moment(strong_id) is not None
    assert database.count_moments_on_day("2025-06-15") == 1

AUDIO_BYTES = bytes(range(160))


def build_multipart_with_audio(meta, audio=AUDIO_BYTES, audio_name="clip.g711a") -> bytes:
    head_meta = (
        f"--{BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="meta"\r\n'
        "Content-Type: application/json\r\n\r\n"
    )
    meta_body = json.dumps(meta).encode("utf-8")
    head_video = (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="video"; filename="clip.h264"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    )
    head_audio = (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="audio"; filename="{audio_name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    )
    tail = f"\r\n--{BOUNDARY}--\r\n".encode("utf-8")
    return (
        head_meta.encode("utf-8")
        + meta_body
        + b"\r\n"
        + head_video.encode("utf-8")
        + VIDEO_BYTES
        + b"\r\n"
        + head_audio.encode("utf-8")
        + audio
        + tail
    )


def test_parse_multipart_with_audio():
    body = build_multipart_with_audio(SAMPLE_META)
    parts = parse_multipart(body, BOUNDARY)
    assert [part.name for part in parts] == ["meta", "video", "audio"]
    assert parts[2].filename == "clip.g711a"
    assert parts[2].data == AUDIO_BYTES


def test_spool_writes_audio_file(tmp_path):
    settings = replace(load_settings("/nonexistent.env"), board_ingest_dir=tmp_path / "spool")
    spool = IngestSpool(settings)
    meta = dict(SAMPLE_META, audio_codec="PCMA", audio_rate=8000, audio_channels=1)
    job_dir = spool.spool(dict(meta), VIDEO_BYTES, audio=AUDIO_BYTES, audio_name="clip.g711a")
    audio_path = spool.job_audio_path(job_dir, dict(meta))
    assert audio_path is not None
    assert audio_path.read_bytes() == AUDIO_BYTES

    no_audio = spool.job_audio_path(job_dir, dict(SAMPLE_META))
    assert no_audio is None


def test_ingest_endpoint_accepts_audio_part(tmp_path):
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.migrate()
    meta = dict(SAMPLE_META, audio_codec="PCMA", audio_rate=8000, audio_channels=1)
    body = build_multipart_with_audio(meta)

    handler, stored = _ingest_handler(settings, database)
    handler.headers = {
        "Content-Type": f"multipart/form-data; boundary={BOUNDARY}",
        "Content-Length": str(len(body)),
    }
    handler.rfile = BytesIO(body)
    handler._handle_board_ingest()
    assert stored["payload"]["accepted"] is True

    spool = IngestSpool(settings)
    jobs = spool.pending_jobs()
    assert len(jobs) == 1
    audio_path = spool.job_audio_path(jobs[0], dict(meta))
    assert audio_path is not None and audio_path.name == "clip.g711a"
    assert audio_path.read_bytes() == AUDIO_BYTES


def test_ingest_endpoint_rejects_unknown_audio_filename(tmp_path):
    settings = _make_settings(tmp_path)
    database = Database(settings.database_path)
    database.migrate()
    body = build_multipart_with_audio(SAMPLE_META, audio_name="clip.exe")

    handler, stored = _ingest_handler(settings, database)
    handler.headers = {
        "Content-Type": f"multipart/form-data; boundary={BOUNDARY}",
        "Content-Length": str(len(body)),
    }
    handler.rfile = BytesIO(body)
    handler._handle_board_ingest()
    assert stored["payload"]["accepted"] is True
    spool = IngestSpool(settings)
    jobs = spool.pending_jobs()
    assert len(jobs) == 1
    assert spool.job_audio_path(jobs[0], dict(SAMPLE_META)) is None


def test_save_ingested_clip_passes_audio_to_remux(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    database = Database(settings.database_path)
    database.migrate()

    meta = dict(SAMPLE_META, audio_codec="PCMA", audio_rate=8000, audio_channels=1)
    spool = IngestSpool(settings)
    job_dir = spool.spool(dict(meta), VIDEO_BYTES, audio=AUDIO_BYTES, audio_name="clip.g711a")

    calls = {}

    async def fake_remux(settings_, input_path, output_path, **kwargs):
        calls.update(kwargs)
        output_path.write_bytes(input_path.read_bytes())

    monkeypatch.setattr(ingest_module, "remux_elementary_stream", fake_remux)
    moment_id = asyncio.run(save_ingested_clip(settings, database, spool, job_dir))
    assert moment_id is not None
    assert calls["audio_path"].name == "clip.g711a"
    assert calls["audio_format"] == "alaw"
    assert calls["audio_rate"] == 8000
    assert calls["audio_channels"] == 1


def test_save_ingested_clip_audio_failure_falls_back_to_video_only(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    database = Database(settings.database_path)
    database.migrate()

    meta = dict(SAMPLE_META, audio_codec="PCMA", audio_rate=8000, audio_channels=1)
    spool = IngestSpool(settings)
    job_dir = spool.spool(dict(meta), VIDEO_BYTES, audio=AUDIO_BYTES, audio_name="clip.g711a")

    attempts = []

    async def fake_remux(settings_, input_path, output_path, **kwargs):
        attempts.append(kwargs.get("audio_path"))
        if kwargs.get("audio_path") is not None:
            raise RuntimeError("bad audio")
        output_path.write_bytes(input_path.read_bytes())

    monkeypatch.setattr(ingest_module, "remux_elementary_stream", fake_remux)
    moment_id = asyncio.run(save_ingested_clip(settings, database, spool, job_dir))
    assert moment_id is not None
    assert len(attempts) == 2  # 第一次带音频失败, 第二次纯视频成功
    assert attempts[1] is None
    clip = database.get_moment(moment_id)
    assert Path(clip["clip_path"]).read_bytes() == VIDEO_BYTES


def _probable_meta() -> dict:
    return dict(
        SAMPLE_META,
        identity="probable",
        event="end",
        score=0.75,
        face_score=0.0,
        person_score=0.9,
        activity_score=0.3,
    )


class _FakeVerification:
    def __init__(self, accepted: bool, decision: str):
        self.accepted = accepted
        self.decision = decision


class _FakeVerifier:
    def __init__(self, accepted: bool, decision: str = "child_face"):
        self.accepted = accepted
        self.decision = decision
        self.calls: list[list[Path]] = []

    def verify_board_probable_paths(self, paths, **kwargs):
        self.calls.append(list(paths))
        return _FakeVerification(self.accepted, self.decision)


def test_save_ingested_clip_verify_accepts_probable(tmp_path, monkeypatch):
    settings = replace(_make_settings(tmp_path), rv1106_probable_policy="verify")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    database = Database(settings.database_path)
    database.migrate()

    spool = IngestSpool(settings)
    job_dir = spool.spool(_probable_meta(), VIDEO_BYTES)

    async def fake_remux(settings_, input_path, output_path, **kwargs):
        output_path.write_bytes(input_path.read_bytes())

    async def fake_extract(settings_, video_path_, output_path, offset, *, roi, output_width):
        output_path.write_bytes(b"jpg")

    monkeypatch.setattr(ingest_module, "remux_elementary_stream", fake_remux)
    monkeypatch.setattr(ingest_module, "extract_cropped_frame", fake_extract)

    verifier = _FakeVerifier(accepted=True)
    moment_id = asyncio.run(
        save_ingested_clip(settings, database, spool, job_dir, probable_verifier=verifier)
    )
    assert moment_id is not None
    assert len(verifier.calls) == 1
    assert len(verifier.calls[0]) == 5  # 均匀抽 5 帧
    assert not job_dir.exists()


def test_save_ingested_clip_verify_rejects_adult_probable(tmp_path, monkeypatch):
    settings = replace(_make_settings(tmp_path), rv1106_probable_policy="verify")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    database = Database(settings.database_path)
    database.migrate()

    spool = IngestSpool(settings)
    job_dir = spool.spool(_probable_meta(), VIDEO_BYTES)

    async def fake_extract(settings_, video_path_, output_path, offset, *, roi, output_width):
        output_path.write_bytes(b"jpg")

    monkeypatch.setattr(ingest_module, "extract_cropped_frame", fake_extract)

    verifier = _FakeVerifier(accepted=False, decision="adult_face")
    moment_id = asyncio.run(
        save_ingested_clip(settings, database, spool, job_dir, probable_verifier=verifier)
    )
    assert moment_id is None
    assert not job_dir.exists()  # 拒绝后清理 spool
    assert database.count_ingested_done() == 0


def test_save_ingested_clip_verify_fail_open_without_verifier(tmp_path, monkeypatch):
    settings = replace(_make_settings(tmp_path), rv1106_probable_policy="verify")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    database = Database(settings.database_path)
    database.migrate()

    spool = IngestSpool(settings)
    job_dir = spool.spool(_probable_meta(), VIDEO_BYTES)

    async def fake_remux(settings_, input_path, output_path, **kwargs):
        output_path.write_bytes(input_path.read_bytes())

    monkeypatch.setattr(ingest_module, "remux_elementary_stream", fake_remux)

    # 无校验器 (OpenCV 缺失等): 宁多存, 放行。
    moment_id = asyncio.run(
        save_ingested_clip(settings, database, spool, job_dir, probable_verifier=None)
    )
    assert moment_id is not None


def test_save_ingested_clip_verify_confirmed_skips_verification(tmp_path, monkeypatch):
    settings = replace(_make_settings(tmp_path), rv1106_probable_policy="verify")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    database = Database(settings.database_path)
    database.migrate()

    spool = IngestSpool(settings)
    job_dir = spool.spool(dict(SAMPLE_META), VIDEO_BYTES)  # identity=confirmed

    async def fake_remux(settings_, input_path, output_path, **kwargs):
        output_path.write_bytes(input_path.read_bytes())

    monkeypatch.setattr(ingest_module, "remux_elementary_stream", fake_remux)

    verifier = _FakeVerifier(accepted=False)  # 若被调用将拒绝
    moment_id = asyncio.run(
        save_ingested_clip(settings, database, spool, job_dir, probable_verifier=verifier)
    )
    assert moment_id is not None
    assert verifier.calls == []  # confirmed 不走校验
