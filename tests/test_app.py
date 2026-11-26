from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path

from nas_video_summarizer import app as app_module
from nas_video_summarizer.app import (
    AppState,
    RequestHandler,
    _html_shell,
    _moment_api_payload,
    _rejected_api_payload,
    _rejected_name_from_path,
    _safe_rejected_entry,
)
from nas_video_summarizer.config import load_settings
from nas_video_summarizer.database import Database
from nas_video_summarizer.workers import Supervisor


def _make_handler(
    settings,
    database: Database,
    *,
    path: str = "/",
    body: bytes = b"",
):
    handler = object.__new__(RequestHandler)
    handler.state = AppState(settings, database, Supervisor(settings, database))
    handler.path = path
    handler.headers = {"Content-Length": str(len(body))} if body else {}
    handler.rfile = BytesIO(body)
    handler.wfile = BytesIO()
    statuses: list[int] = []
    headers: dict[str, str] = {}
    handler.send_response = lambda value: statuses.append(value)
    handler.send_header = lambda name, value: headers.__setitem__(name, value)
    handler.end_headers = lambda: None
    return handler, statuses, headers


def test_web_lists_downloads_without_video_previews(tmp_path):
    settings = load_settings("/nonexistent.env")
    database = Database(tmp_path / "app.sqlite3")
    database.migrate()
    clip = tmp_path / "moment.mp4"
    metadata = tmp_path / "moment.json"
    clip.write_bytes(b"h265-video")
    metadata.write_text("{}")
    moment_id = database.create_moment(
        camera_name="home-camera",
        title="Daughter playing",
        summary="",
        tags=["daughter"],
        confidence=0.9,
        source_low_segment_id=None,
        source_started_at="2026-07-19T10:00:00+08:00",
        source_ended_at="2026-07-19T10:02:00+08:00",
        clip_path=clip,
        metadata_path=metadata,
    )

    moment = _moment_api_payload(database, 100)[0]
    assert moment["download_url"] == f"/api/moments/{moment_id}/download"
    assert moment["preview_url"] == f"/api/moments/{moment_id}/preview.jpg"
    assert moment["feedback_label"] is None
    assert "video_url" not in moment

    handler, status, headers = _make_handler(settings, database)
    handler._send_moment_download(moment_id)

    assert status == [200]
    assert handler.wfile.getvalue() == b"h265-video"
    assert headers["Content-Type"] == "application/octet-stream"
    assert headers["Content-Disposition"] == 'attachment; filename="moment.mp4"'

    javascript = (
        Path(app_module.__file__).parent / "static" / "app.js"
    ).read_text()
    assert "<video" not in javascript
    assert 'preload="metadata"' not in javascript
    assert "/api/comparison" not in javascript
    assert "comparison" not in _html_shell()


def test_moment_feedback_tag_and_delete_archives_hard_negative(tmp_path):
    base = load_settings("/nonexistent.env")
    settings = replace(
        base,
        data_dir=tmp_path / "var",
        output_dir=tmp_path / "out",
        board_ingest_dir=tmp_path / "board_ingest",
        rv1106_keep_rejected=True,
    )
    database = Database(tmp_path / "app.sqlite3")
    database.migrate()

    day_dir = settings.output_dir / "2026-11-26"
    day_dir.mkdir(parents=True)
    clip = day_dir / "100000_probable.mp4"
    metadata = day_dir / "100000_probable.json"
    clip.write_bytes(b"fake-mp4-bytes")
    metadata.write_text(
        json.dumps(
            {
                "category": "rv1106_probable",
                "model_raw": {
                    "session_id": "s1",
                    "clip_start": 100.0,
                    "clip_end": 120.0,
                    "best_ts": 110.0,
                    "best_box": [0.2, 0.2, 0.4, 0.7],
                },
            }
        ),
        encoding="utf-8",
    )
    moment_id = database.create_moment(
        camera_name="home-camera",
        title="Daughter probable",
        summary="",
        tags=["daughter"],
        confidence=0.82,
        source_low_segment_id=None,
        source_started_at="2026-11-26T10:00:00+08:00",
        source_ended_at="2026-11-26T10:00:20+08:00",
        clip_path=clip,
        metadata_path=metadata,
        category="rv1106_probable",
        trigger_key="ev-fp-1",
    )

    # 1. Tag as daughter (keep clip)
    body = json.dumps({"label": "daughter", "delete_clip": False}).encode("utf-8")
    handler, statuses, _ = _make_handler(
        settings, database, path=f"/api/moments/{moment_id}/feedback", body=body
    )
    handler.do_POST()
    assert statuses == [200]
    resp = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert resp["label"] == "daughter"
    assert resp["deleted"] is False
    assert clip.exists()
    assert _moment_api_payload(database, 10)[0]["feedback_label"] == "daughter"

    # 2. Re-tag as adult_only with delete_clip=True -> archives to rejected/ & deletes moment
    body2 = json.dumps({"label": "adult_only", "delete_clip": True}).encode("utf-8")
    handler2, statuses2, _ = _make_handler(
        settings, database, path=f"/api/moments/{moment_id}/feedback", body=body2
    )
    handler2.do_POST()
    assert statuses2 == [200]
    resp2 = json.loads(handler2.wfile.getvalue().decode("utf-8"))
    assert resp2["deleted"] is True
    assert resp2["archived_path"] is not None
    archived = Path(resp2["archived_path"])
    assert archived.is_file()
    assert archived.read_bytes() == b"fake-mp4-bytes"
    decision = json.loads((archived.parent / "decision.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "human_adult_only"
    assert decision["meta"]["best_ts"] == 110.0

    assert not clip.exists()
    assert not metadata.exists()
    assert database.get_moment(moment_id) is None

    fb_rows = database.list_clip_feedback(label="adult_only")
    assert len(fb_rows) == 1
    assert fb_rows[0]["deleted_clip"] is True
    assert fb_rows[0]["clip_path"] == str(archived)
    assert fb_rows[0]["meta"]["best_ts"] == 110.0


def test_rejected_queue_list_feedback_and_traversal_guard(tmp_path):
    base = load_settings("/nonexistent.env")
    settings = replace(
        base,
        data_dir=tmp_path / "var",
        board_ingest_dir=tmp_path / "board_ingest",
    )
    database = Database(tmp_path / "app.sqlite3")
    database.migrate()

    rej_entry = settings.board_ingest_dir / "rejected" / "20261126-100000_abc123"
    rej_entry.mkdir(parents=True)
    (rej_entry / "clip.hevc").write_bytes(b"hevc-data")
    (rej_entry / "decision.json").write_text(
        json.dumps(
            {
                "decision": "face_without_child",
                "archived_at": "2026-11-26T10:00:05",
                "meta": {"identity": "probable", "score": 0.77, "camera_id": "cam1"},
            }
        ),
        encoding="utf-8",
    )

    items = _rejected_api_payload(settings, database, 20)
    assert len(items) == 1
    assert items[0]["name"] == "20261126-100000_abc123"
    assert items[0]["decision"] == "face_without_child"
    assert items[0]["preview_url"] == "/api/rejected/20261126-100000_abc123/preview.jpg"
    assert items[0]["feedback_label"] is None

    # Submit feedback on rejected item
    body = json.dumps({"label": "daughter"}).encode("utf-8")
    handler, statuses, _ = _make_handler(
        settings,
        database,
        path="/api/rejected/20261126-100000_abc123/feedback",
        body=body,
    )
    handler.do_POST()
    assert statuses == [200]
    resp = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert resp["label"] == "daughter"
    assert resp["summary"]["by_source"]["rejected"]["daughter"] == 1
    assert _rejected_api_payload(settings, database, 20)[0]["feedback_label"] == "daughter"

    # Path traversal attempts must be rejected
    assert _rejected_name_from_path("/api/rejected/../etc/feedback", "/feedback") is None
    assert _rejected_name_from_path("/api/rejected/..%2Fetc/feedback", "/feedback") is None
    assert _rejected_name_from_path("/api/rejected/a/b/feedback", "/feedback") is None
    assert _safe_rejected_entry(settings, "..") is None
    assert _safe_rejected_entry(settings, "../board_ingest") is None


def test_preview_jpeg_caches_and_serves_image(tmp_path, monkeypatch):
    base = load_settings("/nonexistent.env")
    settings = replace(
        base,
        data_dir=tmp_path / "var",
        board_ingest_dir=tmp_path / "board_ingest",
    )
    database = Database(tmp_path / "app.sqlite3")
    database.migrate()

    clip = tmp_path / "moment.mp4"
    metadata = tmp_path / "moment.json"
    clip.write_bytes(b"mp4-content")
    metadata.write_text("{}", encoding="utf-8")
    moment_id = database.create_moment(
        camera_name="home-camera",
        title="Smile",
        summary="",
        tags=["daughter"],
        confidence=0.95,
        source_low_segment_id=None,
        source_started_at="2026-11-26T10:00:00+08:00",
        source_ended_at="2026-11-26T10:00:15+08:00",
        clip_path=clip,
        metadata_path=metadata,
    )

    calls = []

    def fake_extract(_settings, _clip_path, _meta, output_path: Path) -> bool:
        calls.append(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
        return True

    monkeypatch.setattr(app_module, "_extract_preview_jpeg", fake_extract)

    handler1, status1, headers1 = _make_handler(
        settings, database, path=f"/api/moments/{moment_id}/preview.jpg"
    )
    handler1.do_GET()
    assert status1 == [200]
    assert headers1["Content-Type"] == "image/jpeg"
    assert headers1["Cache-Control"] == "public, max-age=86400"
    assert handler1.wfile.getvalue() == b"\xff\xd8\xff\xe0fake-jpeg"
    assert len(calls) == 1

    # Second request hits cache without calling ffmpeg extractor again
    handler2, status2, _ = _make_handler(
        settings, database, path=f"/api/moments/{moment_id}/preview.jpg"
    )
    handler2.do_GET()
    assert status2 == [200]
    assert handler2.wfile.getvalue() == b"\xff\xd8\xff\xe0fake-jpeg"
    assert len(calls) == 1



def test_rejected_rescue_publishes_moment_and_is_idempotent(tmp_path, monkeypatch):
    base = load_settings("/nonexistent.env")
    settings = replace(
        base,
        data_dir=tmp_path / "var",
        output_dir=tmp_path / "out",
        board_ingest_dir=tmp_path / "board_ingest",
    )
    database = Database(tmp_path / "app.sqlite3")
    database.migrate()

    entry = settings.board_ingest_dir / "rejected" / "20261126-101500_rescue01"
    entry.mkdir(parents=True)
    (entry / "clip.hevc").write_bytes(b"hevc-stream")
    (entry / "decision.json").write_text(
        json.dumps(
            {
                "decision": "face_without_child",
                "archived_at": "2026-11-26T10:15:10",
                "meta": {
                    "identity": "probable",
                    "score": 0.81,
                    "camera_id": "living-room",
                    "codec": "H265",
                    "clip_start": 1764123300.0,
                    "clip_end": 1764123320.0,
                },
            }
        ),
        encoding="utf-8",
    )

    remux_calls = []

    def fake_remux(settings_, video_path, audio_path, meta, output_path, *, creation_time):
        remux_calls.append((video_path.name, creation_time))
        output_path.write_bytes(b"rescued-mp4-bytes")

    monkeypatch.setattr(app_module, "_remux_rejected_clip_sync", fake_remux)

    # First rescue publishes MP4 + JSON + DB moment + daughter feedback
    handler1, statuses1, _ = _make_handler(
        settings,
        database,
        path="/api/rejected/20261126-101500_rescue01/rescue",
        body=b"{}",
    )
    handler1.do_POST()
    assert statuses1 == [200]
    resp1 = json.loads(handler1.wfile.getvalue().decode("utf-8"))
    assert resp1["rescued"] is True
    assert resp1["already_rescued"] is False
    moment_id = resp1["moment_id"]
    assert len(remux_calls) == 1

    moment = database.get_moment(moment_id)
    assert moment is not None
    assert moment["category"] == "rv1106_rescued"
    assert moment["trigger_key"] == "rescued:20261126-101500_rescue01"
    assert Path(moment["clip_path"]).read_bytes() == b"rescued-mp4-bytes"
    assert Path(moment["metadata_path"]).is_file()
    day_dir = Path(moment["clip_path"]).parent
    assert (day_dir / "manifest.json").is_file()
    assert (day_dir / "summary.md").is_file()

    # Rejected list marks it as rescued & labeled daughter
    items = _rejected_api_payload(settings, database, 20)
    assert items[0]["rescued"] is True
    assert items[0]["rescued_moment_id"] == moment_id
    assert items[0]["feedback_label"] == "daughter"

    # Second rescue is idempotent and does not call remux again
    handler2, statuses2, _ = _make_handler(
        settings,
        database,
        path="/api/rejected/20261126-101500_rescue01/rescue",
        body=b"{}",
    )
    handler2.do_POST()
    assert statuses2 == [200]
    resp2 = json.loads(handler2.wfile.getvalue().decode("utf-8"))
    assert resp2["already_rescued"] is True
    assert resp2["moment_id"] == moment_id
    assert len(remux_calls) == 1
    assert len(database.list_moments(limit=20)) == 1

    # Traversal attempt returns 404
    handler3, statuses3, _ = _make_handler(
        settings,
        database,
        path="/api/rejected/..%2Fetc/rescue",
        body=b"{}",
    )
    handler3.do_POST()
    assert statuses3 == [404]
