from io import BytesIO
from pathlib import Path

from nas_video_summarizer import app as app_module
from nas_video_summarizer.app import (
    AppState,
    RequestHandler,
    _html_shell,
    _moment_api_payload,
)
from nas_video_summarizer.config import load_settings
from nas_video_summarizer.database import Database
from nas_video_summarizer.workers import Supervisor


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
    assert "video_url" not in moment

    handler = object.__new__(RequestHandler)
    handler.state = AppState(settings, database, Supervisor(settings, database))
    handler.headers = {}
    handler.wfile = BytesIO()
    status = []
    headers = {}
    handler.send_response = lambda value: status.append(value)
    handler.send_header = lambda name, value: headers.__setitem__(name, value)
    handler.end_headers = lambda: None
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
