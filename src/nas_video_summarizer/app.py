from __future__ import annotations

import asyncio
import errno
import json
import mimetypes
import signal
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .archive import rebuild_day_archive
from .config import Settings, ensure_directories, load_settings
from .database import Database
from .workers import Supervisor, health_snapshot


def _html_shell() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>NAS Video Moments</title>
        <link rel="stylesheet" href="/static/app.css" />
      </head>
      <body>
        <main class="layout">
          <header class="topbar">
            <div>
              <h1>NAS Video Moments</h1>
              <p>RTSP capture, local model analysis, Nextcloud moment archive.</p>
            </div>
            <button id="refresh-button" class="button">Refresh</button>
          </header>

          <section class="status-grid" id="status-grid"></section>

          <section class="section-header">
            <div>
              <h2>Saved Moments</h2>
              <p>Automatically saved 4K clips. Download them for local playback in VLC.</p>
            </div>
          </section>

          <section id="moments" class="moments"></section>
        </main>
        <script src="/static/app.js"></script>
      </body>
    </html>
    """


@dataclass
class AppState:
    settings: Settings
    database: Database
    supervisor: Supervisor


class WorkerRuntime:
    def __init__(self, supervisor: Supervisor):
        self.supervisor = supervisor
        self.loop = asyncio.new_event_loop()
        self.started = threading.Event()
        self.thread = threading.Thread(target=self._run, name="nas-video-workers", daemon=True)

    def start(self) -> None:
        self.thread.start()
        self.started.wait(timeout=10)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=15)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.supervisor.start())
        self.started.set()
        try:
            self.loop.run_forever()
        finally:
            self.loop.run_until_complete(self.supervisor.stop())
            self.loop.close()


def create_state() -> AppState:
    settings = load_settings()
    ensure_directories(settings)
    database = Database(settings.database_path)
    database.migrate()
    supervisor = Supervisor(settings, database)
    return AppState(settings=settings, database=database, supervisor=supervisor)


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _is_client_disconnect(exc: BaseException) -> bool:
    if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
        return True
    if isinstance(exc, OSError) and exc.errno in (errno.EPIPE, errno.ECONNRESET):
        return True
    return False


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _moment_id_from_path(path: str, suffix: str = "") -> int | None:
    prefix = "/api/moments/"
    if not path.startswith(prefix):
        return None
    tail = path.removeprefix(prefix)
    if suffix:
        if not tail.endswith(suffix):
            return None
        tail = tail[: -len(suffix)]
    if not tail.isdigit():
        return None
    return int(tail)


def _moment_api_payload(database: Database, limit: int) -> list[dict[str, Any]]:
    moments = database.list_moments(limit=max(1, min(limit, 500)))
    for moment in moments:
        moment["download_url"] = f"/api/moments/{moment['id']}/download"
    return moments


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "NASVideo/0.1"
    state: AppState

    def do_HEAD(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_text(_html_shell(), "text/html; charset=utf-8", include_body=False)
            return
        if path.startswith("/static/"):
            self._send_static(path, include_body=False)
            return

        moment_id = _moment_id_from_path(path, "/download")
        if moment_id is not None:
            self._send_moment_download(moment_id, include_body=False)
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, "not found", include_body=False)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send_text(_html_shell(), "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            self._send_static(path)
            return
        if path == "/api/health":
            self._send_json(health_snapshot(self.state.settings, self.state.database, self.state.supervisor))
            return
        if path == "/api/moments":
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", ["100"])[0])
            moments = _moment_api_payload(self.state.database, limit)
            self._send_json({"moments": moments})
            return

        moment_id = _moment_id_from_path(path, "/download")
        if moment_id is not None:
            self._send_moment_download(moment_id)
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        moment_id = _moment_id_from_path(path, "/favorite")
        if moment_id is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return

        moment = self.state.database.get_moment(moment_id)
        if not moment:
            self._send_error_json(HTTPStatus.NOT_FOUND, "moment not found")
            return

        try:
            payload = _read_json(self)
        except json.JSONDecodeError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "invalid JSON body")
            return

        favorited = bool(payload.get("favorited", not moment["favorited"]))
        self.state.database.set_favorite(moment_id, favorited)
        self._send_json({"id": moment_id, "favorited": favorited})

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        moment_id = _moment_id_from_path(path)
        if moment_id is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return

        moment = self.state.database.get_moment(moment_id)
        if not moment:
            self._send_error_json(HTTPStatus.NOT_FOUND, "moment not found")
            return

        for key in ("clip_path", "metadata_path"):
            Path(moment[key]).unlink(missing_ok=True)
        self.state.database.delete_moment_record(moment_id)
        day = Path(moment["clip_path"]).parent.name
        rebuild_day_archive(self.state.settings, self.state.database, day)
        self._send_json({"deleted": True})

    def _send_json(
        self,
        payload: Any,
        status: HTTPStatus = HTTPStatus.OK,
        *,
        include_body: bool = True,
    ) -> None:
        data = _json_bytes(payload)
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not include_body:
            return
        try:
            self.wfile.write(data)
        except Exception as exc:
            if not _is_client_disconnect(exc):
                raise

    def _send_error_json(self, status: HTTPStatus, message: str, *, include_body: bool = True) -> None:
        self._send_json({"detail": message}, status, include_body=include_body)

    def _send_text(self, text: str, content_type: str, *, include_body: bool = True) -> None:
        data = text.encode("utf-8")
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if not include_body:
            return
        try:
            self.wfile.write(data)
        except Exception as exc:
            if not _is_client_disconnect(exc):
                raise

    def _send_static(self, path: str, *, include_body: bool = True) -> None:
        static_dir = Path(__file__).parent / "static"
        name = path.removeprefix("/static/")
        file_path = (static_dir / name).resolve()
        if static_dir.resolve() not in file_path.parents or not file_path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "static file not found")
            return
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self._send_file(file_path, content_type, include_body=include_body)

    def _send_moment_download(self, moment_id: int, *, include_body: bool = True) -> None:
        moment = self.state.database.get_moment(moment_id)
        if not moment:
            self._send_error_json(HTTPStatus.NOT_FOUND, "moment not found", include_body=include_body)
            return
        clip_path = Path(moment["clip_path"])
        if not clip_path.exists():
            self._send_error_json(HTTPStatus.NOT_FOUND, "clip file not found", include_body=include_body)
            return
        self._send_file(
            clip_path,
            "application/octet-stream",
            include_body=include_body,
            download_name=clip_path.name,
        )

    def _send_file(
        self,
        path: Path,
        content_type: str,
        *,
        allow_range: bool = False,
        include_body: bool = True,
        download_name: str | None = None,
    ) -> None:
        size = path.stat().st_size
        start = 0
        end = size - 1
        status = HTTPStatus.OK

        if allow_range:
            range_header = self.headers.get("Range", "")
            if range_header.startswith("bytes="):
                start_text, _, end_text = range_header.removeprefix("bytes=").partition("-")
                try:
                    if start_text:
                        start = int(start_text)
                    if end_text:
                        end = int(end_text)
                    start = max(0, min(start, size - 1))
                    end = max(start, min(end, size - 1))
                    status = HTTPStatus.PARTIAL_CONTENT
                except ValueError:
                    self._send_error_json(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "invalid range")
                    return

        length = end - start + 1
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        if download_name:
            safe_name = download_name.replace('"', "").replace("\r", "").replace("\n", "")
            self.send_header(
                "Content-Disposition", f'attachment; filename="{safe_name}"'
            )
        if allow_range:
            self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        if not include_body:
            return

        try:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except Exception as exc:
            if not _is_client_disconnect(exc):
                raise


def run() -> None:
    state = create_state()
    runtime = WorkerRuntime(state.supervisor)
    runtime.start()

    class BoundHandler(RequestHandler):
        pass

    BoundHandler.state = state
    server = ThreadingHTTPServer((state.settings.app_host, state.settings.app_port), BoundHandler)
    print(f"NAS Video Moments running on http://{state.settings.app_host}:{state.settings.app_port}")

    # Daemon(8) forwards SIGTERM to this Python process. Python's default
    # SIGTERM disposition exits immediately WITHOUT running finally
    # blocks or letting Supervisor.stop() cancel the recorder tasks, so
    # spawned ffmpeg children would be orphaned and keep recording.
    # Install handlers that trigger server.shutdown() so serve_forever
    # returns and the finally block can run runtime.stop() -> Supervisor
    # .stop() -> recorder task -> process.terminate() on ffmpeg.
    def _request_shutdown(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, name="nas-video-shutdown", daemon=True).start()

    previous_sigterm = signal.signal(signal.SIGTERM, _request_shutdown)
    previous_sigint = signal.signal(signal.SIGINT, _request_shutdown)
    try:
        server.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
        server.server_close()
        runtime.stop()
