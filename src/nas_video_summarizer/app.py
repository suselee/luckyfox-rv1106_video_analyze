from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .archive import rebuild_day_archive
from .config import Settings, ensure_directories, load_settings
from .database import (
    Database,
    VALID_FEEDBACK_LABELS,
    moment_feedback_key,
)
from .ingest import (
    IngestSpool,
    _plan_verify_targets,
    ingest_event_key,
    parse_multipart_spans,
    validate_meta,
)
from .workers import Supervisor, health_snapshot


def _html_shell() -> str:
    return """
    <!doctype html>
    <html lang="zh-CN">
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
            <div class="tab-bar" role="tablist">
              <button
                type="button"
                class="tab-button active"
                data-tab="moments"
                aria-selected="true"
              >
                Saved Moments
              </button>
              <button
                type="button"
                class="tab-button"
                data-tab="rejected"
                aria-selected="false"
              >
                Rejected Queue
              </button>
            </div>
            <div class="section-controls">
              <label class="filter-toggle">
                <input type="checkbox" id="unlabeled-only" />
                <span>Unlabeled only</span>
              </label>
              <p id="section-hint">
                Automatically saved 4K clips. Review previews, tag daughter / adult-only, or download for VLC.
              </p>
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
    ingest_semaphore: threading.BoundedSemaphore = field(
        default_factory=lambda: threading.BoundedSemaphore(2)
    )
    preview_semaphore: threading.BoundedSemaphore = field(
        default_factory=lambda: threading.BoundedSemaphore(2)
    )


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


# 板端上传允许的音频裸流文件名 (与 edge 端 audio_clip_name 保持一致)。
_AUDIO_FILENAMES = ("clip.g711a", "clip.g711u", "clip.adts")
_REJECTED_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_PREVIEW_CACHE_MAX_FILES = 600


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


def _rejected_name_from_path(path: str, suffix: str) -> str | None:
    prefix = "/api/rejected/"
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    raw_name = path.removeprefix(prefix)[: -len(suffix)]
    name = unquote(raw_name)
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        return None
    if not _REJECTED_NAME_RE.match(name):
        return None
    return name


def _safe_rejected_entry(settings: Settings, name: str) -> Path | None:
    if not name or name in {".", ".."} or not _REJECTED_NAME_RE.match(name):
        return None
    rejected_root = (settings.board_ingest_dir / "rejected").resolve()
    if not rejected_root.is_dir():
        return None
    entry = (rejected_root / name).resolve()
    if entry.parent != rejected_root or not entry.is_dir():
        return None
    return entry


def _rejected_video_path(entry: Path) -> Path | None:
    for filename in ("clip.mp4", "clip.hevc", "clip.h264"):
        candidate = entry / filename
        if candidate.is_file():
            return candidate
    return None


def _rejected_info(entry: Path) -> tuple[str, str, dict[str, Any]]:
    decision = "unknown"
    archived_at = ""
    meta: dict[str, Any] = {}
    decision_path = entry / "decision.json"
    if decision_path.is_file():
        try:
            data = json.loads(decision_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                decision = str(data.get("decision") or "unknown")
                archived_at = str(data.get("archived_at") or "")
                raw_meta = data.get("meta")
                if isinstance(raw_meta, dict):
                    meta = raw_meta
        except (OSError, json.JSONDecodeError):
            pass
    if not meta:
        meta_path = entry / "meta.json"
        if meta_path.is_file():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    meta = loaded
            except (OSError, json.JSONDecodeError):
                pass
    return decision, archived_at, meta


def _read_moment_board_meta(moment: dict[str, Any]) -> dict[str, Any]:
    metadata_path = Path(str(moment.get("metadata_path") or ""))
    if not metadata_path.is_file():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    raw = data.get("model_raw") or data.get("board") or {}
    return raw if isinstance(raw, dict) else {}


def _preview_offset_seconds(meta: dict[str, Any]) -> float:
    if not meta:
        return 0.0
    targets, err = _plan_verify_targets(
        meta,
        width_scale=1.5,
        height_scale=1.5,
        sample_count=1,
    )
    if err is None and targets:
        return max(0.0, float(targets[0].offset_seconds))
    return 0.0


def _preview_cache_key(clip_path: Path) -> str:
    try:
        stat = clip_path.stat()
        token = f"{clip_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        token = str(clip_path)
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]


def _prune_preview_cache(cache_dir: Path, max_files: int = _PREVIEW_CACHE_MAX_FILES) -> None:
    try:
        files = [path for path in cache_dir.iterdir() if path.is_file() and path.suffix == ".jpg"]
        if len(files) <= max_files:
            return
        files.sort(key=lambda path: path.stat().st_mtime)
        for stale in files[: len(files) - max_files]:
            stale.unlink(missing_ok=True)
    except OSError:
        pass


def _extract_preview_jpeg(
    settings: Settings,
    clip_path: Path,
    meta: dict[str, Any],
    output_path: Path,
) -> bool:
    """Extract a single anchored 480px-wide JPEG preview frame via ffmpeg."""
    offset = _preview_offset_seconds(meta)
    suffix = clip_path.suffix.lower()
    with tempfile.TemporaryDirectory(prefix="nas-preview-") as tmp:
        tmp_dir = Path(tmp)
        source = clip_path
        if suffix in {".h264", ".hevc", ".h265"}:
            es_fmt = "hevc" if suffix in {".hevc", ".h265"} else "h264"
            muxed = tmp_dir / "clip.mp4"
            try:
                remux = subprocess.run(
                    [
                        settings.ffmpeg_bin,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-f",
                        es_fmt,
                        "-i",
                        str(clip_path),
                        "-c",
                        "copy",
                        "-y",
                        str(muxed),
                    ],
                    check=False,
                    timeout=20,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (OSError, subprocess.SubprocessError):
                return False
            if remux.returncode == 0 and muxed.is_file() and muxed.stat().st_size > 0:
                source = muxed

        tmp_jpg = tmp_dir / "preview.jpg"
        attempts = [offset, 0.0] if offset > 0.0 else [0.0]
        for seek in attempts:
            cmd = [
                settings.ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{max(0.0, seek):.3f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                "scale=480:-2",
                "-q:v",
                "4",
                "-y",
                str(tmp_jpg),
            ]
            try:
                subprocess.run(
                    cmd,
                    check=False,
                    timeout=20,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (OSError, subprocess.SubprocessError):
                return False
            if tmp_jpg.is_file() and tmp_jpg.stat().st_size > 0:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                staged = output_path.with_name(f".{output_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
                try:
                    shutil.copyfile(tmp_jpg, staged)
                    os.replace(staged, output_path)
                finally:
                    staged.unlink(missing_ok=True)
                return True
    return False


def _ensure_preview_jpeg(
    settings: Settings,
    clip_path: Path,
    meta: dict[str, Any],
    semaphore: threading.BoundedSemaphore,
) -> Path | None:
    if not clip_path.is_file():
        return None
    cache_dir = settings.data_dir / "previews"
    cache_path = cache_dir / f"{_preview_cache_key(clip_path)}.jpg"
    if cache_path.is_file() and cache_path.stat().st_size > 0:
        return cache_path
    if not semaphore.acquire(timeout=15.0):
        return None
    try:
        if cache_path.is_file() and cache_path.stat().st_size > 0:
            return cache_path
        if not _extract_preview_jpeg(settings, clip_path, meta, cache_path):
            return None
        _prune_preview_cache(cache_dir)
        return cache_path
    finally:
        semaphore.release()


def _archive_false_positive_moment(
    settings: Settings,
    moment: dict[str, Any],
    board_meta: dict[str, Any],
) -> Path | None:
    """Preserve a user-deleted adult_only moment in rejected/ as a verified hard negative."""
    if not settings.rv1106_keep_rejected:
        return None
    clip_path = Path(str(moment.get("clip_path") or ""))
    if not clip_path.is_file():
        return None
    rejected_dir = settings.board_ingest_dir / "rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    raw_key = str(moment.get("trigger_key") or f"m{moment['id']}")
    safe_key = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_key).strip("_")[:48] or f"m{moment['id']}"
    target = rejected_dir / f"{stamp}_fp_{safe_key}"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    archived_clip = target / "clip.mp4"
    shutil.copy2(clip_path, archived_clip)
    info = {
        "decision": "human_adult_only",
        "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_moment_id": moment["id"],
        "category": str(moment.get("category") or ""),
        "meta": board_meta,
    }
    (target / "decision.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return archived_clip


def _moment_api_payload(database: Database, limit: int) -> list[dict[str, Any]]:
    moments = database.list_moments(limit=max(1, min(limit, 500)))
    keys = [moment_feedback_key(moment) for moment in moments]
    feedback_map = database.get_feedback_map(keys)
    for moment, item_key in zip(moments, keys):
        moment["download_url"] = f"/api/moments/{moment['id']}/download"
        moment["preview_url"] = f"/api/moments/{moment['id']}/preview.jpg"
        moment["feedback_key"] = item_key
        fb = feedback_map.get(item_key)
        moment["feedback_label"] = fb["label"] if fb else None
    return moments


def _rejected_api_payload(
    settings: Settings, database: Database, limit: int
) -> list[dict[str, Any]]:
    rejected_root = settings.board_ingest_dir / "rejected"
    if not rejected_root.is_dir():
        return []
    max_items = max(1, min(limit, 200))
    entries = sorted(
        (
            entry
            for entry in rejected_root.iterdir()
            if entry.is_dir() and _REJECTED_NAME_RE.match(entry.name)
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    items: list[dict[str, Any]] = []
    for entry in entries:
        if len(items) >= max_items:
            break
        video_path = _rejected_video_path(entry)
        if video_path is None:
            continue
        decision, archived_at, meta = _rejected_info(entry)
        item_key = f"rejected:{entry.name}"
        items.append(
            {
                "name": entry.name,
                "decision": decision,
                "archived_at": archived_at,
                "identity": str(meta.get("identity") or ""),
                "score": float(meta.get("score") or 0.0),
                "person_score": float(meta.get("person_score") or 0.0),
                "camera_id": str(meta.get("camera_id") or ""),
                "preview_url": f"/api/rejected/{entry.name}/preview.jpg",
                "feedback_key": item_key,
                "feedback_label": None,
            }
        )
    feedback_map = database.get_feedback_map([item["feedback_key"] for item in items])
    for item in items:
        fb = feedback_map.get(item["feedback_key"])
        if fb:
            item["feedback_label"] = fb["label"]
    return items


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

        preview_moment_id = _moment_id_from_path(path, "/preview.jpg")
        if preview_moment_id is not None:
            self._send_moment_preview(preview_moment_id, include_body=False)
            return

        rejected_name = _rejected_name_from_path(path, "/preview.jpg")
        if rejected_name is not None:
            self._send_rejected_preview(rejected_name, include_body=False)
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
        if path == "/api/rejected":
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", ["60"])[0])
            rejected = _rejected_api_payload(self.state.settings, self.state.database, limit)
            self._send_json({"rejected": rejected})
            return
        if path == "/api/feedback/summary":
            self._send_json(self.state.database.feedback_summary())
            return

        moment_id = _moment_id_from_path(path, "/download")
        if moment_id is not None:
            self._send_moment_download(moment_id)
            return

        preview_moment_id = _moment_id_from_path(path, "/preview.jpg")
        if preview_moment_id is not None:
            self._send_moment_preview(preview_moment_id)
            return

        rejected_name = _rejected_name_from_path(path, "/preview.jpg")
        if rejected_name is not None:
            self._send_rejected_preview(rejected_name)
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/ingest":
            self._handle_board_ingest()
            return

        feedback_moment_id = _moment_id_from_path(path, "/feedback")
        if feedback_moment_id is not None:
            self._handle_moment_feedback(feedback_moment_id)
            return

        rejected_name = _rejected_name_from_path(path, "/feedback")
        if rejected_name is not None:
            self._handle_rejected_feedback(rejected_name)
            return

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

    def _handle_moment_feedback(self, moment_id: int) -> None:
        moment = self.state.database.get_moment(moment_id)
        if not moment:
            self._send_error_json(HTTPStatus.NOT_FOUND, "moment not found")
            return
        try:
            payload = _read_json(self)
        except json.JSONDecodeError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "invalid JSON body")
            return
        label = str(payload.get("label") or "").strip().lower()
        if label not in VALID_FEEDBACK_LABELS:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                f"label must be one of: {', '.join(VALID_FEEDBACK_LABELS)}",
            )
            return
        delete_clip = bool(payload.get("delete_clip", False))
        board_meta = _read_moment_board_meta(moment)
        item_key = moment_feedback_key(moment)
        day = Path(str(moment["clip_path"])).parent.name
        stored_clip_path = str(moment["clip_path"])
        archived_path: Path | None = None

        if delete_clip:
            if label == "adult_only":
                archived_path = _archive_false_positive_moment(
                    self.state.settings, moment, board_meta
                )
                if archived_path is not None:
                    stored_clip_path = str(archived_path)
            self.state.database.upsert_clip_feedback(
                source="moment",
                item_key=item_key,
                label=label,
                moment_id=moment_id,
                deleted_clip=True,
                clip_path=stored_clip_path,
                meta=board_meta,
                day=day,
                category=str(moment.get("category") or ""),
            )
            for key in ("clip_path", "metadata_path"):
                Path(moment[key]).unlink(missing_ok=True)
            self.state.database.delete_moment_record(moment_id)
            rebuild_day_archive(self.state.settings, self.state.database, day)
        else:
            self.state.database.upsert_clip_feedback(
                source="moment",
                item_key=item_key,
                label=label,
                moment_id=moment_id,
                deleted_clip=False,
                clip_path=stored_clip_path,
                meta=board_meta,
                day=day,
                category=str(moment.get("category") or ""),
            )

        self._send_json(
            {
                "id": moment_id,
                "item_key": item_key,
                "label": label,
                "deleted": delete_clip,
                "archived_path": str(archived_path) if archived_path else None,
                "summary": self.state.database.feedback_summary(),
            }
        )

    def _handle_rejected_feedback(self, name: str) -> None:
        entry = _safe_rejected_entry(self.state.settings, name)
        if entry is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "rejected clip not found")
            return
        video_path = _rejected_video_path(entry)
        if video_path is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "rejected video not found")
            return
        try:
            payload = _read_json(self)
        except json.JSONDecodeError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "invalid JSON body")
            return
        label = str(payload.get("label") or "").strip().lower()
        if label not in VALID_FEEDBACK_LABELS:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                f"label must be one of: {', '.join(VALID_FEEDBACK_LABELS)}",
            )
            return

        decision, archived_at, meta = _rejected_info(entry)
        item_key = f"rejected:{entry.name}"
        day = archived_at[:10] if len(archived_at) >= 10 else ""
        self.state.database.upsert_clip_feedback(
            source="rejected",
            item_key=item_key,
            label=label,
            moment_id=None,
            deleted_clip=False,
            clip_path=str(video_path),
            meta=meta,
            day=day,
            category=decision,
        )
        self._send_json(
            {
                "name": entry.name,
                "item_key": item_key,
                "label": label,
                "summary": self.state.database.feedback_summary(),
            }
        )

    def _handle_board_ingest(self) -> None:
        """接收 RV1106 板端 multipart 上传 (meta JSON + 4K 基本流片段)。"""
        settings = self.state.settings
        if not settings.board_ingest_enabled:
            self._send_error_json(HTTPStatus.NOT_FOUND, "board ingest disabled")
            return

        content_type = self.headers.get("Content-Type", "")
        boundary = None
        for token in content_type.split(";"):
            key, _, value = token.strip().partition("=")
            if key.lower() == "boundary":
                boundary = value.strip().strip('"')
                break
        if not boundary:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "missing multipart boundary")
            return

        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
            return
        if length <= 0:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "missing Content-Length")
            return
        if length > settings.board_ingest_max_bytes:
            self._send_error_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"upload exceeds {settings.board_ingest_max_bytes} bytes",
            )
            return

        semaphore = self.state.ingest_semaphore
        if not semaphore.acquire(timeout=5.0):
            self._send_error_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "board ingest busy, retry later",
            )
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "short body")
                return

            try:
                spans = parse_multipart_spans(raw, boundary)
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            meta_span = next((span for span in spans if span.name == "meta"), None)
            video_span = next(
                (span for span in spans if span.name == "video" and span.filename),
                None,
            )
            if meta_span is None or video_span is None:
                self._send_error_json(
                    HTTPStatus.BAD_REQUEST, "multipart must contain meta and video parts"
                )
                return
            audio_span = next(
                (
                    span
                    for span in spans
                    if span.name == "audio"
                    and span.filename in _AUDIO_FILENAMES
                ),
                None,
            )

            try:
                meta_text = raw[meta_span.start : meta_span.end].decode("utf-8")
                meta, meta_error = validate_meta(json.loads(meta_text))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_error_json(HTTPStatus.BAD_REQUEST, "meta is not valid JSON")
                return
            if meta is None:
                self._send_error_json(HTTPStatus.BAD_REQUEST, meta_error or "invalid meta")
                return

            event_key = ingest_event_key(meta)
            existing = self.state.database.ingested_moment_id(event_key)
            spool = IngestSpool(settings)
            if existing is not None or (spool.pending_dir / event_key).exists():
                self._send_json(
                    {"accepted": True, "duplicate": True, "event_key": event_key}
                )
                return

            raw_view = memoryview(raw)
            try:
                spool.spool(
                    meta,
                    raw_view[video_span.start : video_span.end],
                    audio=(
                        raw_view[audio_span.start : audio_span.end]
                        if audio_span is not None
                        else None
                    ),
                    audio_name=audio_span.filename if audio_span is not None else None,
                )
            finally:
                raw_view.release()
            self._send_json(
                {"accepted": True, "duplicate": False, "event_key": event_key},
                status=HTTPStatus.ACCEPTED,
            )
        finally:
            semaphore.release()

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

    def _send_moment_preview(self, moment_id: int, *, include_body: bool = True) -> None:
        moment = self.state.database.get_moment(moment_id)
        if not moment:
            self._send_error_json(HTTPStatus.NOT_FOUND, "moment not found", include_body=include_body)
            return
        clip_path = Path(moment["clip_path"])
        if not clip_path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "clip file not found", include_body=include_body)
            return
        board_meta = _read_moment_board_meta(moment)
        preview_path = _ensure_preview_jpeg(
            self.state.settings, clip_path, board_meta, self.state.preview_semaphore
        )
        if preview_path is None:
            self._send_error_json(
                HTTPStatus.NOT_FOUND, "preview unavailable", include_body=include_body
            )
            return
        self._send_file(
            preview_path,
            "image/jpeg",
            include_body=include_body,
            cache_control="public, max-age=86400",
        )

    def _send_rejected_preview(self, name: str, *, include_body: bool = True) -> None:
        entry = _safe_rejected_entry(self.state.settings, name)
        if entry is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "rejected clip not found", include_body=include_body)
            return
        video_path = _rejected_video_path(entry)
        if video_path is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "rejected video not found", include_body=include_body)
            return
        _, _, meta = _rejected_info(entry)
        preview_path = _ensure_preview_jpeg(
            self.state.settings, video_path, meta, self.state.preview_semaphore
        )
        if preview_path is None:
            self._send_error_json(
                HTTPStatus.NOT_FOUND, "preview unavailable", include_body=include_body
            )
            return
        self._send_file(
            preview_path,
            "image/jpeg",
            include_body=include_body,
            cache_control="public, max-age=86400",
        )

    def _send_file(
        self,
        path: Path,
        content_type: str,
        *,
        allow_range: bool = False,
        include_body: bool = True,
        download_name: str | None = None,
        cache_control: str | None = None,
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
        if cache_control:
            self.send_header("Cache-Control", cache_control)
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
    # Install handlers that trigger server.shutdown() so serve_forever()
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
