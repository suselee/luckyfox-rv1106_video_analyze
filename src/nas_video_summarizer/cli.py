from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .config import Settings, ensure_directories, load_settings
from .database import Database
from .daughter_detector import DaughterDetector
from .person_filter import PersonFilter


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _redact_url(value: str) -> str:
    if not value:
        return ""
    try:
        parts = urlsplit(value)
    except ValueError:
        return "<invalid-url>"
    if not parts.username and not parts.password:
        return value

    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    netloc = f"<credentials>@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _path_writable(path: Path) -> bool:
    if path.exists():
        return path.is_dir() and os.access(path, os.W_OK)
    return path.parent.exists() and os.access(path.parent, os.W_OK)


def _rtsp_credentials_check(settings: Settings) -> tuple[bool, str]:
    if settings.rtsp_username:
        return True, "configured"
    if urlsplit(settings.rtsp_low_url).username or urlsplit(settings.rtsp_high_url).username:
        return True, "embedded in URL"
    return False, "not configured; OK only for cameras without auth"


def _person_filter_check(settings: Settings) -> Check:
    if not settings.person_filter_enabled or settings.analysis_backend == "rv1106":
        detail = "inactive in rv1106 mode" if settings.analysis_backend == "rv1106" else "disabled"
        return Check("person filter", True, detail, required=False)

    try:
        detector = PersonFilter(
            threshold=settings.person_filter_threshold,
            backend=settings.person_filter_backend,
            model_url=settings.person_filter_model_url,
            model_dir=settings.person_filter_model_dir,
            face_threshold=settings.person_filter_face_threshold,
            adult_threshold=settings.person_filter_adult_threshold,
        )
        model_path = detector.prepare()
        import cv2

        return Check(
            "person filter",
            True,
            f"{settings.person_filter_backend}, OpenCV {cv2.__version__}, {model_path}",
        )
    except ImportError as exc:
        return Check(
            "person filter",
            False,
            f"{exc}; on FreeBSD recreate .venv with uv venv --clear "
            "--system-site-packages --python /usr/local/bin/python3.11",
        )
    except Exception as exc:
        return Check(
            "person filter",
            False,
            f"{settings.person_filter_backend} initialization failed: {exc}",
        )


def _daughter_detector_check(settings: Settings) -> Check:
    continuous_detector_active = (
        settings.analysis_enabled
        and settings.analysis_backend == "daughter_detector"
    )
    probable_verifier_active = settings.rv1106_probable_policy == "verify"
    if not continuous_detector_active and not probable_verifier_active:
        return Check("daughter detector", True, "inactive", required=False)
    try:
        detail = DaughterDetector(settings).prepare()
        return Check(
            "daughter detector",
            True,
            f"{settings.daughter_detector_mode}: {detail}",
        )
    except Exception as exc:
        return Check("daughter detector", False, str(exc))


def build_checks(settings: Settings) -> list[Check]:
    credentials_ok, credentials_detail = _rtsp_credentials_check(settings)
    return [
        Check(
            "analysis backend",
            settings.analysis_backend in {"vlm", "daughter_detector", "rv1106"},
            settings.analysis_backend,
        ),
        Check(
            "ffmpeg",
            shutil.which(settings.ffmpeg_bin) is not None,
            settings.ffmpeg_bin,
        ),
        Check(
            "ffprobe",
            shutil.which(settings.ffprobe_bin) is not None,
            settings.ffprobe_bin,
        ),
        Check(
            "ffmpeg hwaccel",
            not settings.ffmpeg_hwaccel or settings.ffmpeg_hwaccel in ("vaapi", "auto", "none"),
            settings.ffmpeg_hwaccel or "disabled (software decode)",
            required=False,
        ),
        Check(
            "low RTSP stream",
            bool(settings.rtsp_low_url),
            _redact_url(settings.rtsp_low_url_for_ffmpeg) or "RTSP_LOW_URL is empty",
            required=(
                settings.analysis_enabled
                and settings.analysis_stream_role == "low"
            ),
        ),
        Check(
            "4K RTSP stream",
            bool(settings.rtsp_high_url),
            _redact_url(settings.rtsp_high_url_for_ffmpeg) or "RTSP_HIGH_URL is empty",
        ),
        Check(
            "RTSP credentials",
            credentials_ok,
            credentials_detail,
            required=False,
        ),
        Check(
            "llama.cpp endpoint",
            settings.analysis_backend != "vlm" or bool(settings.llama_base_url),
            "inactive" if settings.analysis_backend != "vlm" else settings.llama_base_url or "LLAMA_BASE_URL is empty",
            required=settings.analysis_backend == "vlm",
        ),
        Check(
            "output directory",
            _path_writable(settings.output_dir),
            str(settings.output_dir),
        ),
        Check(
            "buffer directory",
            _path_writable(settings.buffer_dir),
            str(settings.buffer_dir),
        ),
        Check(
            "database parent",
            settings.database_path.parent.exists(),
            str(settings.database_path.parent),
        ),
        Check(
            "MQTT configuration",
            not settings.mqtt_enabled
            or (
                bool(settings.mqtt_host)
                and 0 < settings.mqtt_port < 65536
                and bool(settings.mqtt_daughter_topic)
                and bool(settings.mqtt_status_topic)
                and (not settings.mqtt_password or bool(settings.mqtt_username))
            ),
            (
                "disabled"
                if not settings.mqtt_enabled
                else (
                    f"{settings.mqtt_host}:{settings.mqtt_port} "
                    f"{settings.mqtt_daughter_topic}, {settings.mqtt_status_topic}"
                )
            ),
            required=settings.mqtt_enabled,
        ),
        _person_filter_check(settings),
        _daughter_detector_check(settings),
    ]


def check_main() -> None:
    settings = load_settings()
    ensure_directories(settings)
    Database(settings.database_path).migrate()

    checks = build_checks(settings)
    print("NAS Video preflight")
    print(f"camera: {settings.camera_name}")
    print(f"analysis backend: {settings.analysis_backend}")
    print(
        f"model: {settings.llama_model}"
        if settings.analysis_backend == "vlm"
        else f"detector: {settings.daughter_detector_mode}"
    )
    print(
        "analysis: "
        f"{settings.analysis_image_mode}, "
        f"{settings.sample_frame_count} frames, "
        f"{settings.analysis_frame_width}px per frame"
    )
    print()

    failed_required = False
    for check in checks:
        marker = "OK" if check.ok else "FAIL" if check.required else "WARN"
        required = "required" if check.required else "optional"
        print(f"[{marker}] {check.name} ({required}): {check.detail}")
        if check.required and not check.ok:
            failed_required = True

    if failed_required:
        print("\nFix the failed required checks before enabling real recording.")
        raise SystemExit(1)

    print("\nPreflight passed.")
    raise SystemExit(0)


if __name__ == "__main__":
    check_main()
