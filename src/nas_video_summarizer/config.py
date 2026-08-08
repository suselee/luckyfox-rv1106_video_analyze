from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit


DEFAULT_ANALYSIS_PROMPT = (
    "Find short, precious everyday moments of my ~1.5-year-old daughter at home. "
    "There are three family members: my daughter (a toddler girl), her mom "
    "(a young adult woman, NOT the grandma), and her grandma (an older woman) - "
    "distinct people, do not confuse mom with grandma. "
    "KEEP a clip when my daughter is clearly visible AND something real is happening: "
    "she is interacting with family (playing, talking, cuddling, reading, eating "
    "together, being washed/dressed with engagement, reacting to someone) or doing "
    "something on her own (playing with a toy, cruising/walking, babbling, exploring, "
    "laughing, dancing, looking at a book). Warm family interaction IS a highlight - "
    "high energy is NOT required. A calm but genuine moment with her is worth saving. "
    "EXCLUDE only these truly low-value scenes: "
    "an empty or static room with NO child visible (do not invent a child off-screen); "
    "child sleeping or drowsy/being rocked to sleep (passive, no engagement); "
    "child being fed with zero interaction (bottle propped, no eye contact/play); "
    "child sitting totally blank, idle, staring at nothing; "
    "child watching a screen (TV/iPad/phone) as passive consumption; "
    "blurry or black/empty frames; "
    "an adult doing chores while the child is merely present in the background and "
    "NOT engaging with them. "
    "If you cannot see my daughter clearly in a frame, that frame is keep=false. "
    "Skip outdoor views and pets-only scenes. "
    "When describing the scene, name people correctly: daughter, mom, or grandma. "
    "Keep clips concise. Aim to capture a balanced, joyful record of her day - "
    "around 10-20 good clips per day is the goal, so keep genuine interactions and "
    "her own activities, and only drop the empty/low-value ones. "
    "Return JSON only with keep, title, summary, tags, confidence, start_offset_seconds, and end_offset_seconds."
)


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _optional_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return float(value)


def _path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else None


def _rv1106_probable_policy() -> str:
    value = os.getenv("RV1106_PROBABLE_POLICY", "").strip().lower()
    if value:
        if value not in {"verify", "reject", "accept"}:
            raise ValueError(
                "RV1106_PROBABLE_POLICY must be verify, reject, or accept"
            )
        return value
    # Backward compatibility for deployments that have not adopted the new
    # policy setting yet.
    return "accept" if _bool("RV1106_ACCEPT_PROBABLE", True) else "reject"


def _parse_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
        if value:
            value = value.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
    return value


def load_env_file(env_file: str | Path) -> None:
    path = Path(env_file)
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name or name in os.environ:
            continue
        os.environ[name] = _parse_env_value(value)


def with_rtsp_credentials(url: str, username: str, password: str) -> str:
    if not url or not username:
        return url

    parts = urlsplit(url)
    if parts.username or parts.password:
        return url

    userinfo = quote(username, safe="")
    if password:
        userinfo = f"{userinfo}:{quote(password, safe='')}"
    netloc = f"{userinfo}@{parts.netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


@dataclass(frozen=True)
class Settings:
    app_host: str
    app_port: int
    workers_enabled: bool
    data_dir: Path
    buffer_dir: Path
    output_dir: Path
    database_path: Path
    camera_name: str
    camera_time_offset_seconds: int
    rtsp_username: str
    rtsp_password: str
    rtsp_low_url: str
    rtsp_high_url: str
    rtsp_transport: str
    ffmpeg_bin: str
    ffprobe_bin: str
    ffmpeg_hwaccel: str
    segment_seconds: int
    segment_at_clocktime: bool
    segment_stable_seconds: int
    stream_alignment_tolerance_seconds: float
    stream_alignment_sample_count: int
    retention_hours: int
    analysis_enabled: bool
    analysis_backend: str
    analysis_delay_seconds: int
    analysis_interval_seconds: int
    analysis_max_attempts: int
    analysis_cooldown_seconds: int
    analysis_stream_role: str
    analysis_window_start: str
    analysis_window_end: str
    record_window_start: str
    record_window_end: str
    analysis_image_mode: str
    analysis_frame_width: int
    contact_sheet_columns: int
    contact_sheet_padding: int
    sample_frame_count: int
    sample_every_seconds: int
    sample_mode: str
    motion_threshold: float
    moment_keep_threshold: float
    max_moments_per_day: int
    max_moments_per_period: int
    moment_period_boundaries: str
    moment_cooldown_seconds: int
    context_before_seconds: int
    context_after_seconds: int
    max_moment_seconds: int
    clip_video_codec: str
    clip_audio_codec: str
    clip_video_preset: str
    clip_video_crf: int
    llama_base_url: str
    llama_api_key: str
    llama_model: str
    llama_timeout_seconds: int
    llama_analysis_temperature: float | None
    llama_verification_temperature: float | None
    llama_timeout_fallback: bool
    llama_circuit_breaker_failures: int
    llama_circuit_breaker_seconds: int
    verification_frame_width: int
    analysis_prompt: str
    person_filter_enabled: bool
    person_filter_backend: str
    person_filter_model_url: str
    person_filter_model_dir: Path
    person_filter_threshold: float
    person_filter_face_threshold: float
    person_filter_adult_threshold: float
    person_filter_child_threshold: float
    person_filter_sample_count: int
    daughter_detector_mode: str
    daughter_detector_model_path: Path | None
    daughter_detector_input_size: int
    daughter_detector_threshold: float
    daughter_age_check_every: int
    daughter_body_fallback_enabled: bool
    daughter_body_height_ratio: float
    daughter_body_area_ratio: float
    daughter_scan_fps: float
    daughter_event_min_hits: int
    daughter_event_max_gap_seconds: float
    daughter_event_min_seconds: float
    moment_category_targets: str
    day_ready_grace_seconds: int
    mqtt_enabled: bool
    mqtt_host: str
    mqtt_port: int
    mqtt_username: str
    mqtt_password: str
    mqtt_daughter_topic: str
    mqtt_status_topic: str
    mqtt_client_id: str
    mqtt_keepalive_seconds: int
    rv1106_session_timeout_seconds: float
    rv1106_save_wait_seconds: float
    rv1106_probable_policy: str
    rv1106_accept_probable: bool
    rv1106_verify_roi_width_scale: float
    rv1106_verify_roi_height_scale: float
    rv1106_verify_frame_width: int
    rv1106_verify_person_frames: int
    rv1106_verify_board_score: float
    rv1106_verify_board_person_score: float
    rv1106_verify_faceless_score_multiplier: float

    @property
    def low_buffer_dir(self) -> Path:
        return self.buffer_dir / self.camera_name / "low"

    @property
    def high_buffer_dir(self) -> Path:
        return self.buffer_dir / self.camera_name / "high"

    @property
    def rtsp_low_url_for_ffmpeg(self) -> str:
        return with_rtsp_credentials(self.rtsp_low_url, self.rtsp_username, self.rtsp_password)

    @property
    def rtsp_high_url_for_ffmpeg(self) -> str:
        return with_rtsp_credentials(self.rtsp_high_url, self.rtsp_username, self.rtsp_password)


def load_settings(env_file: str | Path = ".env") -> Settings:
    load_env_file(env_file)
    data_dir = _path("DATA_DIR", "./var")
    rv1106_probable_policy = _rv1106_probable_policy()

    return Settings(
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=_int("APP_PORT", 8000),
        workers_enabled=_bool("WORKERS_ENABLED", True),
        data_dir=data_dir,
        buffer_dir=_path("BUFFER_DIR", "./var/buffer"),
        output_dir=_path("NEXTCLOUD_OUTPUT_DIR", "./var/nextcloud_moments"),
        database_path=_path("DATABASE_PATH", "./var/app.sqlite3"),
        camera_name=os.getenv("CAMERA_NAME", "home-camera"),
        camera_time_offset_seconds=_int("CAMERA_TIME_OFFSET_SECONDS", 0),
        rtsp_username=os.getenv("RTSP_USERNAME", ""),
        rtsp_password=os.getenv("RTSP_PASSWORD", ""),
        rtsp_low_url=os.getenv("RTSP_LOW_URL", ""),
        rtsp_high_url=os.getenv("RTSP_HIGH_URL", ""),
        rtsp_transport=os.getenv("RTSP_TRANSPORT", "tcp"),
        ffmpeg_bin=os.getenv("FFMPEG_BIN", "ffmpeg"),
        ffprobe_bin=os.getenv("FFPROBE_BIN", "ffprobe"),
        ffmpeg_hwaccel=os.getenv("FFMPEG_HWACCEL", "").strip().lower(),
        segment_seconds=_int("SEGMENT_SECONDS", 120),
        segment_at_clocktime=_bool("SEGMENT_AT_CLOCKTIME", True),
        segment_stable_seconds=_int("SEGMENT_STABLE_SECONDS", 8),
        stream_alignment_tolerance_seconds=_float(
            "STREAM_ALIGNMENT_TOLERANCE_SECONDS", 2.0
        ),
        stream_alignment_sample_count=_int("STREAM_ALIGNMENT_SAMPLE_COUNT", 5),
        retention_hours=_int("RETENTION_HOURS", 168),
        analysis_enabled=_bool("ANALYSIS_ENABLED", True),
        analysis_backend=os.getenv("ANALYSIS_BACKEND", "vlm").strip().lower(),
        analysis_delay_seconds=_int("ANALYSIS_DELAY_SECONDS", 300),
        analysis_interval_seconds=_int("ANALYSIS_INTERVAL_SECONDS", 15),
        analysis_max_attempts=_int("ANALYSIS_MAX_ATTEMPTS", 3),
        analysis_cooldown_seconds=_int("ANALYSIS_COOLDOWN_SECONDS", 5),
        analysis_stream_role=os.getenv("ANALYSIS_STREAM_ROLE", "low").strip().lower(),
        analysis_window_start=os.getenv("ANALYSIS_WINDOW_START", "").strip(),
        analysis_window_end=os.getenv("ANALYSIS_WINDOW_END", "").strip(),
        record_window_start=os.getenv("RECORD_WINDOW_START", "").strip(),
        record_window_end=os.getenv("RECORD_WINDOW_END", "").strip(),
        analysis_image_mode=os.getenv("ANALYSIS_IMAGE_MODE", "frames").strip().lower(),
        analysis_frame_width=_int("ANALYSIS_FRAME_WIDTH", 384),
        contact_sheet_columns=_int("CONTACT_SHEET_COLUMNS", 2),
        contact_sheet_padding=_int("CONTACT_SHEET_PADDING", 8),
        sample_frame_count=_int("SAMPLE_FRAME_COUNT", 4),
        sample_every_seconds=_int("SAMPLE_EVERY_SECONDS", 30),
        sample_mode=os.getenv("SAMPLE_MODE", "even").strip().lower(),
        motion_threshold=_float("MOTION_THRESHOLD", 0.02),
        moment_keep_threshold=_float("MOMENT_KEEP_THRESHOLD", 0.5),
        max_moments_per_day=_int("MAX_MOMENTS_PER_DAY", 0),
        max_moments_per_period=_int("MAX_MOMENTS_PER_PERIOD", 0),
        moment_period_boundaries=os.getenv(
            "MOMENT_PERIOD_BOUNDARIES", "07:00,12:00,17:00,21:00"
        ).strip(),
        moment_cooldown_seconds=_int("MOMENT_COOLDOWN_SECONDS", 0),
        context_before_seconds=_int("CONTEXT_BEFORE_SECONDS", 5),
        context_after_seconds=_int("CONTEXT_AFTER_SECONDS", 10),
        max_moment_seconds=_int("MAX_MOMENT_SECONDS", 45),
        clip_video_codec=os.getenv("CLIP_VIDEO_CODEC", "copy").strip().lower(),
        clip_audio_codec=os.getenv("CLIP_AUDIO_CODEC", "copy").strip().lower(),
        clip_video_preset=os.getenv("CLIP_VIDEO_PRESET", "veryfast").strip().lower(),
        clip_video_crf=_int("CLIP_VIDEO_CRF", 23),
        llama_base_url=os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080/v1"),
        llama_api_key=os.getenv("LLAMA_API_KEY", ""),
        llama_model=os.getenv("LLAMA_MODEL", "Qwen3-VL-2B-Instruct"),
        llama_timeout_seconds=_int("LLAMA_TIMEOUT_SECONDS", 180),
        llama_analysis_temperature=_optional_float(
            "LLAMA_ANALYSIS_TEMPERATURE"
        ),
        llama_verification_temperature=_optional_float(
            "LLAMA_VERIFICATION_TEMPERATURE"
        ),
        llama_timeout_fallback=_bool("LLAMA_TIMEOUT_FALLBACK", True),
        llama_circuit_breaker_failures=_int("LLAMA_CIRCUIT_BREAKER_FAILURES", 3),
        llama_circuit_breaker_seconds=_int("LLAMA_CIRCUIT_BREAKER_SECONDS", 300),
        verification_frame_width=_int("VERIFICATION_FRAME_WIDTH", 512),
        analysis_prompt=os.getenv("ANALYSIS_PROMPT", DEFAULT_ANALYSIS_PROMPT),
        person_filter_enabled=_bool("PERSON_FILTER_ENABLED", False),
        person_filter_backend=os.getenv("PERSON_FILTER_BACKEND", "yolov11n").strip().lower(),
        person_filter_model_url=os.getenv("PERSON_FILTER_MODEL_URL", "").strip(),
        person_filter_model_dir=_path(
            "PERSON_FILTER_MODEL_DIR", str(data_dir / "person_filter_models")
        ),
        person_filter_threshold=_float("PERSON_FILTER_THRESHOLD", 0.3),
        person_filter_face_threshold=_float("PERSON_FILTER_FACE_THRESHOLD", 0.7),
        person_filter_adult_threshold=_float("PERSON_FILTER_ADULT_THRESHOLD", 0.9),
        person_filter_child_threshold=_float("PERSON_FILTER_CHILD_THRESHOLD", 0.6),
        person_filter_sample_count=_int("PERSON_FILTER_SAMPLE_COUNT", 12),
        daughter_detector_mode=os.getenv(
            "DAUGHTER_DETECTOR_MODE", "heuristic"
        ).strip().lower(),
        daughter_detector_model_path=_optional_path("DAUGHTER_DETECTOR_MODEL_PATH"),
        daughter_detector_input_size=_int("DAUGHTER_DETECTOR_INPUT_SIZE", 416),
        daughter_detector_threshold=_float("DAUGHTER_DETECTOR_THRESHOLD", 0.45),
        daughter_age_check_every=_int("DAUGHTER_AGE_CHECK_EVERY", 3),
        daughter_body_fallback_enabled=_bool(
            "DAUGHTER_BODY_FALLBACK_ENABLED", True
        ),
        daughter_body_height_ratio=_float("DAUGHTER_BODY_HEIGHT_RATIO", 0.72),
        daughter_body_area_ratio=_float("DAUGHTER_BODY_AREA_RATIO", 0.50),
        daughter_scan_fps=_float("DAUGHTER_SCAN_FPS", 0.5),
        daughter_event_min_hits=_int("DAUGHTER_EVENT_MIN_HITS", 2),
        daughter_event_max_gap_seconds=_float(
            "DAUGHTER_EVENT_MAX_GAP_SECONDS", 6.0
        ),
        daughter_event_min_seconds=_float("DAUGHTER_EVENT_MIN_SECONDS", 4.0),
        moment_category_targets=os.getenv(
            "MOMENT_CATEGORY_TARGETS", "active:3,multi_person:3,quiet:2"
        ).strip(),
        day_ready_grace_seconds=_int("DAY_READY_GRACE_SECONDS", 120),
        mqtt_enabled=_bool("MQTT_ENABLED", False),
        mqtt_host=os.getenv("MQTT_HOST", "127.0.0.1").strip(),
        mqtt_port=_int("MQTT_PORT", 1883),
        mqtt_username=os.getenv("MQTT_USERNAME", ""),
        mqtt_password=os.getenv("MQTT_PASSWORD", ""),
        mqtt_daughter_topic=os.getenv(
            "MQTT_DAUGHTER_TOPIC", "homecam/daughter/hit"
        ).strip(),
        mqtt_status_topic=os.getenv(
            "MQTT_STATUS_TOPIC", "homecam/daughter/status"
        ).strip(),
        mqtt_client_id=os.getenv("MQTT_CLIENT_ID", "nas-video").strip(),
        mqtt_keepalive_seconds=_int("MQTT_KEEPALIVE_SECONDS", 30),
        rv1106_session_timeout_seconds=_float(
            "RV1106_SESSION_TIMEOUT_SECONDS", 20.0
        ),
        rv1106_save_wait_seconds=_float("RV1106_SAVE_WAIT_SECONDS", 180.0),
        rv1106_probable_policy=rv1106_probable_policy,
        rv1106_accept_probable=rv1106_probable_policy != "reject",
        rv1106_verify_roi_width_scale=_float(
            "RV1106_VERIFY_ROI_WIDTH_SCALE", 3.0
        ),
        rv1106_verify_roi_height_scale=_float(
            "RV1106_VERIFY_ROI_HEIGHT_SCALE", 2.0
        ),
        rv1106_verify_frame_width=_int("RV1106_VERIFY_FRAME_WIDTH", 960),
        rv1106_verify_person_frames=_int("RV1106_VERIFY_PERSON_FRAMES", 3),
        rv1106_verify_board_score=_float("RV1106_VERIFY_BOARD_SCORE", 0.70),
        rv1106_verify_board_person_score=_float(
            "RV1106_VERIFY_BOARD_PERSON_SCORE", 0.70
        ),
        rv1106_verify_faceless_score_multiplier=_float(
            "RV1106_VERIFY_FACELESS_SCORE_MULTIPLIER", 0.75
        ),
    )


def ensure_directories(settings: Settings) -> None:
    for path in (
        settings.data_dir,
        settings.buffer_dir,
        settings.output_dir,
        settings.database_path.parent,
        settings.low_buffer_dir,
        settings.high_buffer_dir,
        settings.person_filter_model_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
