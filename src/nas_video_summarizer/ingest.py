"""NAS 侧接收板端上传的 4K 片段 (board-primary 模式)。

RV1106 板从内存环形缓冲切片段后, 以 multipart/form-data POST 到
/api/ingest (字段: "meta" = JSON, "video" = 裸 Annex-B 基本流,
文件名 clip.h264 / clip.hevc)。本模块负责:

  * 手写 multipart/form-data 解析 (仅标准库);
  * 板端 meta 字段校验与归一化;
  * 上传落盘排队 (board_ingest_dir, 幂等 event_key 去重);
  * 基本流 -c copy 重封装为 MP4, 发布到 NEXTCLOUD_OUTPUT_DIR 布局
    (由 worker 调用 save_ingested_clip)。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .archive import rebuild_day_archive
from .config import Settings
from .database import Database
from .ffmpeg_tools import extract_cropped_frame, remux_elementary_stream


@dataclass(frozen=True)
class MultipartPart:
    name: str
    filename: str | None
    content_type: str
    data: bytes


def parse_multipart(body: bytes, boundary: str) -> list[MultipartPart]:
    """手写 multipart/form-data 解析器。

    body 必须符合 RFC 2046 结构:
        --boundary CRLF
        headers CRLF
        CRLF content CRLF
        --boundary--
    返回所有 part; 解析失败抛 ValueError。只处理 content-disposition 字段。
    """
    if not boundary:
        raise ValueError("empty boundary")
    delim = b"--" + boundary.encode("utf-8")
    parts: list[MultipartPart] = []
    pos = 0
    scanned = 0
    while True:
        start = body.find(delim, pos)
        if start < 0:
            raise ValueError("multipart boundary not found")
        line_end = body.find(b"\r\n", start)
        next_seg = body[start + len(delim) : line_end if line_end >= 0 else len(body)]
        if next_seg.startswith(b"--"):
            break  # 终止边界
        header_end = body.find(b"\r\n\r\n", line_end if line_end >= 0 else start)
        if header_end < 0:
            raise ValueError("multipart headers unterminated")
        headers_raw = body[line_end + 2 : header_end].decode("utf-8", errors="replace")
        content_end = body.find(delim, header_end)
        if content_end < 0:
            raise ValueError("multipart part unterminated")
        # 与边界前的 \r\n 分隔符精确切掉 2 字节 (视频二进制可能以任意字节结尾)
        content = body[header_end + 4 : content_end - 2]
        parts.append(_part_from_headers(headers_raw, content))
        scanned += 1
        if scanned > 1000:
            raise ValueError("too many multipart parts")
        pos = content_end
    return parts


def _part_from_headers(headers_raw: str, content: bytes) -> MultipartPart:
    name = ""
    filename: str | None = None
    content_type = ""
    for line in headers_raw.split("\r\n"):
        if not line:
            continue
        if line.lower().startswith("content-disposition:"):
            for token in line.split(";")[1:]:
                key, _, value = token.strip().partition("=")
                value = value.strip().strip('"')
                if key == "name":
                    name = value
                elif key == "filename":
                    filename = value or None
        elif line.lower().startswith("content-type:"):
            content_type = line.split(":", 1)[1].strip()
    return MultipartPart(name=name, filename=filename, content_type=content_type, data=content)


# 板端 meta 必须携带的字段。
REQUIRED_META_KEYS = (
    "session_id",
    "event",
    "identity",
    "clip_start",
    "clip_end",
)


def validate_meta(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """校验/归一化板 meta。返回 (normalized, None) 或 (None, 错误信息)。"""
    if not isinstance(raw, dict):
        return None, "meta must be a JSON object"
    missing = [key for key in REQUIRED_META_KEYS if key not in raw]
    if missing:
        return None, f"meta missing required fields: {', '.join(missing)}"
    try:
        clip_start = float(raw["clip_start"])
        clip_end = float(raw["clip_end"])
    except (TypeError, ValueError):
        return None, "meta clip_start/clip_end must be numbers"
    if not (clip_end > clip_start > 0):
        return None, "meta clip_start/clip_end out of range"
    if clip_end - clip_start > 24 * 3600:
        return None, "meta clip window too long"
    identity = str(raw.get("identity") or "confirmed").strip().lower()
    event = str(raw.get("event") or "confirmed").strip().lower()
    normalized = dict(raw)
    normalized["identity"] = identity
    normalized["event"] = event
    normalized["clip_start"] = clip_start
    normalized["clip_end"] = clip_end
    return normalized, None


def ingest_event_key(meta: dict[str, Any]) -> str:
    """板端片段幂等键: 同一 session 同一切片重传得到同一键。"""
    canonical = {
        "camera_id": str(meta.get("camera_id") or ""),
        "session_id": str(meta.get("session_id") or ""),
        "session_start": float(meta.get("session_start") or meta.get("clip_start") or 0),
        "best_ts": float(meta.get("best_ts") or 0),
        "clip_start": float(meta["clip_start"]),
        "clip_end": float(meta["clip_end"]),
        "identity": meta.get("identity"),
    }
    digest = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest[:24]


class IngestSpool:
    """文件系统落盘队列: board_ingest_dir/pending/<key>/, 失败移入 failed/。"""

    def __init__(self, settings: Settings):
        self.root = settings.board_ingest_dir
        self.pending_dir = self.root / "pending"
        self.failed_dir = self.root / "failed"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)

    def spool(
        self,
        meta: dict[str, Any],
        video: bytes,
        meta_filename: str = "meta.json",
        audio: bytes | None = None,
        audio_name: str | None = None,
    ) -> Path:
        key = ingest_event_key(meta)
        job_dir = self.pending_dir / key
        job_dir.mkdir(parents=True, exist_ok=True)
        meta_path = job_dir / meta_filename
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        video_name = _video_filename(meta)
        (job_dir / video_name).write_bytes(video)
        if audio is not None and audio_name:
            (job_dir / audio_name).write_bytes(audio)
        return job_dir

    @staticmethod
    def job_meta(job_dir: Path) -> dict[str, Any]:
        candidates = sorted(job_dir.glob("meta*.json"))
        if not candidates:
            raise ValueError(f"spool job lacks meta.json: {job_dir}")
        return json.loads(candidates[0].read_text(encoding="utf-8"))

    @staticmethod
    def job_video_path(job_dir: Path, meta: dict[str, Any]) -> Path:
        video_name = _video_filename(meta)
        video_path = job_dir / video_name
        if not video_path.exists():
            raise ValueError(f"spool job lacks video file: {job_dir}")
        return video_path

    @staticmethod
    def job_audio_path(job_dir: Path, meta: dict[str, Any]) -> Path | None:
        """板端音频裸流 (可选)。无音频时返回 None。"""
        audio_name = _audio_filename(meta)
        if not audio_name:
            return None
        audio_path = job_dir / audio_name
        return audio_path if audio_path.exists() else None

    def pending_jobs(self) -> list[Path]:
        return sorted(
            [entry for entry in self.pending_dir.iterdir() if entry.is_dir()],
            key=lambda path: path.name,
        )

    def remove(self, job_dir: Path) -> None:
        import shutil

        if job_dir.is_dir():
            shutil.rmtree(job_dir, ignore_errors=True)

    def fail(self, job_dir: Path) -> None:
        import shutil

        if not job_dir.is_dir():
            return
        target = self.failed_dir / job_dir.name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(job_dir), str(target))

    def pending_count(self) -> int:
        return len(self.pending_jobs())


def _video_filename(meta: dict[str, Any]) -> str:
    codec = str(meta.get("codec") or "H264").upper()
    return "clip.hevc" if codec.startswith("H265") else "clip.h264"


def _audio_filename(meta: dict[str, Any]) -> str | None:
    """按板端 meta 的 audio_codec 推导音频裸流文件名。"""
    codec = str(meta.get("audio_codec") or "").upper()
    if codec == "PCMA":
        return "clip.g711a"
    if codec == "PCMU":
        return "clip.g711u"
    if codec == "AAC":
        return "clip.adts"
    return None


def _audio_demux(meta: dict[str, Any]) -> tuple[str, int, int] | None:
    """音频裸流对应的 ffmpeg demuxer + 采样率 + 声道。"""
    codec = str(meta.get("audio_codec") or "").upper()
    if codec == "PCMA":
        return "alaw", int(meta.get("audio_rate") or 8000), int(meta.get("audio_channels") or 1)
    if codec == "PCMU":
        return "mulaw", int(meta.get("audio_rate") or 8000), int(meta.get("audio_channels") or 1)
    if codec == "AAC":
        return "aac", 0, 0   # ffmpeg 的 ADTS demuxer 名为 "aac"
    return None


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).lower()).strip("-")
    return slug[:64] or "family-moment"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate unique path for {path}")


def _moment_period(
    started_at: datetime, boundaries_value: str
) -> tuple[str, datetime, datetime] | None:
    """阶段划分 (与 workers._moment_period 同构): morning/afternoon/evening。

    返回 (label, start_dt, end_dt); 时间落在夜间 (21:00-07:00) 或无有效边界时
    返回 None (该片段不受阶段上限约束, 仅受每日上限约束)。
    """
    parts = [part.strip() for part in boundaries_value.split(",")]
    if len(parts) != 4:
        return None
    try:
        minutes = []
        for part in parts:
            hour_text, minute_text = part.split(":", 1)
            hour, minute = int(hour_text), int(minute_text)
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                return None
            minutes.append(hour * 60 + minute)
    except (TypeError, ValueError):
        return None
    if minutes != sorted(set(minutes)):
        return None

    current = started_at.hour * 60 + started_at.minute
    labels = ("morning", "afternoon", "evening")
    for index, label in enumerate(labels):
        if minutes[index] <= current < minutes[index + 1]:
            start = started_at.replace(
                hour=minutes[index] // 60,
                minute=minutes[index] % 60,
                second=0,
                microsecond=0,
            )
            end = started_at.replace(
                hour=minutes[index + 1] // 60,
                minute=minutes[index + 1] % 60,
                second=0,
                microsecond=0,
            )
            return label, start, end
    return None


def _normalized_board_roi(
    meta: dict[str, Any],
    *,
    width_scale: float,
    height_scale: float,
) -> tuple[float, float, float, float] | None:
    """把板端人物框映射为裁剪用的归一化扩展 ROI。

    兼容两种输入: HTTP 上传 meta 的 best_box 为归一化值 (0~1);
    MQTT 会话 payload 为像素值 (配 frame_width/frame_height)。
    """
    raw_box = meta.get("best_box") or meta.get("box")
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return None
    try:
        raw = [float(value) for value in raw_box]
    except (TypeError, ValueError):
        return None
    if any(value <= 0 for value in raw):
        return None
    if max(raw[0], raw[2]) > 1.5 or max(raw[1], raw[3]) > 1.5:
        # 像素盒: 归一化需要帧宽高
        frame_width = float(meta.get("frame_width") or 0)
        frame_height = float(meta.get("frame_height") or 0)
        if frame_width <= 0 or frame_height <= 0:
            return None
        x = raw[0] / frame_width
        y = raw[1] / frame_height
        width = raw[2] / frame_width
        height = raw[3] / frame_height
    else:
        x, y, width, height = raw
    if width <= 0 or height <= 0:
        return None

    center_x = x + width / 2
    center_y = y + height / 2
    expanded_width = min(1.0, width * max(1.0, width_scale))
    expanded_height = min(1.0, height * max(1.0, height_scale))
    roi_left = max(0.0, center_x - expanded_width / 2)
    roi_top = max(0.0, center_y - expanded_height / 2)
    roi_right = min(1.0, center_x + expanded_width / 2)
    roi_bottom = min(1.0, center_y + expanded_height / 2)
    if roi_right <= roi_left or roi_bottom <= roi_top:
        return None
    return roi_left, roi_top, roi_right - roi_left, roi_bottom - roi_top


async def _verify_probable_clip(
    settings: Settings,
    video_path: Path,
    meta: dict[str, Any],
    detector: Any,
) -> tuple[bool, str]:
    """对板端上传切片自身抽帧校验是否含儿童证据。

    上传物是裸 Annex-B 基本流 (无容器时间戳), ffmpeg -ss 输入侧寻址
    对其无效 —— 先零拷贝重封装为 MP4 (c:v copy, 容器级操作) 再按轨迹
    ROI 均匀抽 5 帧, 交给 DaughterDetector.verify_board_probable_paths:
    儿童人脸/ONNX 证据接受; 连续成人脸或无人在场拒绝; 无脸但人物稳定
    且板端分数强则按高召回放行。返回 (accepted, decision)。
    """
    roi = _normalized_board_roi(
        meta,
        width_scale=settings.rv1106_verify_roi_width_scale,
        height_scale=settings.rv1106_verify_roi_height_scale,
    )
    if roi is None:
        return True, "invalid_roi_fallback"
    clip_start = float(meta.get("clip_start") or 0.0)
    clip_end = float(meta.get("clip_end") or 0.0)
    duration = max(0.0, clip_end - clip_start)
    if duration <= 0:
        return True, "invalid_duration_fallback"
    with tempfile.TemporaryDirectory(prefix="nas-video-clip-verify-") as temp_dir:
        mp4_path = Path(temp_dir) / "clip.mp4"
        try:
            await remux_elementary_stream(settings, video_path, mp4_path)
        except Exception as exc:
            print(f"[INGEST] verify remux failed: {exc}", flush=True)
            return True, "remux_failed_fallback"
        if not mp4_path.exists() or mp4_path.stat().st_size == 0:
            return True, "empty_remux_fallback"
        offsets = [duration * fraction for fraction in (0.15, 0.3, 0.5, 0.7, 0.85)]
        frames = []
        for index, offset in enumerate(offsets, start=1):
            frame_path = Path(temp_dir) / f"clip-roi-{index}.jpg"
            await extract_cropped_frame(
                settings,
                mp4_path,
                frame_path,
                offset,
                roi=roi,
                output_width=settings.rv1106_verify_frame_width,
            )
            if frame_path.exists():
                frames.append(frame_path)
        if not frames:
            # 切片解不出帧: 无法证伪, 宁多存。
            return True, "no_frames_fallback"
        verification = detector.verify_board_probable_paths(
            frames,
            required_frames=2,
            required_person_frames=settings.rv1106_verify_person_frames,
            board_score=float(meta.get("score") or 0.0),
            board_person_score=float(meta.get("person_score") or 0.0),
            board_score_threshold=settings.rv1106_verify_board_score,
            board_person_score_threshold=(
                settings.rv1106_verify_board_person_score
            ),
        )
        return verification.accepted, verification.decision


async def save_ingested_clip(
    settings: Settings,
    database: Database,
    spool: IngestSpool,
    job_dir: Path,
    *,
    probable_verifier: Any = None,
) -> int | None:
    """把排队的板端上传转成最终 moment, 返回 moment id (或 None 表示跳过)。

    失败抛异常, 由 worker 记错误次数; 成功时同时落 DB 并清掉 spool 目录。
    probable_verifier: 可选 DaughterDetector 实例; RV1106_PROBABLE_POLICY=verify
    时用它对上传切片自身抽帧校验 (board-primary 模式没有 NAS 录像段可查)。
    """
    meta = spool.job_meta(job_dir)
    key = ingest_event_key(meta)
    existing_moment = database.ingested_moment_id(key)
    if existing_moment is not None:
        spool.remove(job_dir)
        return existing_moment

    video_path = spool.job_video_path(job_dir, meta)
    identity = str(meta.get("identity") or "confirmed").lower()
    camera_name = str(meta.get("camera_id") or settings.camera_name)

    clip_start_dt = datetime.fromtimestamp(float(meta["clip_start"])).astimezone()
    clip_end_dt = datetime.fromtimestamp(float(meta["clip_end"])).astimezone()
    display_offset = timedelta(seconds=settings.camera_time_offset_seconds)
    display_clip_start = clip_start_dt + display_offset
    display_clip_end = clip_end_dt + display_offset
    day = display_clip_start.strftime("%Y-%m-%d")

    # 产品规则: probable 是否保存由 RV1106_PROBABLE_POLICY 决定
    # (accept=信任板端识别直接保存, verify=抽帧校验后决定, reject=不保存)。
    if identity != "confirmed" and settings.rv1106_probable_policy == "reject":
        spool.remove(job_dir)
        return None
    if (
        identity != "confirmed"
        and settings.rv1106_probable_policy == "verify"
    ):
        if probable_verifier is None:
            # 校验器不可用 (OpenCV/模型缺失): 宁多存, 放行并留痕。
            print(
                f"[INGEST] verify skipped (no verifier) {key}; accepting",
                flush=True,
            )
        else:
            accepted, decision = await _verify_probable_clip(
                settings, video_path, meta, probable_verifier
            )
            if not accepted:
                print(
                    f"[INGEST] verify reject {key} decision={decision}",
                    flush=True,
                )
                spool.remove(job_dir)
                return None
            print(f"[INGEST] verify accept {key} decision={decision}", flush=True)

    # 频率闸门 (纯 DB, 零 AI): 距最近一次已保存 moment 不足冷却窗口
    # (RV1106_INGEST_COOLDOWN_SECONDS) 时丢弃, 与板端切片冷却互为兜底。
    cooldown = getattr(settings, "rv1106_ingest_cooldown_seconds", 0) or 0
    if cooldown > 0:
        previous = database.nearest_moment_before(
            display_clip_start.isoformat(timespec="milliseconds")
        )
        if previous is not None:
            prev_key = str(previous.get("clip_started_at") or previous.get("created_at") or "")
            try:
                prev_dt = datetime.fromisoformat(prev_key)
                gap = (display_clip_start - prev_dt).total_seconds()
            except (TypeError, ValueError):
                prev_dt = None
                gap = None
            if gap is not None and 0 <= gap < cooldown:
                print(
                    f"[INGEST] cooldown skip {key} gap={gap:.0f}s "
                    f"clip_start={display_clip_start.isoformat()} prev={prev_key}",
                    flush=True,
                )
                spool.remove(job_dir)
                return None

    score = float(meta.get("score") or 0.5)
    activity_score = float(meta.get("activity_score") or 0.0)
    face_score = float(meta.get("face_score") or 0.0)
    selection_score = min(1.0, score * 0.75 + activity_score * 0.25)
    confirmed = identity == "confirmed" or face_score >= 0.5
    title = (
        "Daughter confirmed by RV1106"
        if confirmed
        else "Probable daughter activity"
    )
    summary = (
        "RV1106 板端人脸与轨迹融合确认女儿出现在画面中, 4K 片段由板载环形缓冲直接上传。"
        if confirmed
        else "RV1106 板端检测到持续稳定的儿童活动轨迹, 作为高召回候选由板端上传 4K 片段。"
    )
    tags = ["daughter", "rv1106", identity, "board_high_ring"]
    category = f"rv1106_{identity}"

    title_slug = _slugify(title)
    day_dir = settings.output_dir / day
    clip_path = _unique_path(day_dir / f"{display_clip_start.strftime('%H%M%S')}_{title_slug}.mp4")
    metadata_path = clip_path.with_suffix(".json")

    # 阶段上限 + 每日上限: 超限时只有比该范围内最弱片段更强才允许保存,
    # 并移除最弱片段 (评分 = selection_score, 与 NAS 分析管线共用同一池)。
    evictions: list[dict[str, Any]] = []
    if settings.max_moments_per_period:
        period = _moment_period(display_clip_start, settings.moment_period_boundaries)
        if period is not None:
            start_iso = period[1].isoformat(timespec="milliseconds")
            end_iso = period[2].isoformat(timespec="milliseconds")
            count_period = database.count_moments_between(start_iso, end_iso)
            if count_period >= settings.max_moments_per_period:
                weakest = database.weakest_moment_between(start_iso, end_iso)
                weakest_score = (
                    float(weakest.get("selection_score") or weakest.get("confidence") or 0.0)
                    if weakest
                    else 0.0
                )
                if weakest and selection_score > weakest_score:
                    evictions.append(weakest)
                else:
                    spool.remove(job_dir)
                    return None

    if settings.max_moments_per_day:
        count_day = database.count_moments_on_day(day)
        if count_day >= settings.max_moments_per_day:
            weakest = database.weakest_moment_on_day(day)
            weakest_score = (
                float(weakest.get("selection_score") or weakest.get("confidence") or 0.0)
                if weakest
                else 0.0
            )
            if weakest and selection_score > weakest_score:
                if not any(item["id"] == weakest["id"] for item in evictions):
                    evictions.append(weakest)
            else:
                spool.remove(job_dir)
                return None

    day_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".nas-video-staged-", dir=day_dir
    ) as staging_dir:
        staged_clip_path = Path(staging_dir) / clip_path.name
        audio_path = spool.job_audio_path(job_dir, meta)
        audio_demux = _audio_demux(meta) if audio_path is not None else None
        try:
            await remux_elementary_stream(
                settings,
                video_path,
                staged_clip_path,
                es_format=_es_format(meta.get("codec")),
                audio_path=audio_path,
                audio_format=audio_demux[0] if audio_demux else None,
                audio_rate=audio_demux[1] if audio_demux else None,
                audio_channels=audio_demux[2] if audio_demux else None,
            )
        except RuntimeError:
            # 音频裸流损坏/编码误判时降级为纯视频重封装, 不阻塞片段保存。
            if audio_path is None:
                raise
            print(
                f"[INGEST] remux with audio failed for {key}; retrying video-only",
                flush=True,
            )
            await remux_elementary_stream(
                settings,
                video_path,
                staged_clip_path,
                es_format=_es_format(meta.get("codec")),
            )
        os.replace(staged_clip_path, clip_path)

    # 新片段已安全落盘; 现在移除被替换的最弱片段 (文件 + DB 记录)。
    for evict_moment in evictions:
        for path_key in ("clip_path", "metadata_path"):
            path = evict_moment.get(path_key)
            if path:
                Path(path).unlink(missing_ok=True)
        database.delete_moment_record(evict_moment["id"])

    try:
            metadata = {
                "schema_version": 3,
                "owner": "nas",
                "camera_name": camera_name,
                "analysis_backend": "rv1106_edge",
                "category": category,
                "title": title,
                "summary": summary,
                "tags": tags,
                "confidence": min(1.0, max(0.0, score)),
                "selection_score": selection_score,
                "keep_consistency_repaired": False,
                "local_child_confirmed": confirmed,
                "local_child_score": score,
                "source": "board_high_ring",
                "source_stream_role": "board",
                "source_started_at": display_clip_start.isoformat(timespec="seconds"),
                "source_ended_at": display_clip_end.isoformat(timespec="seconds"),
                "clip_start": display_clip_start.isoformat(timespec="seconds"),
                "clip_end": display_clip_end.isoformat(timespec="seconds"),
                "clip_duration_seconds": max(1.0, (display_clip_end - display_clip_start).total_seconds()),
                "source_paths": [str(video_path)],
                "model_raw": meta,
                "event_id": None,
                "event_key": key,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

            moment_id = database.create_moment(
                camera_name=camera_name,
                title=title,
                summary=summary,
                tags=tags,
                confidence=score,
                source_low_segment_id=None,
                source_started_at=display_clip_start.isoformat(timespec="seconds"),
                source_ended_at=display_clip_end.isoformat(timespec="seconds"),
                clip_path=clip_path,
                metadata_path=metadata_path,
                analysis_backend="rv1106_edge",
                category=category,
                selection_score=selection_score,
                clip_started_at=display_clip_start.isoformat(timespec="seconds"),
                clip_ended_at=display_clip_end.isoformat(timespec="seconds"),
                trigger_key=key,
                source_segment_id=None,
                source_stream_role="board",
            )
            metadata["event_id"] = moment_id
            metadata["daily_summary_path"] = str(day_dir / "summary.md")
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

            rebuild_day_archive(settings, database, day)
            database.record_ingested_done(key, moment_id)
            spool.remove(job_dir)
    except Exception:
        Path(clip_path).unlink(missing_ok=True)
        Path(metadata_path).unlink(missing_ok=True)
        raise
    return moment_id


def _es_format(codec: Any) -> str | None:
    name = str(codec or "").lower()
    if name.startswith("h265") or "hevc" in name:
        return "hevc"
    if name and name.startswith("h264"):
        return "h264"
    return None