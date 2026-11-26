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
import math
import os
import re
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .archive import rebuild_day_archive
from .config import Settings
from .database import Database
from .ffmpeg_tools import (
    extract_cropped_frame,
    extract_roi_and_full_frame,
    remux_elementary_stream,
)

_DEFAULT_EXTRACT_CROPPED_FRAME = extract_cropped_frame


@dataclass(frozen=True)
class MultipartPart:
    name: str
    filename: str | None
    content_type: str
    data: bytes


@dataclass(frozen=True)
class MultipartSpan:
    name: str
    filename: str | None
    content_type: str
    start: int
    end: int


def parse_multipart_spans(body: bytes, boundary: str) -> list[MultipartSpan]:
    """零拷贝 multipart/form-data 索引解析器 (仅返回各 part 在 body 中的起止偏移)。

    body 必须符合 RFC 2046 结构:
        --boundary CRLF
        headers CRLF
        CRLF content CRLF
        --boundary--
    解析失败抛 ValueError。只处理 content-disposition / content-type 字段。
    """
    if not boundary:
        raise ValueError("empty boundary")
    delim = b"--" + boundary.encode("utf-8")
    spans: list[MultipartSpan] = []
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
        content_start = header_end + 4
        data_end = max(content_start, content_end - 2)
        name, filename, content_type = _parse_part_headers(headers_raw)
        spans.append(
            MultipartSpan(
                name=name,
                filename=filename,
                content_type=content_type,
                start=content_start,
                end=data_end,
            )
        )
        scanned += 1
        if scanned > 1000:
            raise ValueError("too many multipart parts")
        pos = content_end
    return spans


def parse_multipart(body: bytes, boundary: str) -> list[MultipartPart]:
    """手写 multipart/form-data 解析器 (返回拷贝后的 bytes 载荷)。"""
    return [
        MultipartPart(
            name=span.name,
            filename=span.filename,
            content_type=span.content_type,
            data=body[span.start : span.end],
        )
        for span in parse_multipart_spans(body, boundary)
    ]


def _parse_part_headers(headers_raw: str) -> tuple[str, str | None, str]:
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
    return name, filename, content_type


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
        # 被校验拒绝的片段: 保留下来做训练负样本 (Phase 0)
        self.rejected_dir = self.root / "rejected"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)

    def spool(
        self,
        meta: dict[str, Any],
        video: bytes | bytearray | memoryview,
        meta_filename: str = "meta.json",
        audio: bytes | bytearray | memoryview | None = None,
        audio_name: str | None = None,
    ) -> Path:
        key = ingest_event_key(meta)
        job_dir = self.pending_dir / key
        if job_dir.is_dir():
            return job_dir
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".staging-{key}-", dir=self.pending_dir)
        )
        try:
            meta_path = staging_dir / meta_filename
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            video_name = _video_filename(meta)
            with (staging_dir / video_name).open("wb") as handle:
                handle.write(video)
            if audio is not None and audio_name:
                with (staging_dir / audio_name).open("wb") as handle:
                    handle.write(audio)
            try:
                os.replace(staging_dir, job_dir)
            except OSError:
                if job_dir.is_dir():
                    shutil.rmtree(staging_dir, ignore_errors=True)
                    return job_dir
                raise
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
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
            [
                entry
                for entry in self.pending_dir.iterdir()
                if entry.is_dir() and not entry.name.startswith(".")
            ],
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

    def archive_rejected(self, job_dir: Path, decision: str,
                         meta: dict[str, Any] | None = None) -> Path | None:
        """把被拒片段移到 rejected/, 并写 decision.json 供训练/审计。

        之前拒绝即 rmtree, 导致"纯大人"这类硬负样本一条都留不下, 训练
        儿童/成人分类器时无数据可用。
        """
        import shutil

        if not job_dir.is_dir():
            return None
        try:
            self.rejected_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            target = self.rejected_dir / f"{stamp}_{job_dir.name}"
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(job_dir), str(target))
            info = {
                "decision": decision,
                "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "meta": meta or {},
            }
            (target / "decision.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return target
        except Exception as exc:  # noqa: BLE001 - 归档失败不应影响主流程
            print(f"[INGEST] archive_rejected failed: {exc}", flush=True)
            return None

    def prune_rejected(self, retention_days: int) -> int:
        """删除超过保留期的 rejected/ 子目录, 返回删除数量。"""
        import shutil

        if retention_days <= 0 or not self.rejected_dir.exists():
            return 0
        cutoff = time.time() - retention_days * 86400
        removed = 0
        for entry in self.rejected_dir.iterdir():
            try:
                if entry.is_dir() and entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
        return removed

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


def _box_height(raw_box: Any, *, frame_height: float = 0.0) -> float:
    """单个板端人体框的归一化高度 (全帧坐标系); 无法解析时返回 0.0。"""
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return 0.0
    try:
        x1, y1, x2, y2 = (float(v) for v in raw_box)
    except (TypeError, ValueError):
        return 0.0
    height = y2 - y1
    if height <= 0:
        return 0.0
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        if frame_height <= 0:
            return 0.0
        height /= frame_height
    return min(1.0, height)


def _board_box_height(meta: dict[str, Any]) -> float:
    """板端上传 meta 里的人体框高度 (归一化, 全帧坐标系)。

    板端 box/best_box 语义为 (x1, y1, x2, y2); 若数值 > 1.5 视为像素坐标,
    用 frame_height 归一化。无法解析时返回 0.0 (闸门不生效)。
    """
    return _box_height(
        meta.get("best_box") or meta.get("box"),
        frame_height=float(meta.get("frame_height") or 0),
    )


def _box_to_normalized_roi(
    raw_box: Any,
    *,
    width_scale: float,
    height_scale: float,
    frame_width: float = 0.0,
    frame_height: float = 0.0,
) -> tuple[float, float, float, float] | None:
    """把单个板端框映射为裁剪用的归一化扩展 ROI (left, top, width, height)。

    板端 best_box / box / track_samples 归一化语义为 (x1, y1, x2, y2);
    MQTT 会话 payload 为像素值 (配 frame_width/frame_height)。
    """
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return None
    try:
        raw = [float(value) for value in raw_box]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in raw):
        return None

    if max(raw[0], raw[2]) > 1.5 or max(raw[1], raw[3]) > 1.5:
        # 像素坐标 (x1, y1, x2, y2): 归一化需要帧宽高
        if frame_width <= 0 or frame_height <= 0:
            return None
        x1 = raw[0] / frame_width
        y1 = raw[1] / frame_height
        x2 = raw[2] / frame_width
        y2 = raw[3] / frame_height
    else:
        # 归一化坐标, 板端语义为 (x1, y1, x2, y2)
        x1, y1, x2, y2 = raw

    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        # 兼容旧语义 (x, y, w, h)
        x1, y1, width, height = raw
        if width <= 0 or height <= 0:
            return None

    center_x = x1 + width / 2
    center_y = y1 + height / 2
    expanded_width = min(1.0, width * max(1.0, width_scale))
    expanded_height = min(1.0, height * max(1.0, height_scale))
    roi_left = max(0.0, center_x - expanded_width / 2)
    roi_top = max(0.0, center_y - expanded_height / 2)
    roi_right = min(1.0, center_x + expanded_width / 2)
    roi_bottom = min(1.0, center_y + expanded_height / 2)
    if roi_right <= roi_left or roi_bottom <= roi_top:
        return None
    return roi_left, roi_top, roi_right - roi_left, roi_bottom - roi_top


def _normalized_board_roi(
    meta: dict[str, Any],
    *,
    width_scale: float,
    height_scale: float,
) -> tuple[float, float, float, float] | None:
    """把板端人物框映射为裁剪用的归一化扩展 ROI。

    2026-11-04 修正: 旧实现把归一化框按 (x, y, w, h) 解释, 导致 74% 的
    历史框"越界"(x+w > 1), ROI 中心向右下偏移且尺寸被放大 3~6 倍,
    裁剪框经常框不到人, 表现为 no_person 大量误拒。
    """
    return _box_to_normalized_roi(
        meta.get("best_box") or meta.get("box"),
        width_scale=width_scale,
        height_scale=height_scale,
        frame_width=float(meta.get("frame_width") or 0),
        frame_height=float(meta.get("frame_height") or 0),
    )


@dataclass(frozen=True)
class _VerifyTarget:
    offset_seconds: float
    roi: tuple[float, float, float, float]
    box_height: float


def _spaced_offsets(start: float, end: float, count: int) -> list[float]:
    if count <= 1 or end <= start:
        return [start]
    step = (end - start) / (count - 1)
    return [start + step * index for index in range(count)]


def _anchored_window(
    anchor: float,
    width: float,
    min_off: float,
    max_off: float,
) -> tuple[float, float]:
    """围绕 anchor 构造宽度为 width 的抽帧窗, 越界时整体平移而非单端截断重复。"""
    span = min(max(0.0, width), max(0.0, max_off - min_off))
    lo = anchor - span / 2.0
    hi = anchor + span / 2.0
    if lo < min_off:
        lo = min_off
        hi = min(max_off, lo + span)
    if hi > max_off:
        hi = max_off
        lo = max(min_off, hi - span)
    return lo, hi


def _parse_track_samples(
    meta: dict[str, Any],
    *,
    origin_ts: float,
    end_ts: float,
    width_scale: float,
    height_scale: float,
) -> list[tuple[float, tuple[float, float, float, float], float]]:
    """解析板端上传的稀疏轨迹点 track_samples: [[ts, x1, y1, x2, y2], ...]。"""
    raw_samples = meta.get("track_samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        return []
    frame_width = float(meta.get("frame_width") or 0)
    frame_height = float(meta.get("frame_height") or 0)
    parsed: list[tuple[float, tuple[float, float, float, float], float]] = []
    for item in raw_samples:
        if isinstance(item, (list, tuple)) and len(item) == 5:
            raw_ts, raw_box = item[0], item[1:5]
        elif isinstance(item, dict):
            raw_ts, raw_box = item.get("ts"), item.get("box")
        else:
            continue
        try:
            ts = float(raw_ts)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ts) or ts < origin_ts - 0.5 or ts > end_ts + 0.5:
            continue
        roi = _box_to_normalized_roi(
            raw_box,
            width_scale=width_scale,
            height_scale=height_scale,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if roi is None:
            continue
        height = _box_height(raw_box, frame_height=frame_height)
        parsed.append((ts, roi, height))
    parsed.sort(key=lambda entry: entry[0])
    return parsed


def _plan_verify_targets(
    meta: dict[str, Any],
    *,
    width_scale: float,
    height_scale: float,
    sample_count: int = 5,
    anchor_window_seconds: float = 6.0,
) -> tuple[list[_VerifyTarget], str | None]:
    """规划 probable 校验的 (抽帧偏移, 对应 ROI, 框高) 列表。

    优先级:
      1. 板端携带 eff_start/eff_end 时以实际关键帧起点作为 0s 原点, 消除 GOP 回溯漂移;
      2. 若存在落在切片窗内的 track_samples, 逐帧绑定对应时刻的 ROI (时空对齐);
      3. 否则围绕 best_ts / activity_peak_ts / last_active_ts 在 ±3s 窗口内抽帧,
         避免 context_before=20s 导致的早期空场景误抽; 均不可用时退回均匀抽帧。
    """
    clip_start = float(meta.get("clip_start") or 0.0)
    clip_end = float(meta.get("clip_end") or 0.0)
    eff_start = float(meta.get("eff_start") or 0.0)
    eff_end = float(meta.get("eff_end") or 0.0)
    if (
        math.isfinite(eff_start)
        and math.isfinite(eff_end)
        and eff_start > 0.0
        and eff_end > eff_start
    ):
        origin_ts, end_ts = eff_start, eff_end
    else:
        origin_ts, end_ts = clip_start, clip_end
    duration = max(0.0, end_ts - origin_ts)

    fallback_roi = _normalized_board_roi(
        meta,
        width_scale=width_scale,
        height_scale=height_scale,
    )
    samples = (
        _parse_track_samples(
            meta,
            origin_ts=origin_ts,
            end_ts=end_ts,
            width_scale=width_scale,
            height_scale=height_scale,
        )
        if duration > 0
        else []
    )
    if not samples and fallback_roi is None:
        return [], "invalid_roi_fallback"
    if duration <= 0:
        return [], "invalid_duration_fallback"

    margin = min(0.5, duration * 0.05)
    min_off = margin
    max_off = max(margin, duration - margin)

    if samples:
        if len(samples) >= sample_count:
            targets: list[_VerifyTarget] = []
            for index in range(sample_count):
                sample_idx = round(
                    index * (len(samples) - 1) / max(1, sample_count - 1)
                )
                ts, roi, height = samples[sample_idx]
                off = min(max_off, max(min_off, ts - origin_ts))
                targets.append(_VerifyTarget(off, roi, height))
            return targets, None
        lo_s = samples[0][0] - origin_ts
        hi_s = samples[-1][0] - origin_ts
        if hi_s - lo_s < anchor_window_seconds:
            win_lo, win_hi = _anchored_window(
                (lo_s + hi_s) / 2.0, anchor_window_seconds, min_off, max_off
            )
        else:
            win_lo = min(max_off, max(min_off, lo_s))
            win_hi = min(max_off, max(win_lo, hi_s))
        offsets = _spaced_offsets(win_lo, win_hi, sample_count)
        targets = []
        for off in offsets:
            nearest = min(samples, key=lambda entry: abs((entry[0] - origin_ts) - off))
            targets.append(_VerifyTarget(off, nearest[1], nearest[2]))
        return targets, None

    assert fallback_roi is not None
    fallback_height = _board_box_height(meta)
    anchor_ts: float | None = None
    for key in ("best_ts", "activity_peak_ts", "last_active_ts"):
        raw_val = meta.get(key)
        if raw_val is None:
            continue
        try:
            candidate = float(raw_val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate) and origin_ts <= candidate <= end_ts:
            anchor_ts = candidate
            break

    if anchor_ts is not None:
        win_lo, win_hi = _anchored_window(
            anchor_ts - origin_ts, anchor_window_seconds, min_off, max_off
        )
        offsets = _spaced_offsets(win_lo, win_hi, sample_count)
    else:
        offsets = [duration * fraction for fraction in (0.15, 0.3, 0.5, 0.7, 0.85)]
    return [_VerifyTarget(off, fallback_roi, fallback_height) for off in offsets], None


async def _remux_with_audio_fallback(
    settings: Settings,
    video_path: Path,
    audio_path: Path | None,
    meta: dict[str, Any],
    output_path: Path,
    *,
    key: str,
    creation_time: datetime,
) -> None:
    audio_demux = _audio_demux(meta) if audio_path is not None else None
    try:
        await remux_elementary_stream(
            settings,
            video_path,
            output_path,
            es_format=_es_format(meta.get("codec")),
            audio_path=audio_path,
            audio_format=audio_demux[0] if audio_demux else None,
            audio_rate=audio_demux[1] if audio_demux else None,
            audio_channels=audio_demux[2] if audio_demux else None,
            creation_time=creation_time,
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
            output_path,
            es_format=_es_format(meta.get("codec")),
            creation_time=creation_time,
        )


async def _verify_probable_clip(
    settings: Settings,
    video_path: Path,
    meta: dict[str, Any],
    detector: Any,
    job_key: str = "",
    *,
    mp4_path: Path | None = None,
) -> tuple[bool, str, Any]:
    """对板端上传切片自身抽帧校验是否含儿童证据。

    上传物是裸 Annex-B 基本流 (无容器时间戳), ffmpeg -ss 输入侧寻址
    对其无效 —— 若调用方未提供已重封装的 mp4_path 且源非 .mp4, 先零拷贝
    重封装为 MP4, 再按时空对齐的轨迹 ROI (或 best_ts/activity_peak_ts
    锚定窗口) 抽 5 帧 (单次解码同时产出 ROI 裁切帧与整帧, 回退无需二次
    启动 ffmpeg)。返回 (accepted, decision, verification)。
    """
    targets, plan_fallback = _plan_verify_targets(
        meta,
        width_scale=settings.rv1106_verify_roi_width_scale,
        height_scale=settings.rv1106_verify_roi_height_scale,
    )
    if plan_fallback is not None:
        return True, plan_fallback, None
    heights = [target.box_height for target in targets if target.box_height > 0.0]
    board_box_height = (
        statistics.median(heights) if heights else _board_box_height(meta)
    )
    with tempfile.TemporaryDirectory(prefix="nas-video-clip-verify-") as temp_dir:
        if mp4_path is not None:
            source_mp4 = mp4_path
        elif video_path.suffix.lower() == ".mp4":
            source_mp4 = video_path
        else:
            source_mp4 = Path(temp_dir) / "clip.mp4"
            try:
                await remux_elementary_stream(
                    settings,
                    video_path,
                    source_mp4,
                    es_format=_es_format(meta.get("codec")),
                )
            except Exception as exc:
                print(f"[INGEST] verify remux failed: {exc}", flush=True)
                return True, "remux_failed_fallback", None
        if not source_mp4.exists() or source_mp4.stat().st_size == 0:
            return True, "empty_remux_fallback", None
        frames: list[Path] = []
        full_frames_cached: list[Path] = []
        use_dual_extract = extract_cropped_frame is _DEFAULT_EXTRACT_CROPPED_FRAME
        for index, target in enumerate(targets, start=1):
            roi_path = Path(temp_dir) / f"clip-roi-{index}.jpg"
            full_path = Path(temp_dir) / f"clip-full-{index}.jpg"
            if use_dual_extract:
                await extract_roi_and_full_frame(
                    settings,
                    source_mp4,
                    roi_path,
                    full_path,
                    target.offset_seconds,
                    roi=target.roi,
                    output_width=settings.rv1106_verify_frame_width,
                )
                if full_path.exists():
                    full_frames_cached.append(full_path)
            else:
                await extract_cropped_frame(
                    settings,
                    source_mp4,
                    roi_path,
                    target.offset_seconds,
                    roi=target.roi,
                    output_width=settings.rv1106_verify_frame_width,
                )
            if roi_path.exists():
                frames.append(roi_path)
        if not frames:
            # 切片解不出帧: 无法证伪, 宁多存。
            return True, "no_frames_fallback", None
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
            child_face_enabled=settings.rv1106_verify_child_face_enabled,
            required_child_face_frames=settings.rv1106_verify_child_face_frames,
            board_box_height=board_box_height,
            child_face_max_board_height=(
                settings.rv1106_verify_child_face_max_person_height
            ),
            faceless_max_board_height=(
                settings.rv1106_verify_child_face_max_person_height
            ),
            allow_faceless=getattr(settings, "rv1106_verify_roi_faceless", True),
        )
        if not verification.accepted:
            # 2026-11-04: 板端新模型的 probable 检测框可能指向非人体区域
            # (实测 box 落在右边缘白墙, ROI 裁剪 person_score 恒为 0),
            # 此时 ROI 校验必然 no_person。回退整帧校验, 避免因板端框
            # 不可靠而把真实在场景内的片段全部丢弃。
            full_frames = full_frames_cached
            if not full_frames:
                for index, target in enumerate(targets, start=1):
                    frame_path = Path(temp_dir) / f"clip-full-{index}.jpg"
                    await extract_cropped_frame(
                        settings,
                        source_mp4,
                        frame_path,
                        target.offset_seconds,
                        roi=(0.0, 0.0, 1.0, 1.0),
                        output_width=settings.rv1106_verify_frame_width,
                    )
                    if frame_path.exists():
                        full_frames.append(frame_path)
            if full_frames:
                # 2026-11-05 收紧: faceless_person 只证明"画面里有人", 不证明
                # "画面里是小孩"。整帧回退若照单全收, 会把"只有大人"的片段
                # (爸爸坐沙发 / 奶奶洗衣服) 合法化。默认只接受带儿童证据的
                # 结论 (child_face / daughter_onnx); 确需放宽时置
                # RV1106_VERIFY_FALLBACK_ALLOW_FACELESS=true, 并叠加更高的
                # 人体帧数门槛与人体框高度上限两道硬约束。
                allow_faceless = settings.rv1106_verify_fallback_allow_faceless
                fallback = detector.verify_board_probable_paths(
                    full_frames,
                    required_frames=2,
                    required_person_frames=(
                        settings.rv1106_verify_fallback_person_frames
                        if allow_faceless
                        else settings.rv1106_verify_person_frames
                    ),
                    board_score=float(meta.get("score") or 0.0),
                    board_person_score=float(meta.get("person_score") or 0.0),
                    board_score_threshold=settings.rv1106_verify_board_score,
                    board_person_score_threshold=(
                        settings.rv1106_verify_board_person_score
                    ),
                    max_person_height_limit=(
                        settings.rv1106_verify_fallback_max_person_height
                        if allow_faceless
                        else 0.0
                    ),
                    child_face_enabled=(
                        settings.rv1106_verify_child_face_enabled
                    ),
                    required_child_face_frames=(
                        settings.rv1106_verify_child_face_frames
                    ),
                    board_box_height=board_box_height,
                    child_face_max_board_height=(
                        settings.rv1106_verify_child_face_max_person_height
                    ),
                    faceless_max_board_height=(
                        settings.rv1106_verify_child_face_max_person_height
                        if allow_faceless
                        else 0.0
                    ),
                )
                accept_fallback = fallback.accepted and (
                    allow_faceless or fallback.decision != "faceless_person"
                )
                fb_person_frames = getattr(fallback, "person_frames", 0)
                fb_max_h = float(
                    getattr(fallback, "max_person_height", 0.0) or 0.0
                )
                print(
                    f"[INGEST] {job_key}: roi-miss fallback "
                    f"decision={fallback.decision} "
                    f"person_frames={fb_person_frames} "
                    f"max_h={fb_max_h:.2f} "
                    f"board={float(meta.get('score') or 0.0):.2f}/"
                    f"{float(meta.get('person_score') or 0.0):.2f} -> "
                    f"{'accepted' if accept_fallback else 'rejected'}",
                    flush=True,
                )
                if accept_fallback:
                    verification = fallback
        return verification.accepted, verification.decision, verification


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
    audio_path = spool.job_audio_path(job_dir, meta)
    # 拍摄时刻写入 MP4 creation_time, 供 Nextcloud/Immich 正确按天归档
    clip_creation = datetime.fromtimestamp(
        float(meta["clip_start"]), tz=timezone.utc
    )

    # 儿童证据分 (Q4): confirmed=1.0 (板端人脸命中); probable 经 NAS 安检
    # 按证据映射; 校验器缺失等 fail-open 取中性偏宽 0.5。
    child_evidence = 1.0 if identity == "confirmed" else None
    verify_decision = None
    pre_remux_dir: tempfile.TemporaryDirectory[str] | None = None
    pre_remuxed_mp4: Path | None = None
    try:
        if (
            identity != "confirmed"
            and settings.rv1106_probable_policy == "verify"
        ):
            if probable_verifier is None:
                # 校验器不可用 (OpenCV/模型缺失): 宁多存, 放行并留痕。
                child_evidence = 0.5
                print(
                    f"[INGEST] verify skipped (no verifier) {key}; accepting",
                    flush=True,
                )
            else:
                # 预封装一次并复用于后续发布, 避免 probable 片段被 ffmpeg 重封装两次;
                # 暂存在 spool.root 下, 校验拒绝时不会在 Nextcloud 目录留下空日期文件夹。
                pre_remux_dir = tempfile.TemporaryDirectory(
                    prefix=".nas-video-verify-remux-", dir=spool.root
                )
                candidate_mp4 = Path(pre_remux_dir.name) / "clip.mp4"
                try:
                    await _remux_with_audio_fallback(
                        settings,
                        video_path,
                        audio_path,
                        meta,
                        candidate_mp4,
                        key=key,
                        creation_time=clip_creation,
                    )
                    if candidate_mp4.exists() and candidate_mp4.stat().st_size > 0:
                        pre_remuxed_mp4 = candidate_mp4
                except Exception as exc:
                    print(f"[INGEST] verify remux failed: {exc}", flush=True)
                    pre_remuxed_mp4 = None
                accepted, decision, verification = await _verify_probable_clip(
                    settings,
                    video_path,
                    meta,
                    probable_verifier,
                    job_key=key,
                    mp4_path=pre_remuxed_mp4,
                )
                verify_decision = decision
                if not accepted:
                    print(
                        f"[INGEST] verify reject {key} decision={decision}",
                        flush=True,
                    )
                    if getattr(settings, "rv1106_keep_rejected", False):
                        spool.archive_rejected(job_dir, decision, meta)
                        spool.prune_rejected(
                            int(getattr(settings, "rv1106_rejected_retention_days", 7))
                        )
                    else:
                        spool.remove(job_dir)
                    return None
                if decision in {"child_face", "daughter_onnx"}:
                    child_evidence = 1.0
                elif decision == "faceless_person":
                    child_evidence = 0.6
                print(f"[INGEST] verify accept {key} decision={decision}", flush=True)
        return await _finalize_ingested_clip(
            settings=settings,
            database=database,
            spool=spool,
            job_dir=job_dir,
            meta=meta,
            key=key,
            video_path=video_path,
            audio_path=audio_path,
            clip_creation=clip_creation,
            identity=identity,
            camera_name=camera_name,
            display_clip_start=display_clip_start,
            display_clip_end=display_clip_end,
            day=day,
            child_evidence=child_evidence,
            verify_decision=verify_decision,
            pre_remuxed_mp4=pre_remuxed_mp4,
        )
    finally:
        if pre_remux_dir is not None:
            pre_remux_dir.cleanup()


async def _finalize_ingested_clip(
    *,
    settings: Settings,
    database: Database,
    spool: IngestSpool,
    job_dir: Path,
    meta: dict[str, Any],
    key: str,
    video_path: Path,
    audio_path: Path | None,
    clip_creation: datetime,
    identity: str,
    camera_name: str,
    display_clip_start: datetime,
    display_clip_end: datetime,
    day: str,
    child_evidence: float | None,
    verify_decision: str | None,
    pre_remuxed_mp4: Path | None,
) -> int | None:
    if child_evidence is None:
        # policy=accept: 信任板端判定, 无 NAS 证据时取中性偏宽。
        child_evidence = 0.5

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
    # Q4 三权重: 板端身份 0.6 + 活动量 0.25 + NAS 儿童证据 0.15,
    # 打破 probables 全员同分的排序平局。
    selection_score = min(
        1.0, score * 0.6 + activity_score * 0.25 + child_evidence * 0.15
    )
    confirmed = identity == "confirmed" or face_score >= 0.5
    verified_child_face = verify_decision in {"child_face", "daughter_onnx"}
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
    if verify_decision is not None:
        summary += f" NAS安检通过({verify_decision})。"
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
        if pre_remuxed_mp4 is not None and pre_remuxed_mp4.exists():
            # 复用校验阶段已封装好的 MP4, 跳过第二次 ffmpeg 封装
            shutil.copy2(pre_remuxed_mp4, staged_clip_path)
        else:
            await _remux_with_audio_fallback(
                settings,
                video_path,
                audio_path,
                meta,
                staged_clip_path,
                key=key,
                creation_time=clip_creation,
            )
        os.replace(staged_clip_path, clip_path)

    moment_id: int | None = None
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
            "local_child_confirmed": verified_child_face,
            "local_child_score": child_evidence,
            "source": "board_high_ring",
            "source_stream_role": "board",
            "source_started_at": display_clip_start.isoformat(timespec="seconds"),
            "source_ended_at": display_clip_end.isoformat(timespec="seconds"),
            "clip_start": display_clip_start.isoformat(timespec="seconds"),
            "clip_end": display_clip_end.isoformat(timespec="seconds"),
            "clip_duration_seconds": max(
                1.0, (display_clip_end - display_clip_start).total_seconds()
            ),
            "source_paths": [str(video_path)],
            "model_raw": meta,
            "event_id": None,
            "event_key": key,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

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
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        database.record_ingested_done(key, moment_id)
    except Exception:
        if moment_id is not None:
            database.delete_moment_record(moment_id)
        Path(clip_path).unlink(missing_ok=True)
        Path(metadata_path).unlink(missing_ok=True)
        raise

    # 新片段已安全落盘并登记入库; 现在才移除被替换的最弱片段,
    # 防止写入/入库异常时新旧片段一起丢失。
    for evict_moment in evictions:
        for path_key in ("clip_path", "metadata_path"):
            path = evict_moment.get(path_key)
            if path:
                Path(path).unlink(missing_ok=True)
        database.delete_moment_record(evict_moment["id"])

    rebuild_day_archive(settings, database, day)
    spool.remove(job_dir)
    return moment_id


def _es_format(codec: Any) -> str | None:
    name = str(codec or "").lower()
    if name.startswith("h265") or "hevc" in name:
        return "hevc"
    if name and name.startswith("h264"):
        return "h264"
    return None
