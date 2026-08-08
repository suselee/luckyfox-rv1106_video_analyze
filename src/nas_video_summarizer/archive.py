from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from .config import Settings
from .database import Database


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def rebuild_day_archive(
    settings: Settings,
    database: Database,
    day: str,
    *,
    ready: bool = False,
    error_count: int = 0,
) -> Path:
    """Rebuild NAS-owned day files from SQLite, never touching desktop output."""
    day_dir = settings.output_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)
    moments = database.moments_on_day(day)
    revision = datetime.now().astimezone().isoformat(timespec="microseconds")
    clips = []
    summary = [f"# Family Moments Index - {day}", ""]
    for moment in moments:
        clip = Path(moment["clip_path"])
        metadata = Path(moment["metadata_path"])
        clip_size = clip.stat().st_size if clip.exists() else 0
        metadata_size = metadata.stat().st_size if metadata.exists() else 0
        clips.append(
            {
                "event_id": int(moment["id"]),
                "clip": clip.name,
                "metadata": metadata.name,
                "clip_size_bytes": clip_size,
                "metadata_size_bytes": metadata_size,
                "clip_started_at": moment.get("clip_started_at"),
                "clip_ended_at": moment.get("clip_ended_at"),
                "analysis_backend": moment.get("analysis_backend", "vlm"),
                "category": moment.get("category", "semantic"),
                "confidence": float(moment["confidence"]),
                "selection_score": float(moment.get("selection_score", moment["confidence"])),
            }
        )
        started = str(moment.get("clip_started_at") or moment["source_started_at"])
        time_text = started[11:19] if len(started) >= 19 else started
        summary.extend(
            [
                f"## {time_text} - {moment['title']}",
                "",
                f"- Clip: [{clip.name}]({clip.name})",
                f"- Category: {moment.get('category', 'semantic')}",
                f"- Confidence: {float(moment['confidence']):.2f}",
                f"- Selection score: {float(moment.get('selection_score', moment['confidence'])):.2f}",
                "",
                str(moment["summary"]),
                "",
            ]
        )
    manifest = {
        "schema_version": 1,
        "owner": "nas",
        "date": day,
        "revision": revision,
        "clip_count": len(clips),
        "clips": clips,
    }
    _atomic_text(day_dir / "summary.md", "\n".join(summary).rstrip() + "\n")
    _atomic_json(day_dir / "manifest.json", manifest)
    ready_path = day_dir / "_READY.json"
    if ready:
        _atomic_json(
            ready_path,
            {
                "schema_version": 1,
                "owner": "nas",
                "date": day,
                "status": "complete_with_errors" if error_count else "complete",
                "manifest_revision": revision,
                "clip_count": len(clips),
                "error_count": error_count,
                "completed_at": revision,
            },
        )
    else:
        ready_path.unlink(missing_ok=True)
    return day_dir / "manifest.json"
