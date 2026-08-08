from __future__ import annotations

import base64
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from .config import Settings
from .analysis import ClipCandidate

AnalysisResult = ClipCandidate


_SEMANTIC_REPAIR_MIN_CONFIDENCE = 0.75
_CHILD_EVIDENCE_TERMS = (
    "daughter",
    "young girl",
    "young child",
    "toddler",
    "child",
)
_ACTIVITY_EVIDENCE_TERMS = (
    "play",
    "activity",
    "interact",
    "engaged",
    "explor",
    "walk",
    "read",
    "laugh",
    "babbl",
    "cuddl",
    "danc",
    "draw",
)
_EXCLUSION_EVIDENCE_TERMS = (
    "empty room",
    "no child",
    "no young girl",
    "no girl",
    "cannot see",
    "can't see",
    "adults-only",
    "adult only",
    "only adults",
    "sleeping",
    "drowsy",
    "screen time",
    "blurry",
    "black frame",
    "not clearly visible",
    "not engaged",
    "no activity",
    "merely idle",
    "sitting idle",
    "passive",
)


@dataclass(frozen=True)
class DaughterVerification:
    visible: bool
    confidence: float
    description: str
    repaired: bool
    raw_text: str


def _snap_highlight_offsets(
    start: int,
    end: int,
    frame_offsets: list[float],
    duration_seconds: int,
) -> tuple[int, int]:
    offsets = sorted(
        {
            max(0.0, min(float(offset), float(duration_seconds)))
            for offset in frame_offsets
        }
    )
    if not offsets:
        return start, end

    snapped_start = min(offsets, key=lambda offset: abs(offset - start))
    snapped_end = min(offsets, key=lambda offset: abs(offset - end))
    if snapped_end <= snapped_start:
        later = [offset for offset in offsets if offset > snapped_start]
        snapped_end = later[0] if later else min(duration_seconds, snapped_start + 10)
    return int(round(snapped_start)), max(
        int(round(snapped_start)) + 1,
        int(round(snapped_end)),
    )


def _parse_daughter_verification(data: dict[str, Any], raw_text: str) -> DaughterVerification:
    has_daughter = _coerce_bool(data.get("has_daughter"))
    confidence = max(0.0, min(_coerce_float(data.get("confidence"), 0.0), 1.0))
    description = str(data.get("description") or "").strip()
    evidence = description.lower()
    repaired = (
        not has_daughter
        and confidence >= _SEMANTIC_REPAIR_MIN_CONFIDENCE
        and any(term in evidence for term in _CHILD_EVIDENCE_TERMS)
        and not any(term in evidence for term in _EXCLUSION_EVIDENCE_TERMS)
    )
    return DaughterVerification(
        visible=(has_daughter and confidence >= 0.5) or repaired,
        confidence=confidence,
        description=description,
        repaired=repaired,
        raw_text=raw_text,
    )


def _image_content(path: Path) -> dict[str, Any]:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{data}",
        },
    }


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("model response did not contain a JSON object")
    return json.loads(match.group(0))


def _coerce_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "save", "keep"}
    return bool(value)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _estimated_offsets(duration_seconds: int, count: int) -> list[float]:
    if duration_seconds <= 0 or count <= 0:
        return []
    return [i * duration_seconds / (count + 1) for i in range(1, count + 1)]


def _format_seconds(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _post_json(endpoint: str, headers: dict[str, str], payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama.cpp request failed: HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"llama.cpp request failed: {exc.reason}") from exc


class LlamaAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def analyze(
        self,
        *,
        video_path: Path,
        image_paths: list[Path],
        duration_seconds: int,
        frame_offsets_seconds: list[float] | None = None,
        image_mode: str | None = None,
    ) -> AnalysisResult:
        if not image_paths:
            raise ValueError("no sampled images available for analysis")

        endpoint = self.settings.llama_base_url.rstrip("/") + "/chat/completions"
        effective_image_mode = image_mode or self.settings.analysis_image_mode
        image_note = (
            "The image is a contact sheet. Read cells chronologically from left to right, "
            "then top to bottom."
            if effective_image_mode == "contact_sheet"
            else "The images are sampled video frames in chronological order."
        )
        annotation_offsets = (
            frame_offsets_seconds
            if frame_offsets_seconds
            else _estimated_offsets(duration_seconds, len(image_paths))
        )
        annotation_label = "cell" if effective_image_mode == "contact_sheet" else "frame"
        frame_offsets = ", ".join(
            f"{annotation_label} #{index}: ~{_format_seconds(offset)}s"
            for index, offset in enumerate(annotation_offsets, start=1)
        )
        instructions = (
            f"{self.settings.analysis_prompt}\n\n"
            f"{image_note}\n\n"
            "Return exactly one JSON object with these fields:\n"
            "- keep: boolean\n"
            "- title: short human title\n"
            "- summary: one or two sentences\n"
            "- tags: array of short strings\n"
            "- confidence: number from 0 to 1\n"
            "- start_offset_seconds: integer offset inside the segment\n"
            "- end_offset_seconds: integer offset inside the segment\n\n"
            "Tighten start_offset_seconds and end_offset_seconds around the actual highlight; "
            "do not span the whole segment unless every frame is a genuine moment.\n\n"
            "Be strict about keep=false: an empty or quiet room with no child visible is "
            "keep=false regardless of furniture, toys, lighting, or audio. If you are unsure "
            "whether my daughter is visible and active, set keep=false and confidence below 0.5. "
            "Only set keep=true when you can clearly identify my daughter in motion or in a "
            "genuine interaction.\n\n"
            "Frames may be sampled with motion awareness: some come from moments of detected "
            "scene motion, others are static baseline frames. Motion alone does not mean my "
            "daughter is present - a curtain moving, a pet walking, or a camera re-adjusting "
            "is keep=false if my daughter is not visible. Conversely, a static frame may still "
            "be a genuine moment if my daughter is visible and engaged in a quiet activity "
            "(e.g. sitting and drawing). Judge each frame by whether my daughter is visible, "
            "not by whether pixels changed.\n\n"
            "Use the time annotations below to set start_offset_seconds and "
            "end_offset_seconds accurately. These annotations are the actual sampled video "
            "times; motion-aware sampling means they may not be evenly spaced. Count frames "
            "or contact-sheet cells from the beginning: #1 is near the first listed time. "
            "If my daughter is clearly active in frame/cell #4, start_offset_seconds should "
            "be near the listed #4 time, NOT 0-5s. Getting the offset wrong means the saved "
            "clip will miss my daughter entirely, so be precise.\n\n"
            f"Segment file: {video_path.name}\n"
            f"Segment duration seconds: {duration_seconds}\n"
            f"Number of provided images: {len(image_paths)}\n"
            f"Time annotations: {frame_offsets}"
        )

        user_content: list[dict[str, Any]] = [{"type": "text", "text": instructions}]
        user_content.extend(_image_content(path) for path in image_paths)

        headers = {"Content-Type": "application/json"}
        if self.settings.llama_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llama_api_key}"

        payload = {
            "model": self.settings.llama_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful family video curator. Return JSON only.",
                },
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.settings.llama_analysis_temperature is not None:
            payload["temperature"] = self.settings.llama_analysis_temperature

        body = await asyncio.to_thread(
            _post_json,
            endpoint,
            headers,
            payload,
            self.settings.llama_timeout_seconds,
        )

        content = body["choices"][0]["message"]["content"]
        raw_text = _extract_message_text(content)
        data = _extract_json(raw_text)

        confidence = max(0.0, min(_coerce_float(data.get("confidence"), 0.0), 1.0))
        start_offset = max(0, _coerce_int(data.get("start_offset_seconds"), 0))
        end_offset = _coerce_int(data.get("end_offset_seconds"), duration_seconds)
        end_offset = max(start_offset + 1, min(end_offset, duration_seconds))
        if annotation_offsets:
            start_offset, end_offset = _snap_highlight_offsets(
                start_offset,
                end_offset,
                annotation_offsets,
                duration_seconds,
            )

        title = str(data.get("title") or "Family moment").strip()[:120]
        summary = str(data.get("summary") or "Saved by the family moment analyzer.").strip()

        return AnalysisResult(
            keep=_coerce_bool(data.get("keep")),
            title=title,
            summary=summary,
            tags=_coerce_tags(data.get("tags")),
            confidence=confidence,
            start_offset_seconds=start_offset,
            end_offset_seconds=end_offset,
            raw=data,
            raw_text=raw_text,
        )

    async def verify_daughter_visible(
        self, frame_paths: list[Path]
    ) -> DaughterVerification:
        """Check whether any verification frame clearly shows the daughter."""
        endpoint = self.settings.llama_base_url.rstrip("/") + "/chat/completions"
        instructions = (
            "You are verifying three chronological frames from a saved family clip. "
            "Is there a young girl (my daughter) clearly visible in any frame? "
            "Answer JSON only with has_daughter (boolean), confidence (0 to 1), "
            "and a short factual description of what is visible. "
            "Only set has_daughter=true if you can clearly see a young girl. "
            "Shadows, furniture, toys, or vague shapes do not count."
        )
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": instructions}
        ]
        user_content.extend(_image_content(path) for path in frame_paths)
        headers = {"Content-Type": "application/json"}
        if self.settings.llama_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llama_api_key}"
        payload = {
            "model": self.settings.llama_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict visual verifier. Return JSON only.",
                },
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.settings.llama_verification_temperature is not None:
            payload["temperature"] = self.settings.llama_verification_temperature
        body = await asyncio.to_thread(
            _post_json,
            endpoint,
            headers,
            payload,
            self.settings.llama_timeout_seconds,
        )
        content = body["choices"][0]["message"]["content"]
        raw_text = _extract_message_text(content)
        data = _extract_json(raw_text)
        return _parse_daughter_verification(data, raw_text)
