from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_CHILD_TERMS = ("daughter", "young girl", "young child", "toddler", "child")
_ACTIVITY_TERMS = (
    "play", "activity", "interact", "engaged", "explor", "walk", "read",
    "laugh", "babbl", "cuddl", "danc", "draw",
)
_EXCLUSION_TERMS = (
    "empty room", "no child", "no young girl", "no girl", "cannot see",
    "can't see", "adults-only", "adult only", "only adults", "sleeping",
    "drowsy", "screen time", "blurry", "black frame", "not clearly visible",
    "not engaged", "no activity", "merely idle", "sitting idle", "passive",
)


@dataclass(frozen=True)
class ClipCandidate:
    """Backend-neutral candidate passed to the clip publisher."""

    keep: bool
    title: str
    summary: str
    tags: list[str]
    confidence: float
    start_offset_seconds: int
    end_offset_seconds: int
    raw: dict[str, Any]
    raw_text: str = ""
    local_child_confirmed: bool = False
    local_child_score: float = 0.0
    analysis_backend: str = "vlm"
    category: str = "semantic"
    selection_score: float | None = None

    @property
    def effective_selection_score(self) -> float:
        return self.confidence if self.selection_score is None else self.selection_score

    def should_save(self, threshold: float) -> bool:
        return self.confidence >= threshold and (
            self.keep or self.keep_consistency_repaired(threshold)
        )

    def keep_consistency_repaired(self, threshold: float) -> bool:
        if (
            self.keep
            or self.analysis_backend != "vlm"
            or not self.local_child_confirmed
            or self.confidence < max(threshold, 0.75)
        ):
            return False
        evidence = " ".join([self.title, self.summary, *self.tags]).lower()
        if any(term in evidence for term in _EXCLUSION_TERMS):
            return False
        return any(term in evidence for term in _CHILD_TERMS) and any(
            term in evidence for term in _ACTIVITY_TERMS
        )
