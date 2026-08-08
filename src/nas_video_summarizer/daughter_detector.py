from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from .analysis import ClipCandidate
from .config import Settings
from .ffmpeg_tools import SampledFrame
from .person_filter import PersonFilter


@dataclass(frozen=True)
class DaughterObservation:
    offset_seconds: float
    confidence: float
    boxes: list[list[float]]
    person_count: int
    evidence: str = ""

    @property
    def positive(self) -> bool:
        return self.confidence > 0 and bool(self.boxes)


@dataclass(frozen=True)
class ProbableVerification:
    accepted: bool
    positive_frames: int
    required_frames: int
    evidence: tuple[str, ...]
    reason: str
    decision: str = "inconclusive"
    sampled_frames: int = 0
    person_frames: int = 0
    face_frames: int = 0
    child_face_frames: int = 0
    adult_face_frames: int = 0
    max_person_score: float = 0.0


class DaughterDetector:
    """CPU-only daughter detector using OpenCV DNN.

    ``heuristic`` reuses conservative face/age evidence. ``onnx`` expects a
    one-class YOLO export whose class 0 is ``daughter``.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.mode = settings.daughter_detector_mode
        self._net = None
        self._cv2 = None
        self._np = None
        self._person_filter: PersonFilter | None = None
        self._heuristic_frame_index = 0
        self._next_age_check = 0
        self._cached_child_score = 0.0
        self._cached_child_until = -1.0

    def reset_segment(self) -> None:
        self._heuristic_frame_index = 0
        self._next_age_check = 0
        self._cached_child_score = 0.0
        self._cached_child_until = -1.0

    def prepare(self) -> str:
        if self.mode == "heuristic":
            self._get_person_filter().prepare()
            return "person+face-age heuristic"
        if self.mode != "onnx":
            raise ValueError(f"unsupported DAUGHTER_DETECTOR_MODE: {self.mode}")
        path = self.settings.daughter_detector_model_path
        if path is None or not path.is_file():
            raise FileNotFoundError(f"daughter detector model not found: {path}")
        self._init_onnx()
        return str(path)

    def _get_person_filter(self) -> PersonFilter:
        if self._person_filter is None:
            self._person_filter = PersonFilter(
                threshold=self.settings.person_filter_threshold,
                backend=self.settings.person_filter_backend,
                model_url=self.settings.person_filter_model_url,
                model_dir=self.settings.person_filter_model_dir,
                face_threshold=self.settings.person_filter_face_threshold,
                adult_threshold=self.settings.person_filter_adult_threshold,
            )
        return self._person_filter

    def _init_onnx(self) -> None:
        if self._net is not None:
            return
        import cv2
        import numpy as np

        self._cv2 = cv2
        self._np = np
        self._net = cv2.dnn.readNetFromONNX(
            str(self.settings.daughter_detector_model_path)
        )
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    @staticmethod
    def _encoded(path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("ascii")

    def detect_path(self, frame: SampledFrame) -> DaughterObservation:
        encoded = self._encoded(frame.path)
        if self.mode == "heuristic":
            self._heuristic_frame_index += 1
            check_age = self._heuristic_frame_index >= self._next_age_check
            if check_age:
                self._next_age_check = (
                    self._heuristic_frame_index
                    + max(1, self.settings.daughter_age_check_every)
                )
            info = self._get_person_filter().detect(encoded, classify_age=check_age)
            detected_child_score = float(info.get("child_score", 0.0))
            if detected_child_score >= self.settings.person_filter_child_threshold:
                self._cached_child_score = detected_child_score
                self._cached_child_until = (
                    frame.offset_seconds + self.settings.daughter_event_max_gap_seconds
                )
            child_score = (
                detected_child_score
                if check_age
                else self._cached_child_score
                if frame.offset_seconds <= self._cached_child_until
                else 0.0
            )
            boxes = list(info.get("person_boxes", []))
            if child_score < self.settings.person_filter_child_threshold or not boxes:
                body_box = self._relative_child_box(boxes)
                if body_box is None:
                    return DaughterObservation(
                        frame.offset_seconds, 0.0, [], len(boxes)
                    )
                return DaughterObservation(
                    frame.offset_seconds,
                    max(0.55, self.settings.daughter_detector_threshold),
                    [body_box],
                    len(boxes),
                    "relative_body_size",
                )
            # The age model currently exposes aggregate evidence. A toddler is
            # normally the smallest visible person in this fixed indoor view.
            daughter_box = min(boxes, key=lambda box: float(box[2]) * float(box[3]))
            return DaughterObservation(
                frame.offset_seconds,
                child_score,
                [daughter_box],
                len(boxes),
                "face_age",
            )

        boxes, score = self._detect_onnx(frame.path)
        if score < self.settings.daughter_detector_threshold:
            return DaughterObservation(frame.offset_seconds, 0.0, [], 0)
        # Only daughter-positive frames pay the secondary generic-person cost.
        context = self._get_person_filter().detect(encoded, classify_age=False)
        return DaughterObservation(
            frame.offset_seconds,
            score,
            boxes,
            max(1, len(context.get("person_boxes", []))),
            "daughter_onnx",
        )

    def _relative_child_box(
        self, boxes: list[list[float]]
    ) -> list[float] | None:
        if not self.settings.daughter_body_fallback_enabled or len(boxes) < 2:
            return None
        valid = [
            box for box in boxes
            if float(box[2]) > 0 and float(box[3]) > 0
        ]
        if len(valid) < 2:
            return None
        smallest = min(valid, key=lambda box: float(box[2]) * float(box[3]))
        largest = max(valid, key=lambda box: float(box[2]) * float(box[3]))
        small_area = float(smallest[2]) * float(smallest[3])
        large_area = float(largest[2]) * float(largest[3])
        if large_area <= 0 or float(largest[3]) <= 0:
            return None
        height_ratio = float(smallest[3]) / float(largest[3])
        area_ratio = small_area / large_area
        if (
            height_ratio <= self.settings.daughter_body_height_ratio
            and area_ratio <= self.settings.daughter_body_area_ratio
        ):
            return smallest
        return None

    def _detect_onnx(self, path: Path) -> tuple[list[list[float]], float]:
        self._init_onnx()
        data = self._np.frombuffer(path.read_bytes(), dtype=self._np.uint8)
        image = self._cv2.imdecode(data, self._cv2.IMREAD_COLOR)
        if image is None:
            return [], 0.0
        size = max(32, self.settings.daughter_detector_input_size)
        blob = self._cv2.dnn.blobFromImage(
            image, 1 / 255.0, (size, size), swapRB=True, crop=False
        )
        self._net.setInput(blob)
        output = self._net.forward()
        predictions = output[0]
        boxes: list[list[float]] = []
        scores: list[float] = []
        if predictions.shape[-1] == 6:
            for row in predictions:
                score = float(row[4])
                if int(row[5]) != 0 or score < self.settings.daughter_detector_threshold:
                    continue
                x1, y1, x2, y2 = (float(value) for value in row[:4])
                scale = size if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 2 else 1
                boxes.append([x1 / scale, y1 / scale, (x2 - x1) / scale, (y2 - y1) / scale])
                scores.append(score)
        else:
            rows = predictions.T if predictions.shape[0] < predictions.shape[1] else predictions
            for row in rows:
                class_scores = row[4:]
                if len(class_scores) == 0:
                    continue
                class_id = int(self._np.argmax(class_scores))
                score = float(class_scores[class_id])
                if class_id != 0 or score < self.settings.daughter_detector_threshold:
                    continue
                x, y, width, height = (float(value) for value in row[:4])
                boxes.append([
                    (x - width / 2) / size,
                    (y - height / 2) / size,
                    width / size,
                    height / size,
                ])
                scores.append(score)
        if not scores:
            return [], 0.0
        keep = self._nms(boxes, scores, 0.45)
        return [boxes[index] for index in keep], max(scores[index] for index in keep)

    @staticmethod
    def _nms(boxes: list[list[float]], scores: list[float], threshold: float) -> list[int]:
        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        keep: list[int] = []
        while order:
            current = order.pop(0)
            keep.append(current)
            order = [
                index for index in order
                if DaughterDetector._iou(boxes[current], boxes[index]) <= threshold
            ]
        return keep

    @staticmethod
    def _iou(first: list[float], second: list[float]) -> float:
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        left, top = max(ax, bx), max(ay, by)
        right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        union = aw * ah + bw * bh - intersection
        return intersection / union if union > 0 else 0.0

    def candidates(self, observations: list[DaughterObservation]) -> list[ClipCandidate]:
        positives = [observation for observation in observations if observation.positive]
        if not positives:
            return []
        groups: list[list[DaughterObservation]] = []
        current: list[DaughterObservation] = []
        for observation in positives:
            if current and (
                observation.offset_seconds - current[-1].offset_seconds
                > self.settings.daughter_event_max_gap_seconds
            ):
                groups.append(current)
                current = []
            current.append(observation)
        if current:
            groups.append(current)

        interval = 1.0 / max(self.settings.daughter_scan_fps, 0.01)
        candidates: list[ClipCandidate] = []
        for group in groups:
            duration = group[-1].offset_seconds - group[0].offset_seconds + interval
            if len(group) < self.settings.daughter_event_min_hits:
                continue
            if duration < self.settings.daughter_event_min_seconds:
                continue
            confidence = max(item.confidence for item in group)
            areas = [box[2] * box[3] for item in group for box in item.boxes]
            movements = []
            centers = [
                (item.boxes[0][0] + item.boxes[0][2] / 2, item.boxes[0][1] + item.boxes[0][3] / 2)
                for item in group
            ]
            for first, second in zip(centers, centers[1:]):
                movements.append(abs(second[0] - first[0]) + abs(second[1] - first[1]))
            activity = min(1.0, mean(movements) * 8) if movements else 0.0
            multi_ratio = sum(item.person_count >= 2 for item in group) / len(group)
            category = "multi_person" if multi_ratio >= 0.3 else "active" if activity >= 0.18 else "quiet"
            persistence = min(1.0, duration / 20.0)
            visibility = min(1.0, median(areas) / 0.08) if areas else 0.0
            score = min(1.0, 0.55 * confidence + 0.20 * persistence + 0.15 * visibility + 0.10 * activity)
            labels = {
                "active": ("Daughter active", "女儿在室内画面中持续出现，活动程度较高。"),
                "multi_person": ("Daughter with others", "女儿在室内画面中持续出现，并有其他人同框。"),
                "quiet": ("Daughter quiet activity", "女儿在室内画面中持续出现，活动程度较低。"),
            }
            title, summary = labels[category]
            candidates.append(
                ClipCandidate(
                    keep=True,
                    title=title,
                    summary=summary,
                    tags=["daughter", category],
                    confidence=confidence,
                    start_offset_seconds=max(0, int(group[0].offset_seconds)),
                    end_offset_seconds=max(
                        int(group[0].offset_seconds) + 1,
                        int(group[-1].offset_seconds + interval),
                    ),
                    raw={
                        "detector_mode": self.mode,
                        "hit_count": len(group),
                        "duration_seconds": round(duration, 3),
                        "activity_score": round(activity, 4),
                        "max_person_count": max(item.person_count for item in group),
                        "evidence": sorted({item.evidence for item in group if item.evidence}),
                        "median_bbox_area": round(median(areas), 6) if areas else 0.0,
                    },
                    local_child_confirmed=True,
                    local_child_score=confidence,
                    analysis_backend="daughter_detector",
                    category=category,
                    selection_score=score,
                )
            )
        return candidates

    def verify_paths(self, paths: list[Path]) -> bool:
        for index, path in enumerate(paths):
            observation = self.detect_path(
                SampledFrame(path=path, offset_seconds=float(index))
            )
            if observation.positive:
                return True
        return False

    def verify_board_probable_paths(
        self,
        paths: list[Path],
        *,
        required_frames: int = 2,
        required_person_frames: int = 3,
        board_score: float = 0.0,
        board_person_score: float = 0.0,
        board_score_threshold: float = 0.70,
        board_person_score_threshold: float = 0.70,
    ) -> ProbableVerification:
        """Verify a board body-track event using event-local cropped frames.

        A child face is strong positive evidence. Repeated adult/non-child
        faces are negative evidence. When no usable face is visible at all,
        repeated person detections may pass only alongside strong board scores.
        """
        self.reset_segment()
        frame_infos: list[dict[str, object]] = []
        onnx_hits = 0
        person_filter = self._get_person_filter()
        for path in paths:
            encoded = self._encoded(path)
            info = person_filter.detect(encoded, classify_age=True)
            frame_infos.append(info)
            if self.mode == "onnx":
                _, daughter_score = self._detect_onnx(path)
                if daughter_score >= self.settings.daughter_detector_threshold:
                    onnx_hits += 1

        person_frames = sum(
            bool(info.get("person_boxes"))
            and float(info.get("person_score", 0.0))
            >= self.settings.person_filter_threshold
            for info in frame_infos
        )
        face_frames = sum(int(info.get("face_count", 0)) > 0 for info in frame_infos)
        child_face_frames = sum(
            int(info.get("face_count", 0)) > 0
            and float(info.get("child_score", 0.0))
            >= self.settings.person_filter_child_threshold
            for info in frame_infos
        )
        adult_face_frames = sum(bool(info.get("adult_only")) for info in frame_infos)
        max_person_score = max(
            (float(info.get("person_score", 0.0)) for info in frame_infos),
            default=0.0,
        )
        required = max(1, required_frames)
        required_people = max(1, required_person_frames)

        accepted = False
        decision = "inconclusive"
        reason = "insufficient independent evidence"
        if self.mode == "onnx" and onnx_hits >= required:
            accepted = True
            decision = "daughter_onnx"
            reason = "repeated daughter ONNX evidence"
        elif child_face_frames >= 1:
            accepted = True
            decision = "child_face"
            reason = "independent child face-age evidence"
        elif adult_face_frames >= 2:
            decision = "adult_face"
            reason = "repeated adult-only face evidence"
        elif face_frames >= 2:
            decision = "face_without_child"
            reason = "repeated face evidence without child evidence"
        elif (
            face_frames == 0
            and person_frames >= required_people
            and board_score >= board_score_threshold
            and board_person_score >= board_person_score_threshold
        ):
            accepted = True
            decision = "faceless_person"
            reason = "stable person ROI with strong board evidence"
        elif person_frames == 0:
            decision = "no_person"
            reason = "no person detected in the board ROI"

        evidence_values = []
        if onnx_hits:
            evidence_values.append("daughter_onnx")
        if child_face_frames:
            evidence_values.append("child_face")
        if adult_face_frames:
            evidence_values.append("adult_face")
        if person_frames:
            evidence_values.append("person")
        return ProbableVerification(
            accepted=accepted,
            positive_frames=max(child_face_frames, onnx_hits),
            required_frames=required,
            evidence=tuple(evidence_values),
            reason=reason,
            decision=decision,
            sampled_frames=len(paths),
            person_frames=person_frames,
            face_frames=face_frames,
            child_face_frames=child_face_frames,
            adult_face_frames=adult_face_frames,
            max_person_score=max_person_score,
        )
