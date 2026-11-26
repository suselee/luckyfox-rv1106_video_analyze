"""NAS 侧 4K 高清人脸身份复核 (YuNet 5点对齐 + ArcFace ONNX)。

仅用于 RV1106 上报的 probable 片段 (每天 ~10-20 条), 弥补摄像头子码流
640x360 远景人脸过小导致板端认不出的物理限制。

零运行时第三方强依赖: cv2 / numpy 懒加载, 缺失或模型/底库未就绪时
静默降级 (is_ready() == False), 不影响主服务启动与原有校验流程。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# InsightFace 标准 112x112 5点对齐目标坐标 (右眼/左眼/鼻尖/右嘴角/左嘴角)。
_ARCFACE_DST = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)


@dataclass(frozen=True)
class FaceVerifyResult:
    matched: bool
    best_similarity: float
    hit_frames: int
    faces_checked: int
    best_face_px: int


class DaughterFaceVerifier:
    """使用 YuNet + ArcFace ONNX (如 w600k_r50.onnx) 对 4K 抽帧做女儿身份复核。"""

    def __init__(
        self,
        *,
        model_path: Path,
        gallery_path: Path,
        yunet_path: Path,
        threshold: float = 0.36,
        min_face_score: float = 0.55,
        min_face_px: int = 24,
    ) -> None:
        self.model_path = Path(model_path)
        self.gallery_path = Path(gallery_path)
        self.yunet_path = Path(yunet_path)
        self.threshold = float(threshold)
        self.min_face_score = float(min_face_score)
        self.min_face_px = max(12, int(min_face_px))

        self._lock = threading.Lock()
        self._cv2: Any = None
        self._np: Any = None
        self._net: Any = None
        self._yunet: Any = None
        self._gallery: Any = None
        self._gallery_mtime_ns: int = -1
        self._arcface_dst: Any = None
        self._import_failed = False

    def is_ready(self) -> bool:
        if self._import_failed:
            return False
        return (
            self.model_path.is_file()
            and self.gallery_path.is_file()
            and self.yunet_path.is_file()
        )

    def _ensure_loaded(self) -> bool:
        if not self.is_ready():
            return False
        if self._cv2 is None:
            try:
                import cv2
                import numpy as np
            except ImportError:
                self._import_failed = True
                return False
            self._cv2 = cv2
            self._np = np
            self._arcface_dst = np.array(_ARCFACE_DST, dtype=np.float32)

        if self._yunet is None:
            self._yunet = self._cv2.FaceDetectorYN.create(
                str(self.yunet_path),
                "",
                (320, 320),
                self.min_face_score,
                0.3,
                5000,
            )
        if self._net is None:
            self._net = self._cv2.dnn.readNetFromONNX(str(self.model_path))

        try:
            mtime_ns = self.gallery_path.stat().st_mtime_ns
        except OSError:
            return False
        if self._gallery is None or mtime_ns != self._gallery_mtime_ns:
            self._gallery = self._load_gallery(self.gallery_path)
            self._gallery_mtime_ns = mtime_ns
        return self._gallery is not None and len(self._gallery) > 0

    def _load_gallery(self, path: Path) -> Any:
        np = self._np
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            try:
                vecs = np.asarray(data["embeddings"], dtype=np.float32)
            finally:
                data.close()
        else:
            vecs = np.asarray(data, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.ndim != 2 or vecs.shape[0] == 0:
            return None
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-6)

    def detect_and_align_faces(
        self, frame: Any, *, min_score: float | None = None, min_px: int | None = None
    ) -> list[tuple[Any, float, int]]:
        cv2 = self._cv2
        np = self._np
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        h, w = frame.shape[:2]
        if w < 32 or h < 32:
            return []
        score_thresh = self.min_face_score if min_score is None else float(min_score)
        px_thresh = self.min_face_px if min_px is None else int(min_px)
        self._yunet.setInputSize((w, h))
        self._yunet.setScoreThreshold(score_thresh)
        _, faces = self._yunet.detect(frame)
        if faces is None or len(faces) == 0:
            return []
        aligned_faces: list[tuple[Any, float, int]] = []
        for row in faces:
            fw, fh = float(row[2]), float(row[3])
            score = float(row[14])
            min_dim = min(fw, fh)
            if min_dim < px_thresh:
                continue
            pts = np.array(
                [
                    [row[4], row[5]],
                    [row[6], row[7]],
                    [row[8], row[9]],
                    [row[10], row[11]],
                    [row[12], row[13]],
                ],
                dtype=np.float32,
            )
            matrix, _ = cv2.estimateAffinePartial2D(
                pts, self._arcface_dst, method=cv2.LMEDS
            )
            if matrix is None:
                continue
            aligned = cv2.warpAffine(frame, matrix, (112, 112), borderValue=0.0)
            aligned_faces.append((aligned, score, int(round(min_dim))))
        return aligned_faces

    def embed_aligned(self, aligned: Any) -> Any:
        cv2 = self._cv2
        np = self._np
        blob = cv2.dnn.blobFromImage(
            aligned,
            1.0 / 127.5,
            (112, 112),
            (127.5, 127.5, 127.5),
            swapRB=True,
        )
        self._net.setInput(blob)
        vec = self._net.forward().reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-6 else vec

    def verify_paths(
        self,
        image_paths: list[Path],
        *,
        threshold: float | None = None,
        min_hits: int = 1,
    ) -> FaceVerifyResult:
        """在给定 4K 帧中检索女儿人脸, 返回最高相似度与命中帧数。"""
        with self._lock:
            if not self._ensure_loaded():
                return FaceVerifyResult(
                    matched=False,
                    best_similarity=0.0,
                    hit_frames=0,
                    faces_checked=0,
                    best_face_px=0,
                )
            cv2 = self._cv2
            np = self._np
            thresh = self.threshold if threshold is None else float(threshold)
            needed = max(1, int(min_hits))
            best_sim = 0.0
            hit_frames = 0
            faces_checked = 0
            best_px = 0

            for path in image_paths:
                frame = cv2.imread(str(path))
                if frame is None:
                    continue
                frame_best = 0.0
                for aligned, _score, fpx in self.detect_and_align_faces(frame):
                    faces_checked += 1
                    vec = self.embed_aligned(aligned)
                    sim = float(np.max(self._gallery @ vec))
                    if sim > frame_best:
                        frame_best = sim
                    if sim > best_sim:
                        best_sim = sim
                        best_px = fpx
                if frame_best >= thresh:
                    hit_frames += 1
                    # 强命中提前结束, 减少单次校验耗时
                    if hit_frames >= needed and best_sim >= thresh + 0.10:
                        break

            return FaceVerifyResult(
                matched=hit_frames >= needed,
                best_similarity=round(best_sim, 4),
                hit_frames=hit_frames,
                faces_checked=faces_checked,
                best_face_px=best_px,
            )
