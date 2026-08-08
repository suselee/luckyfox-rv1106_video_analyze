from __future__ import annotations

import base64
from pathlib import Path
from urllib import request


_MODEL_DIR = Path(__file__).resolve().parent / "_person_filter_models"

# YOLO-family models. Two ONNX export layouts are supported and detected
# automatically by the decoder:
#   * raw  (one-to-many head):  output (1, nc+4, N)  -> needs NMS
#   * e2e  (one-to-one head):   output (1, 300, 6)   -> already decoded, no NMS
# v8/v11/v12 use the raw layout; YOLO26 uses e2e by default (or raw with
# end2end=False). Weights are downloaded on first use. A `PERSON_FILTER_MODEL_URL`
# env override may point at any compatible ONNX export.
_YOLO_NETS = {
    "yolov11n": {
        "filename": "yolov11n.onnx",
        "url": (
            "https://huggingface.co/unity/inference-engine-yolo/"
            "resolve/main/models/yolo11n.onnx"
        ),
        "input_size": 640,
    },
    "yolov8n": {
        "filename": "yolov8n.onnx",
        "url": (
            "https://github.com/ultralytics/assets/releases/download/"
            "v8.4.0/yolov8n.onnx"
        ),
        "input_size": 640,
    },
    "yolov26n": {
        "filename": "yolov26n.onnx",
        "url": (
            "https://huggingface.co/zwh20081/yolo26-onnx/"
            "resolve/main/yolo26n.onnx"
        ),
        "input_size": 640,
    },
}

# MobileNet-SSD (OpenCV Caffe sample, VOC 20 classes, person = 15)
_MOBILENET_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/"
    "master/deploy.prototxt"
)
_MOBILENET_CAFFEMODEL_URL = (
    "https://github.com/chuanqi305/MobileNet-SSD/raw/"
    "master/mobilenet_iter_73000.caffemodel"
)
_MOBILENET_PERSON_CLASS_ID = 15

# OpenCV's sample face detector plus the Levi-Hassner age classifier. The age
# model predicts eight coarse buckets; the first three are treated as child
# buckets. Adult-only filtering remains conservative: every person must have a
# matched face and every matched face must be confidently adult.
_FACE_PROTO_URL = (
    "https://raw.githubusercontent.com/spmallick/learnopencv/"
    "master/AgeGender/opencv_face_detector.pbtxt"
)
_FACE_MODEL_URL = (
    "https://raw.githubusercontent.com/spmallick/learnopencv/"
    "master/AgeGender/opencv_face_detector_uint8.pb"
)
_AGE_PROTO_URL = (
    "https://raw.githubusercontent.com/spmallick/learnopencv/"
    "master/AgeGender/age_deploy.prototxt"
)
_AGE_MODEL_URL = (
    "https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=1"
)
_AGE_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
_CHILD_AGE_BUCKETS = 3

_YOLO_PERSON_CLASS_ID = 0
_YOLO_INPUT_SIZE = 640
_DEFAULT_MODEL = "yolov11n"


def _ensure_models(
    model_key: str, model_url: str = "", model_dir: Path = _MODEL_DIR
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_key == "mobilenet_ssd":
        prototxt = model_dir / "deploy.prototxt"
        caffemodel = model_dir / "mobilenet_iter_73000.caffemodel"
        if not prototxt.exists():
            _download(_MOBILENET_PROTOTXT_URL, prototxt)
        if not caffemodel.exists():
            _download(_MOBILENET_CAFFEMODEL_URL, caffemodel)
        return caffemodel

    spec = _YOLO_NETS[model_key]
    onnx = model_dir / spec["filename"]
    if not onnx.exists():
        _download(model_url or spec["url"], onnx)
    return onnx


def _ensure_age_models(model_dir: Path = _MODEL_DIR) -> tuple[Path, Path, Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    assets = (
        (model_dir / "opencv_face_detector.pbtxt", _FACE_PROTO_URL),
        (model_dir / "opencv_face_detector_uint8.pb", _FACE_MODEL_URL),
        (model_dir / "age_deploy.prototxt", _AGE_PROTO_URL),
        (model_dir / "age_net.caffemodel", _AGE_MODEL_URL),
    )
    for path, url in assets:
        if not path.exists():
            _download(url, path)
    return tuple(path for path, _ in assets)


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} ...")
    temporary = dest.with_suffix(f"{dest.suffix}.part")
    try:
        with request.urlopen(url) as resp, temporary.open("wb") as output:
            while chunk := resp.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)


class PersonFilter:
    """Local person detection, runs inline on the NAS (no HTTP server).

    Both backends run via OpenCV DNN, so the **only** runtime dependency is
    opencv — no onnxruntime / torch needed.

    Backends (``PERSON_FILTER_BACKEND``):
      * ``yolov11n``    (default) — YOLOv11n ONNX, fastest + most accurate on
                       CPU. Output format is compatible with v8/v12 too.
      * ``yolov8n``     — YOLOv8n ONNX, slightly older but well proven.
      * ``mobilenet_ssd`` — the older OpenCV Caffe sample model, lower
                       accuracy / more false positives.

    Returns person confidence and boxes plus conservative face/age evidence.
    ``adult_only`` is true only when every person box has a matched face and
    every matched face is confidently classified into an adult age bucket.
    """

    def __init__(
        self,
        threshold: float = 0.2,
        backend: str = _DEFAULT_MODEL,
        model_url: str = "",
        model_dir: Path | None = None,
        face_threshold: float = 0.7,
        adult_threshold: float = 0.9,
    ):
        if backend not in _YOLO_NETS and backend != "mobilenet_ssd":
            backend = _DEFAULT_MODEL
        self._backend = backend
        self._threshold = threshold
        self._model_url = model_url
        self._model_dir = model_dir or _MODEL_DIR
        self._face_threshold = face_threshold
        self._adult_threshold = adult_threshold
        self._is_yolo = backend != "mobilenet_ssd"
        self._input_size = _YOLO_NETS.get(backend, {}).get("input_size", _YOLO_INPUT_SIZE)
        self._net = None
        self._cv2 = None
        self._np = None
        self._model_path = None
        self._face_net = None
        self._age_net = None

    def prepare(self) -> Path:
        """Download and load the configured model, returning its local path."""
        self._init_net()
        assert self._model_path is not None
        return self._model_path

    def _init_net(self):
        if (
            self._net is not None
            and self._face_net is not None
            and self._age_net is not None
        ):
            return
        import cv2
        import numpy as np

        self._cv2 = cv2
        self._np = np

        model_path = _ensure_models(
            self._backend, self._model_url, self._model_dir
        )
        self._model_path = model_path
        if self._is_yolo:
            self._net = cv2.dnn.readNetFromONNX(str(model_path))
        else:
            prototxt = self._model_dir / "deploy.prototxt"
            self._net = cv2.dnn.readNetFromCaffe(str(prototxt), str(model_path))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        face_proto, face_model, age_proto, age_model = _ensure_age_models(
            self._model_dir
        )
        self._face_net = cv2.dnn.readNet(str(face_model), str(face_proto))
        self._age_net = cv2.dnn.readNetFromCaffe(str(age_proto), str(age_model))
        for net in (self._face_net, self._age_net):
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(
        self, image_b64: str, *, classify_age: bool = True
    ) -> dict[str, object]:
        self._init_net()

        img_bytes = base64.b64decode(image_b64)
        arr = self._np.frombuffer(img_bytes, dtype=self._np.uint8)
        img = self._cv2.imdecode(arr, self._cv2.IMREAD_COLOR)
        if img is None:
            return {
                "person_score": 0.0,
                "face_score": 0.0,
                "face_bbox_area": 0.0,
                "person_boxes": [],
                "face_count": 0,
                "matched_person_count": 0,
                "child_score": 0.0,
                "adult_score": 0.0,
                "adult_only": False,
            }

        if self._is_yolo:
            result = self._detect_yolo(img)
        else:
            result = self._detect_mobilenet(img)
        if classify_age:
            result.update(self._classify_ages(img, result["person_boxes"]))
        else:
            result.update(self._age_result())
        return result

    def _detect_mobilenet(self, img) -> dict[str, object]:
        blob = self._cv2.dnn.blobFromImage(
            img, 0.007843, (300, 300), 127.5, swapRB=False
        )
        self._net.setInput(blob)
        detections = self._net.forward()

        person_score = 0.0
        boxes: list[list[float]] = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self._threshold:
                continue
            class_id = int(detections[0, 0, i, 1])
            if class_id == _MOBILENET_PERSON_CLASS_ID:
                person_score = max(person_score, confidence)
                x1, y1, x2, y2 = (float(v) for v in detections[0, 0, i, 3:7])
                boxes.append([x1, y1, x2 - x1, y2 - y1])

        return self._pack(1, person_score, boxes)

    def _detect_yolo(self, img) -> dict[str, object]:
        size = self._input_size
        blob = self._cv2.dnn.blobFromImage(
            img,
            1 / 255.0,
            (size, size),
            swapRB=True,
            crop=False,
        )
        self._net.setInput(blob)
        outputs = self._net.forward()
        preds = outputs[0]

        # e2e (one-to-one head) export: (N, 300, 6) -> [x1,y1,x2,y2,score,cls]
        if preds.shape[-1] == 6:
            return self._decode_yolo_e2e(preds, size)
        # raw (one-to-many head) export: (1, nc+4, N) -> transpose to (N, nc+4)
        return self._decode_yolo_raw(preds.T, size)

    def _decode_yolo_raw(self, predictions, size: int) -> dict[str, object]:
        boxes = []
        scores = []
        for pred in predictions:
            class_scores = pred[4:]
            class_id = int(self._np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if class_id != _YOLO_PERSON_CLASS_ID or confidence < self._threshold:
                continue
            x, y, bw, bh = pred[0:4]
            # bbox is center_x, center_y, width, height in input-pixel space
            x1 = x - bw / 2
            y1 = y - bh / 2
            boxes.append([float(x1), float(y1), float(bw), float(bh)])
            scores.append(confidence)

        person_score = 0.0
        kept_boxes: list[list[float]] = []
        if boxes:
            keep = self._nms(boxes, scores, iou_threshold=0.45)
            for i in keep:
                person_score = max(person_score, scores[i])
                kept_boxes.append(boxes[i])

        return self._pack(size, person_score, kept_boxes)

    def _decode_yolo_e2e(self, preds, size: int) -> dict[str, object]:
        # Already-decoded detections: rows of [x1, y1, x2, y2, score, cls].
        person_score = 0.0
        boxes: list[list[float]] = []
        for row in preds:
            score = float(row[4])
            class_id = int(row[5])
            if class_id != _YOLO_PERSON_CLASS_ID or score < self._threshold:
                continue
            x1, y1, x2, y2 = (float(v) for v in row[0:4])
            w = x2 - x1
            h = y2 - y1
            person_score = max(person_score, score)
            boxes.append([x1, y1, w, h])

        return self._pack(size, person_score, boxes)

    @staticmethod
    def _pack(
        size: int, person_score: float, boxes: list[list[float]]
    ) -> dict[str, object]:
        # Normalize bbox area by input resolution for a rough 0..1 occupant
        # signal. e2e coordinates are in input-pixel space when max > 1.
        normalized_boxes: list[list[float]] = []
        for x, y, width, height in boxes:
            if size > 1:
                normalized_boxes.append(
                    [x / size, y / size, width / size, height / size]
                )
            else:
                normalized_boxes.append([x, y, width, height])
        norm_area = max(
            (max(0.0, box[2]) * max(0.0, box[3]) for box in normalized_boxes),
            default=0.0,
        )
        return {
            "person_score": person_score,
            "face_score": 0.0,
            "face_bbox_area": min(1.0, norm_area),
            "person_boxes": normalized_boxes,
        }

    def _classify_ages(
        self, img, person_boxes: list[list[float]]
    ) -> dict[str, object]:
        if not person_boxes:
            return self._age_result()

        height, width = img.shape[:2]
        blob = self._cv2.dnn.blobFromImage(
            img,
            1.0,
            (300, 300),
            (104, 117, 123),
            swapRB=True,
            crop=False,
        )
        self._face_net.setInput(blob)
        detections = self._face_net.forward()

        matched_people: set[int] = set()
        adult_scores: list[float] = []
        child_scores: list[float] = []
        face_count = 0
        for index in range(detections.shape[2]):
            confidence = float(detections[0, 0, index, 2])
            if confidence < self._face_threshold:
                continue
            x1, y1, x2, y2 = (float(v) for v in detections[0, 0, index, 3:7])
            x1 = max(0, min(width - 1, int(x1 * width)))
            y1 = max(0, min(height - 1, int(y1 * height)))
            x2 = max(x1 + 1, min(width, int(x2 * width)))
            y2 = max(y1 + 1, min(height, int(y2 * height)))
            match = self._matching_person(
                ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height)),
                person_boxes,
            )
            if match is None:
                continue

            padding_x = max(2, int((x2 - x1) * 0.1))
            padding_y = max(2, int((y2 - y1) * 0.1))
            crop_x1 = max(0, x1 - padding_x)
            crop_y1 = max(0, y1 - padding_y)
            crop_x2 = min(width, x2 + padding_x)
            crop_y2 = min(height, y2 + padding_y)
            face = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if face.size == 0:
                continue
            age_blob = self._cv2.dnn.blobFromImage(
                face,
                1.0,
                (227, 227),
                _AGE_MEAN_VALUES,
                swapRB=False,
            )
            self._age_net.setInput(age_blob)
            probabilities = self._age_net.forward()[0]
            child_score = float(sum(probabilities[:_CHILD_AGE_BUCKETS]))
            adult_score = float(sum(probabilities[_CHILD_AGE_BUCKETS:]))
            face_count += 1
            matched_people.add(match)
            child_scores.append(child_score)
            adult_scores.append(adult_score)

        all_people_matched = len(matched_people) == len(person_boxes)
        adult_only = (
            all_people_matched
            and bool(adult_scores)
            and all(score >= self._adult_threshold for score in adult_scores)
        )
        return self._age_result(
            face_count=face_count,
            matched_person_count=len(matched_people),
            child_score=max(child_scores, default=0.0),
            adult_score=min(adult_scores, default=0.0),
            adult_only=adult_only,
        )

    @staticmethod
    def _matching_person(
        face_center: tuple[float, float], person_boxes: list[list[float]]
    ) -> int | None:
        x, y = face_center
        candidates: list[tuple[float, int]] = []
        for index, (px, py, width, height) in enumerate(person_boxes):
            if px <= x <= px + width and py <= y <= py + height:
                candidates.append((width * height, index))
        if not candidates:
            return None
        return min(candidates)[1]

    @staticmethod
    def _age_result(
        *,
        face_count: int = 0,
        matched_person_count: int = 0,
        child_score: float = 0.0,
        adult_score: float = 0.0,
        adult_only: bool = False,
    ) -> dict[str, object]:
        return {
            "face_count": face_count,
            "matched_person_count": matched_person_count,
            "child_score": child_score,
            "adult_score": adult_score,
            "adult_only": adult_only,
        }

    def _nms(self, boxes, scores, iou_threshold: float) -> list[int]:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        keep: list[int] = []
        used = [False] * len(boxes)
        for i in order:
            if used[i]:
                continue
            keep.append(i)
            for j in order:
                if j == i or used[j]:
                    continue
                if self._iou(boxes[i], boxes[j]) > iou_threshold:
                    used[j] = True
        return keep

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        if union <= 0:
            return 0.0
        return inter / union
