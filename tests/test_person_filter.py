import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from nas_video_summarizer import person_filter as pf


def test_invalid_backend_falls_back_to_yolov11n():
    f = pf.PersonFilter(backend="bogus")
    assert f._backend == "yolov11n"
    assert pf.PersonFilter(backend="mobilenet_ssd")._backend == "mobilenet_ssd"
    assert pf.PersonFilter(backend="yolov8n")._backend == "yolov8n"


def test_iou_overlap_and_disjoint():
    a = [0.0, 0.0, 10.0, 10.0]
    b = [5.0, 0.0, 10.0, 10.0]
    c = [100.0, 100.0, 10.0, 10.0]
    # half x-overlap of equal boxes -> 50/150 = 1/3
    assert abs(pf.PersonFilter._iou(a, b) - 1 / 3) < 1e-6
    assert pf.PersonFilter._iou(a, c) == 0.0
    assert pf.PersonFilter._iou(a, a) == 1.0


def test_nms_suppresses_overlapping_boxes():
    f = pf.PersonFilter()
    # two heavily overlapping high-score boxes + one disjoint low-score box
    boxes = [
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 10.0, 10.0],
        [100.0, 100.0, 10.0, 10.0],
    ]
    scores = [0.9, 0.8, 0.5]
    keep = f._nms(boxes, scores, iou_threshold=0.45)
    assert keep == [0, 2]


def test_yolo_decode_picks_person_and_skips_other_classes():
    f = pf.PersonFilter(threshold=0.3, backend="yolov11n")
    f._cv2 = _FakeCv2()
    f._np = np

    # raw ONNX layout is (1, 84, N) -> transpose to (N, 84); build (84, 3)
    cols = np.zeros((84, 3), dtype=np.float32)
    # box0: person 0.9, (cx,cy,w,h)=(50,50,40,80)
    cols[0, 0], cols[1, 0], cols[2, 0], cols[3, 0] = 50, 50, 40, 80
    cols[4, 0] = 0.9
    # box1: cat (class 15) 0.9
    cols[4 + 15, 1] = 0.9
    # box2: person 0.4 (below threshold)
    cols[0, 2], cols[1, 2], cols[2, 2], cols[3, 2] = 200, 200, 20, 20
    cols[4, 2] = 0.4
    out = cols[None]  # (1, 84, 3) -> transpose gives (3, 84)
    f._net = _FakeNet(out)
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    result = f._detect_yolo(img)
    assert result["person_score"] == pytest.approx(0.9)
    assert result["person_boxes"] == pytest.approx(
        [[30 / 640, 10 / 640, 40 / 640, 80 / 640]]
    )
    # normalized bbox area of the kept 40x80 person box on 640x640 input
    assert result["face_bbox_area"] == pytest.approx((40 * 80) / (640 * 640), rel=1e-3)


def test_yolo_e2e_decode_picks_person_and_skips_other_classes():
    f = pf.PersonFilter(threshold=0.3, backend="yolov26n")
    f._cv2 = _FakeCv2()
    f._np = np

    # e2e output: (N, 6) rows of [x1, y1, x2, y2, score, cls]
    # box0: person 0.95 (40x80 px on 640 input), box1: cat 0.9, box2: person 0.2
    out = np.array([
        [10.0, 10.0, 50.0, 90.0, 0.95, 0],
        [300.0, 300.0, 330.0, 330.0, 0.9, 15],
        [100.0, 100.0, 120.0, 120.0, 0.2, 0],
    ], dtype=np.float32)
    # FakeNet wraps output as [1, N, 6]; decoder reads outputs[0]
    f._net = _FakeNet(out.reshape(1, -1, 6))
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    result = f._detect_yolo(img)
    assert result["person_score"] == pytest.approx(0.95)
    assert result["face_bbox_area"] == pytest.approx((40 * 80) / (640 * 640), rel=1e-3)


def test_yolo_decode_runs_for_all_yolo_backends():
    for backend in ("yolov8n", "yolov11n", "yolov26n"):
        f = pf.PersonFilter(backend=backend)
        assert f._is_yolo is True
        assert f._input_size == 640


def test_mobilenet_decode_picks_person_class():
    f = pf.PersonFilter(threshold=0.3, backend="mobilenet_ssd")
    f._cv2 = _FakeCv2()
    f._np = np
    # detections: [1,1,N,7] with (image_id, class_id, confidence, x, y, w, h)
    dets = np.array([[[[0, 8, 0.2, 0, 0, 0, 0],
                       [0, 15, 0.85, 0, 0, 0, 0]]]], dtype=np.float32)
    f._net = _FakeNet(dets)
    img = np.zeros((300, 300, 3), dtype=np.uint8)

    result = f._detect_mobilenet(img)
    assert result["person_score"] == pytest.approx(0.85)
    assert result["face_score"] == 0.0


def test_age_filter_requires_every_person_to_be_confidently_adult():
    f = pf.PersonFilter(adult_threshold=0.9)
    f._cv2 = _FakeCv2()
    f._face_net = _FakeNet(
        np.array([[[[0, 0, 0.95, 0.2, 0.1, 0.4, 0.3]]]], dtype=np.float32)
    )
    f._age_net = _FakeNet(
        np.array([[0.01, 0.01, 0.01, 0.2, 0.2, 0.2, 0.17, 0.2]], dtype=np.float32)
    )
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    result = f._classify_ages(img, [[0.0, 0.0, 1.0, 1.0]])

    assert result["adult_only"] is True
    assert result["matched_person_count"] == 1
    assert result["adult_score"] == pytest.approx(0.97)


def test_age_filter_is_uncertain_when_a_person_has_no_matching_face():
    f = pf.PersonFilter(adult_threshold=0.9)
    f._cv2 = _FakeCv2()
    f._face_net = _FakeNet(
        np.array([[[[0, 0, 0.95, 0.1, 0.1, 0.2, 0.2]]]], dtype=np.float32)
    )
    f._age_net = _FakeNet(
        np.array([[0.01, 0.01, 0.01, 0.2, 0.2, 0.2, 0.17, 0.2]], dtype=np.float32)
    )
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    result = f._classify_ages(
        img,
        [[0.0, 0.0, 0.4, 0.5], [0.6, 0.0, 0.4, 0.5]],
    )

    assert result["adult_only"] is False
    assert result["matched_person_count"] == 1


class _FakeNet:
    def __init__(self, out):
        self._out = out

    def setInput(self, blob):
        pass

    def forward(self):
        return self._out


class _FakeDnn:
    def blobFromImage(self, img, scalefactor=1.0, size=None, mean=0.0,
                      swapRB=False, crop=False):
        return np.zeros((1, 3, *(size or (640, 640))), dtype=np.float32)


class _FakeCv2:
    IMREAD_COLOR = 1
    dnn = _FakeDnn()

    def imdecode(self, buf, flag):
        return np.zeros((480, 640, 3), dtype=np.uint8)
