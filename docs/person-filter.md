# Person Filter — Local Deployment

The person detection pre-filter runs **inline on the NAS** (no separate
service, no HTTP). Sampled frames are scored with local OpenCV DNN models so
the NAS can skip empty-room and confidently adult-only segments without
calling the LLM.

## Backends

All models run via OpenCV DNN, so the only runtime packages are `opencv` and
`numpy` — no `onnxruntime` / `torch` needed.

| Backend          | Model            | Accuracy | Weight  | Notes                                   |
|------------------|------------------|----------|---------|-----------------------------------------|
| `yolov11n` (default) | `yolo11n.onnx` | High     | ~5.4 MB | YOLOv11n, fastest + most accurate on CPU. COCO person class 0. |
| `yolov26n`       | `yolo26n.onnx`   | Highest  | ~10 MB  | YOLO26n, ~30% faster than v11n on CPU, more accurate. **Optional** — see caveat below. |
| `yolov8n`        | `yolov8n.onnx`   | High     | ~12 MB  | Older but well-proven YOLOv8n.         |
| `mobilenet_ssd`  | Caffe sample     | Lower    | ~22 MB  | Older model, more false positives.     |

When person filtering is enabled, two additional models are loaded:

| Purpose | Model | Weight | Notes |
|---------|-------|--------|-------|
| Face detection | OpenCV ResNet-10 SSD | ~3 MB | Matches visible faces to YOLO person boxes. |
| Age classification | Levi-Hassner age model | ~45 MB | Produces eight coarse age buckets. |

The YOLO ONNX export is downloaded automatically during `nas-video-check` or
the first analyzed segment to `DATA_DIR/person_filter_models/`. To use a custom
export (e.g. a different YOLO variant or a quantized build), set
`PERSON_FILTER_MODEL_URL` to its download URL — it overrides the built-in
source.  The YOLO decoder auto-detects the export layout (raw `one-to-many`
head `(1, nc+4, N)` **or** end-to-end `one-to-one` head `(1, 300, 6)`), so
both work without config changes.

### `yolov26n` caveat (OpenCV version)

YOLO26 ONNX is best loaded by **OpenCV 5**.  On **OpenCV 4.x** (what
`py311-opencv` currently ships on FreeBSD) the ONNX graph may fail to load
or produce wrong output.  If the filter fails to initialize, the pipeline
**falls back to keeping all frames** (person pre-filter becomes a no-op) —
analysis still runs, only without the empty-room skip.  To use `yolov26n`
safely:

1. Check the OpenCV version on the NAS: `python -c "import cv2; print(cv2.__version__)"`.
2. If it is 5.x, set `PERSON_FILTER_BACKEND=yolov26n` and verify a run.
3. If it is 4.x, either stay on `yolov11n` (proven), upgrade to OpenCV 5,
   or install `onnxruntime` and switch the loader (breaks the zero-extra-dep
   rule — discuss first).

## Prerequisites (FreeBSD)

```bash
pkg install py311-opencv py311-numpy
uv venv --clear --python /usr/local/bin/python3.11 --system-site-packages
uv sync --no-dev
```

FreeBSD `pkg` installs OpenCV and NumPy into the system Python's
`site-packages`. A normal isolated virtual environment cannot import them;
`--system-site-packages` makes those packages visible to `uv run`.

`opencv` must be >= 4.7 for YOLOv8 ONNX support (4.8+ recommended).

For non-FreeBSD / local development you can instead pip-install:

```bash
pip install -r requirements-filter.txt
```

Run `uv run nas-video-check` to verify the imports and download/load the model
before starting the service. A fresh setup downloads roughly 50 MB in addition
to the selected YOLO model.

## Configuration (.env)

```env
PERSON_FILTER_ENABLED=true
PERSON_FILTER_BACKEND=yolov11n
PERSON_FILTER_MODEL_DIR=/var/db/nas-video/person_filter_models
PERSON_FILTER_THRESHOLD=0.3
PERSON_FILTER_FACE_THRESHOLD=0.7
PERSON_FILTER_ADULT_THRESHOLD=0.9
PERSON_FILTER_CHILD_THRESHOLD=0.6
PERSON_FILTER_SAMPLE_COUNT=12
```

- `PERSON_FILTER_SAMPLE_COUNT=12` extracts 12 frames per segment and scores
  them. After scoring, the top `SAMPLE_FRAME_COUNT` (default 6) frames are
  selected for LLM analysis.
- If **every** frame has `person_score < PERSON_FILTER_THRESHOLD`, the
  segment is skipped (no LLM call) and a `person-filter-skip` event is
  logged.
- If every detected person has a matched face and every matched face has
  aggregated adult probability >= `PERSON_FILTER_ADULT_THRESHOLD`, the
  segment is skipped with an `adult-only-filter-skip` event.
- Age buckets `0-2`, `4-6`, and `8-12` are treated as child evidence. A
  hidden face, unmatched person, uncertain age, or possible child keeps the
  segment for LLM analysis. Child-likely and uncertain frames are selected
  ahead of confidently adult frames.
- `PERSON_FILTER_CHILD_THRESHOLD` is the minimum local child probability
  required before text-based keep repair or uncertain post-save verification
  can preserve a clip.
- The same pre-filter runs before both `frames` and `contact_sheet` analysis.
- `/api/health` exposes the latest `workers.prefilter` status, elapsed seconds,
  and input/output frame counts for CPU tuning on the NAS.
- If `opencv` is not installed or detection fails, the filter falls back to
  keeping all frames so analysis still runs.
