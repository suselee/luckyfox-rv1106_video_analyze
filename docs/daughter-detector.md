# Daughter Detector Backend

Set `ANALYSIS_BACKEND=daughter_detector` to remove llama.cpp from the realtime
NAS pipeline. The low stream is sampled chronologically in one ffmpeg process,
OpenCV scores the frames, temporal detections become events, and the aligned
high stream supplies the published video.

## First deployment

```env
ANALYSIS_BACKEND=daughter_detector
DAUGHTER_DETECTOR_MODE=heuristic
DAUGHTER_SCAN_FPS=0.5
DAUGHTER_EVENT_MIN_HITS=2
DAUGHTER_EVENT_MAX_GAP_SECONDS=6
DAUGHTER_EVENT_MIN_SECONDS=4
```

Heuristic mode only accepts explicit child evidence from the existing face/age
models as its strongest signal. In multi-person scenes it can also use a
conservative relative-body-size fallback: a person must be substantially
shorter and smaller than the largest person across the sampled event. This
helps when the toddler is turned away or partially occluded, while similar-size
adults do not pass. Use heuristic mode to validate the detector-only archive
before training the final model.

## Custom ONNX model

Export training frames on a machine that can access copied buffer files:

```sh
python scripts/export_daughter_training_frames.py \
  --input copied-low-buffer --output daughter-dataset/images --every-seconds 10
```

Label only the daughter as class `0` / `daughter`. Include adult-only, empty,
different-clothing, distant, occluded, seated, and back-facing examples. Split
training and validation by recording date, not by randomly mixing neighboring
frames. A useful first dataset is 500-1000 varied images.

Train YOLO11n or YOLOv8n on the NVIDIA desktop, export a fixed 416px ONNX model
compatible with OpenCV, copy it to the NAS, then configure:

```env
DAUGHTER_DETECTOR_MODE=onnx
DAUGHTER_DETECTOR_MODEL_PATH=/mnt/models/daughter-yolo11n-416.onnx
DAUGHTER_DETECTOR_INPUT_SIZE=416
DAUGHTER_DETECTOR_THRESHOLD=0.45
```

Run `uv run nas-video-check` before restarting the service. ONNX mode fails
closed if the model cannot be loaded; it never silently saves generic people.

## Nextcloud contract

The NAS owns MP4 files, matching JSON, `summary.md`, `manifest.json`, and
`_READY.json`. Detector JSON uses `schema_version: 2` and records backend,
category, identity confidence, selection score, timestamps, and raw detector
metrics. The factual categories are `active`, `multi_person`, and `quiet`.

`_READY.json` appears only after the recording window closes and all indexed
segments for that day reach a final state. A desktop diary worker must verify
the sizes listed in `manifest.json` before processing because Nextcloud may
sync the small READY file before the videos. Desktop-owned results belong in
`analysis/`, `diary.json`, and `diary.md` and are never overwritten by the NAS.
