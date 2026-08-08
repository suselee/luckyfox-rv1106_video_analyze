# Mine NAS AI Video

Local FreeBSD-jail service for saving meaningful moments from a home RTSP camera.

The service can record low-resolution and 4K RTSP streams for NAS-side analysis,
or run in the recommended RV1106 board-primary mode where the NAS records only
the 4K quality source. Selected clips and metadata are written into a
Nextcloud-visible folder.

For slow local vision models such as Qwen3-VL-2B in llama.cpp, the default analyzer sends one low-resolution contact sheet per segment instead of many separate images. This keeps model calls bounded while still showing several moments in chronological order.

## Quick Start

```sh
pkg install python311 py311-sqlite3 py311-opencv py311-numpy ffmpeg
uv venv --python /usr/local/bin/python3.11 --system-site-packages
uv sync --no-dev
cp .env.example .env
uv run nas-video-check
uv run nas-video
```

Open `http://NAS-JAIL-IP:8000`.

The app starts even when RTSP URLs are blank. Fill `.env` with the camera URLs and llama.cpp endpoint before enabling real recording.

## Configuration

Runtime settings live in `.env`.

Important values:

- `RTSP_LOW_URL`: the camera's small stream for NAS-side analysis; leave blank
  in board-primary high-only mode.
- `RTSP_HIGH_URL`: the camera's 4K stream, for saved clips.
- `RTSP_USERNAME` / `RTSP_PASSWORD`: optional shared credentials for both RTSP streams.
- `NEXTCLOUD_OUTPUT_DIR`: a folder mounted into this jail and exposed to Nextcloud.
- `ANALYSIS_BACKEND`: `vlm`, `daughter_detector`, or `rv1106`; edge-only mode runs no NAS vision inference.
- `DAUGHTER_DETECTOR_MODE`: `heuristic` initially, or `onnx` for a trained one-class daughter model.
- `MQTT_ENABLED`: subscribe to RV1106 daughter identity hits; requires a separate MQTT broker.
- `RV1106_SESSION_TIMEOUT_SECONDS`: finalize an edge session when its `end` packet is lost.
- `RV1106_PROBABLE_POLICY`: `verify`, `reject`, or `accept` for board events
  without face confirmation.
- `RV1106_SAVE_WAIT_SECONDS`: maximum wait for complete 4K source coverage.
- `LLAMA_BASE_URL`: your llama.cpp jail's OpenAI-compatible base URL, usually ending in `/v1`.
- `LLAMA_MODEL`: the multimodal model name served by llama.cpp.
- `RETENTION_HOURS`: how long raw rolling-buffer segments remain on disk.
- `MAX_MOMENT_SECONDS`: maximum saved clip length after adding before/after context.
- `MAX_MOMENTS_PER_PERIOD`: optional keep-best-N limit applied separately to
  configurable morning, afternoon, and evening ranges.
- `ANALYSIS_IMAGE_MODE`: `contact_sheet` by default; use `frames` only when the model is fast enough.
- `ANALYSIS_FRAME_WIDTH`: width of each sampled frame before it is sent or placed into the contact sheet.
- `PERSON_FILTER_ENABLED`: locally skips frames with no person and segments
  where every visible person is confidently classified as an adult.

## Camera Stream Advice

For NAS-side VLM/detector analysis, keep the camera's low stream at `640x360`
for the first jail test. In board-primary mode the board consumes that stream
and the NAS may leave `RTSP_LOW_URL` blank.

For cameras that require authentication, prefer this form:

```env
RTSP_USERNAME=your-camera-user
RTSP_PASSWORD=your-camera-password
RTSP_LOW_URL=rtsp://camera-ip/low-stream
RTSP_HIGH_URL=rtsp://camera-ip/4k-stream
```

The app will pass `rtsp://user:password@camera-ip/...` to `ffmpeg` internally, but preflight and the dashboard only show redacted URLs. Direct URLs like `rtsp://user:password@camera-ip/...` still work if your camera vendor documents them that way.

If the password contains spaces, `#`, `&`, or other shell-like characters, quote it in `.env`:

```env
RTSP_PASSWORD="your#complex&password"
```

Recommended default for Qwen3-VL-2B in llama.cpp:

```env
SEGMENT_SECONDS=120
ANALYSIS_IMAGE_MODE=contact_sheet
ANALYSIS_FRAME_WIDTH=320
CONTACT_SHEET_COLUMNS=2
SAMPLE_FRAME_COUNT=4
SAMPLE_EVERY_SECONDS=30
RETENTION_HOURS=168
```

If analysis still falls behind badly, try the camera low stream at `352x288` and set `ANALYSIS_FRAME_WIDTH=320`. Use `320x240` only after confirming the model can still recognize the important actions.

## Design Notes

- Python runs orchestration, state, UI, and HTTP model calls.
- Python 3.11 or newer is supported; Python 3.12 is not required.
- Runtime Python dependencies are intentionally zero; the web server and llama.cpp HTTP client use the Python standard library to avoid Pydantic/Rust builds on FreeBSD.
- Detector mode uses the existing FreeBSD OpenCV/NumPy system packages and
  extracts its chronological frame series in one ffmpeg process.
- `ffmpeg` does all RTSP recording, segmentation, frame extraction, and clip concatenation.
- The rolling buffer is stored on disk, not in memory.
- Analysis is deliberately high-recall: low-confidence but promising moments are saved instead of discarded.
- Each day directory has NAS-owned MP4/JSON files, a factual `summary.md`, an
  atomic `manifest.json`, and a `_READY.json` marker once the day is complete.
  A future desktop worker can safely write `analysis/`, `diary.json`, and
  `diary.md`; the NAS does not overwrite those files.

## FreeBSD Service Sketch

See [docs/freebsd-jail.md](docs/freebsd-jail.md) for jail setup, storage layout, and an `rc.d` template.
See [docs/rv1106-mqtt.md](docs/rv1106-mqtt.md) for Mosquitto and durable edge-triggered clip saving.
The versioned RV1106 C++ service, performance probe, and deployment scripts are
under [`edge/rv1106/`](edge/rv1106/README.md). The fusion build uses RockIVA
person tracking plus on-demand face identity and publishes confirmed/probable
MQTT sessions without storing video on the board.
