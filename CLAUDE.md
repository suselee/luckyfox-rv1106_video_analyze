# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` is the compact contributor guide; this file adds the architecture and command detail that needs multiple files to understand. Read both.

The repo holds **two codebases** that talk to each other: the Python NAS service (`src/nas_video_summarizer/`, FreeBSD jail) and the RV1106 edge board service (`edge/rv1106/board_service/`, C++ cross-compiled for an ARM camera board). Changing the clip/event contract usually means touching both.

## Commands

### NAS (Python)

```sh
uv sync --no-dev                      # install runtime (dev group is just pytest)
uv sync                               # include pytest
uv run pytest tests/                  # whole suite
uv run pytest tests/test_workers.py::test_append_daily_summary   # one test
uv run nas-video                      # web UI + background workers (blocks)
uv run nas-video-check                # preflight: .env, ffmpeg/ffprobe, paths, person filter
uv run nas-video-requeue-board        # re-queue board sessions rejected for missing 4K coverage
```

- Tests import `nas_video_summarizer` directly (no `conftest.py`, no `PYTHONPATH` tricks); this works because `[tool.uv] package = true` installs the package editable during `uv sync`. After editing `pyproject.toml` scripts/packaging, re-run `uv sync`.
- **No formatter, linter, or typechecker is configured. Do not add one without team discussion** (see `AGENTS.md`).
- Person-filter / daughter-detector tests need `opencv` + `numpy`: `pip install -r requirements-filter.txt` locally, or system `py311-opencv`/`py311-numpy` + `uv venv --system-site-packages` on FreeBSD. Without them the features and their tests degrade/skip rather than fail the app.
- `scripts/test_llm_compat.py` and `scripts/export_daughter_training_frames.py` are standalone NAS-side diagnostics, not part of the pytest suite.

### Edge board (C++)

```sh
make -C edge/rv1106/board_service              # cross-compile all board binaries
make -C edge/rv1106/board_service install      # stage self-contained install/ dir to copy to the board
```

`TOOLCHAIN_DIR`, `LUCKFOX_SDK_DIR`, `RKNN_SDK_DIR`, `FACE_MODEL_DIR` are make-overridable and default to paths on the maintainer's machine — vendor SDKs and `.rknn` models are deliberately **not** committed, so a clean checkout cannot build the board binaries without them.

Host-side tests are pure-logic files under `board_service/tests/` with **no board dependency and no make target**; each file's header comment carries its exact compile line, e.g.

```sh
cd edge/rv1106/board_service
g++ -std=c++11 -Wall -Isrc tests/nal_test.cpp src/nal_stats.cpp -o /tmp/nal_test && /tmp/nal_test
g++ -std=c++11 -Wall -Isrc tests/audio_test.cpp src/audio_util.cpp src/audio_ring.cpp -o /tmp/audio_test && /tmp/audio_test
```

`track_fusion_test` and `schedule_test` are cross-compiled make targets instead (they run on the board, and `install_on_board.sh` runs them as a pre-install gate). Board `.o` files and built binaries are gitignored.

## Hard constraint: zero runtime dependencies (NAS)

`pyproject.toml` `dependencies = []` is intentional and load-bearing. The target is a FreeBSD jail where Rust/native builds (Pydantic, httpx) are painful, so the HTTP server, llama.cpp client, MQTT subscriber, multipart parser, and `.env` parser are **all hand-rolled on the standard library**. Do not add FastAPI, Pydantic, httpx, requests, paho-mqtt, python-dotenv, etc. `opencv`/`numpy` are the sole exception, and only for the optional person filter and daughter detector (imported lazily, always with a degraded fallback). `ffmpeg` and `ffprobe` are required external binaries on PATH and do all video/audio work.

The board service has the matching constraint: no ffmpeg, no libcurl, no MQTT library. RTSP client, NAL scanner, MQTT publisher, and multipart HTTP uploader are hand-rolled C++11 against the vendor SDK only.

## NAS architecture

Single process, two concerns split across threads:

- **HTTP thread (main):** `app.py` runs a stdlib `ThreadingHTTPServer`. `RequestHandler` serves the vanilla-JS dashboard (`static/`), a JSON API (`/api/health`, `/api/moments`, favorite/delete), moment videos with HTTP range/HEAD support, and `POST /api/ingest` for board-uploaded clips.
- **Worker thread:** `app.py:WorkerRuntime` spins up a second thread with its own asyncio event loop running `workers.py:Supervisor`. `run()` installs SIGTERM/SIGINT handlers that call `server.shutdown()` so the `finally` block can cancel worker tasks and terminate child ffmpeg processes — otherwise `daemon(8)` would orphan the recorders (see the long comment in `app.py:run`).

`Supervisor` owns concurrent asyncio loops, each writing status into `self.state` (surfaced at `/api/health`): recorders (one per low/high stream), scanner, analyzer, cleanup, day-archive, plus `_mqtt_loop`/`_board_events_loop` and `_board_ingest_loop` for edge modes.

### Three analysis backends (`ANALYSIS_BACKEND`)

The same save/quota/archive machinery is shared; only the "is this my daughter?" decision differs.

1. **`vlm`** (default) — llama.cpp vision model over sampled low-stream frames.
2. **`daughter_detector`** — fully local OpenCV (`daughter_detector.py`); no LLM calls.
3. **`rv1106`** — no NAS inference at all. The board decides; the NAS only reacts to MQTT events or ingested clips.

### vlm/detector cascade (`workers.py`)

For each pending segment (`analysis_stream_role`, default `low`), in `_analyze_segment` → keep logic in `_analyzer_loop` → `_save_moment`:

1. Sample frames (motion-aware or even) → drop near-black frames → optional person filter (`filter_frames_by_person_detection` skips no-person and confidently adult-only segments locally, before any LLM call).
2. Build N `frames` or one `contact_sheet` and call `llm.py:LlamaAnalyzer.analyze` → `ClipCandidate` (`analysis.py`, aliased `AnalysisResult` in `llm.py`). Default `ANALYSIS_IMAGE_MODE=frames`; `contact_sheet` is the single-image mode for slow models.
3. Keep decision: `should_save(MOMENT_KEEP_THRESHOLD)` requires `keep` **and** `confidence >= threshold`, with a narrow "keep-consistency repair" for small VLMs that return `keep=false` while their text + local child evidence clearly describe the child.
4. Gating: `MOMENT_COOLDOWN_SECONDS`, then per-period and per-day keep-best-N caps. Evictions are planned but only applied after the new clip is verified and registered.
5. `_save_moment`: collect matching 4K (`high`) segments by time overlap, verify the low-stream candidate, extract a staged high-stream clip on the output filesystem, verify that final clip, atomically publish, then write metadata/summary and insert a `moments` row.

Deliberately **high-recall** while fail-closing final publication: candidate and final-clip verification both require visible-child evidence. Consecutive llama timeouts trip a circuit breaker (`LLAMA_CIRCUIT_BREAKER_*`); a single timeout can fall back to retrying frames as one contact sheet (`LLAMA_TIMEOUT_FALLBACK`).

### Board-primary paths (two of them — know which one you're in)

**MQTT path** (older, `docs/rv1106-mqtt.md`): board publishes confirmed/probable identity sessions; NAS records only 4K and cuts clips itself. `_handle_mqtt_message` accumulates sessions, `_expire_stale_board_sessions` finalizes ones whose `end` packet was lost (`RV1106_SESSION_TIMEOUT_SECONDS`), `_save_board_session` resolves `best_ts` against high-stream segments, and `RV1106_PROBABLE_POLICY` (`verify`/`reject`/`accept`) decides whether probable events get a strict NAS-side frame verification.

**HTTP ingest path** (current target, `docs/board-ingest.md`): the board holds the 4K stream in an in-memory ring, cuts the clip itself, and POSTs elementary streams to `/api/ingest`. The NAS never touches RTSP. `app._handle_board_ingest` parses multipart → validates meta → dedupes on `ingest_event_key()` → spools to `BOARD_INGEST_DIR/pending/<key>/` and returns 202. `_board_ingest_loop` then remuxes with `-c copy` into the Nextcloud layout and inserts a moment with `analysis_backend=rv1106_edge`. Idempotency lives in the `ingested_clips` table; failures move to `failed/` after `BOARD_INGEST_MAX_ATTEMPTS`.

During cutover both paths can run in parallel and produce duplicate clips; `docs/board-ingest.md` describes the switch-off order.

### Supporting modules

- `config.py` — hand-written `.env` loader (quotes, `export `; shell env wins over `.env`) and the frozen `Settings` dataclass that is the single source of every tunable. RTSP credentials are merged into URLs via percent-encoding in `rtsp_low_url_for_ffmpeg`/`rtsp_high_url_for_ffmpeg`; URLs already embedding credentials are left alone. `.env.example` documents every option — check it before guessing a default.
- `database.py` — SQLite WAL + 30s busy timeout, idempotent `migrate()` at startup, tables `segments`/`moments`/`events`/`ingested_clips`. Every method opens a short-lived connection (safe across the worker thread + HTTP threads).
- `llm.py` — synchronous `urllib` POST to the OpenAI-compatible `/chat/completions` (wrapped in `asyncio.to_thread`), base64 images, `response_format=json_object`, tolerant JSON extraction, offset-snapping to real sampled-frame timestamps.
- `ffmpeg_tools.py` — every ffmpeg/ffprobe invocation: recorder command, frame sampling (incl. `motion_aware` scene detection), contact-sheet compositing (`xstack`), clip extract/concat, elementary-stream remux for ingested clips, person-filter frame selection.
- `mqtt.py` — hand-rolled stdlib MQTT **subscriber** (the NAS is never a broker).
- `ingest.py` — multipart parser, filesystem spool queue, `save_ingested_clip`.
- `archive.py` — rebuilds per-day `manifest.json`/`summary.md`/`_READY.json` from the DB.
- `person_filter.py` / `daughter_detector.py` — lazily-loaded OpenCV DNN person + face/age detection; `person_filter` downloads model weights on first use.

## Edge board architecture (`edge/rv1106/board_service/`)

C++11, no dynamic allocation churn, targeted at a **185 MB single-core board** — memory ceilings in `config.example.ini` comments are real operating limits, not suggestions. Two independent RTSP connections:

**Low stream (analysis)** — `h264_source` (raw RTSP client) → `mpp_decoder` (RockIt software VDEC; RV1106 has no VDEC hardware block, so `keyframes_only=true` keeps CPU ~10-20% instead of 90%+) → `rockiva_detector` (RockIVA PFP person/face detection + track IDs at ~1 FPS) → `track_fusion` (child-vs-adult sizing, probable/confirmed session state machine) → on-demand `face_detect` (RetinaFace) + `face_recog` (MobileFaceNet) against `facedb` → `FusionEvent`.

**High stream (4K, evidence)** — `high_stream` runs two pthreads: `feed_loop` pulls 4K NALs into `video_ring` (in-memory, never decoded, never written to disk) plus optional audio into `audio_ring`; `upload_loop` drains a bounded retry queue of clips cut by `ring.cut()` (keyframe-aligned) and POSTs them via `http_uploader` as streaming multipart.

`schedule` gates everything to a local time window (default 07:00–21:00 at a fixed UTC offset — the board has no tzdata); outside it the process stays alive with `pipeline=sleeping` and MQTT heartbeat only. `system_monitor` enforces the `[guard]` thresholds. `mqtt_publisher` is the legacy event path.

Diagnostic binaries built alongside the service: `rockiva_probe` (live detection latency/CPU on the board — run before installing a new binary), `high_stream_probe` (4K stream parameters, no MPP/RKNN link), `decode_test`, `enroll` (build the face database).

`deploy/install_on_board.sh` snapshots the current binary/config/db, runs the schedule + fusion tests and the live RockIVA probe, and only swaps the production binary if those pass; `rollback_on_board.sh` is installed persistently at `/root/daughter_watch/rollback_on_board.sh`.

## Configuration & behavior notes

- The app **starts with blank `RTSP_LOW_URL`/`RTSP_HIGH_URL`** (recorder disabled) — use this for local UI/DB/analyzer development.
- `ANALYSIS_STREAM_ROLE=high` analyzes the 4K stream directly and skips the low→high cross-reference in `_save_moment`. `ANALYSIS_ENABLED=false` stops the continuous analyzer without disabling MQTT/ingest event handling.
- Saved clips default to `CLIP_VIDEO_CODEC=copy` — lossless/fast but may emit HEVC that browsers can't play; set `CLIP_VIDEO_CODEC=libx264` + `CLIP_AUDIO_CODEC=aac` for in-browser playback.
- Output layout in `NEXTCLOUD_OUTPUT_DIR`: `{YYYY-MM-DD}/{HHMMSS}_{slug}.mp4` + `.json`, plus per-day `summary.md`, `manifest.json`, `_READY.json`. The NAS never overwrites `analysis/`, `diary.json`, or `diary.md` — those belong to a downstream desktop worker.
- Buffer segments are deleted after `RETENTION_HOURS`; saved clips/metadata are never auto-deleted.
- Root-level `check*_*.onnx` files are gitignored scratch artifacts from person-filter model experiments — ignore them.
- Deployment is a FreeBSD jail under `daemon(8)`: `docs/freebsd-jail.md` and `deploy/freebsd/` (`nas_video`, `llama_server` rc.d templates, mosquitto config). Tuning guides: `docs/tuning.md`, `docs/person-filter.md`, `docs/daughter-detector.md`, `docs/architecture.md`.

## Conventions

Code comments, commit messages, and the `docs/` prose are predominantly **Chinese**; identifiers, config keys, and log strings are English. Match the surrounding file. Commit subjects follow `area: 描述` (e.g. `edge: ...`, `NAS: ...`, `edge+NAS: ...`).
