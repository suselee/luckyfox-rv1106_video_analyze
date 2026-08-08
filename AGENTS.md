# Agent Guide

Compact guidance for working on this repo. `CLAUDE.md` has the deeper architecture/command detail — read both before changing behavior.

## Project

FreeBSD-jail Python service that saves family moments from RTSP cameras. Two supported modes: NAS-side analysis (low-res stream analyzed by a local llama.cpp vision model, clips cut from the 4K stream) and RV1106 board-primary mode (`edge/rv1106/`, C++ service on the board; NAS only keeps the 4K buffer and saves on MQTT events). Selected clips/metadata are written to a Nextcloud-visible folder.

## Tooling

- Python 3.11+ (see `.python-version`); package manager is `uv`.
- Tests: `uv run pytest tests/`; single test: `uv run pytest tests/test_workers.py::test_name`.
- Tests import `nas_video_summarizer` directly (no conftest/PYTHONPATH) because `[tool.uv] package = true` installs it editable — re-run `uv sync` after editing `pyproject.toml`.
- No formatter/linter/typechecker is configured. Do not add one without team discussion.
- `scripts/test_llm_compat.py` is a standalone NAS diagnostic (`python3 scripts/test_llm_compat.py <low-buffer-dir>`), not part of the pytest suite.

## Runtime constraints

- Runtime Python dependencies are intentionally **zero** (stdlib only): the HTTP server, `.env` parser, and llama.cpp client are hand-rolled. Do not add FastAPI, Pydantic, httpx, python-dotenv, etc.
- Sole exception: `opencv`/`numpy` for the **optional** person filter / daughter detector (`requirements-filter.txt`; FreeBSD: `py311-opencv py311-numpy`). Imported lazily with keep-all-frames fallback; without them features and tests degrade/skip, the app still runs.
- External binaries on PATH: `ffmpeg`, `ffprobe`.
- `.env` (copied from `.env.example`) is read by a custom parser in `src/nas_video_summarizer/config.py`; handles quoted values and `export ` prefixes; shell env wins over `.env`. `.env.example` documents every option — check it before guessing a default.

## Entrypoints

- `uv run nas-video` — web UI + background workers (blocks; SIGTERM/SIGINT handling matters for clean ffmpeg shutdown under `daemon(8)`).
- `uv run nas-video-check` — preflight of `.env`, ffmpeg/ffprobe, paths, person filter.
- `uv run nas-video-requeue-board` — one-shot CLI that re-queues board sessions previously rejected for missing 4K coverage.

## Architecture

- `app.py` — stdlib `ThreadingHTTPServer` (dashboard, `/api/*`, range/HEAD video serving) in the main thread; `WorkerRuntime` runs `Supervisor` in a second thread with its own asyncio loop.
- `workers.py` — `Supervisor` loops: low/high recorders, segment scanner, analyzer, cleanup; plus MQTT-triggered board-session saving. `analysis.py` defines the backend-neutral `ClipCandidate` (aliased `AnalysisResult` in `llm.py`).
- Analysis backends (`ANALYSIS_BACKEND`): `vlm` (llama.cpp, default), `daughter_detector` (fully local OpenCV, `daughter_detector.py`), `rv1106` (no NAS inference; MQTT-triggered only).
- `person_filter.py` — optional local OpenCV pre-filter (skips no-person / confidently adult-only segments before any LLM call); downloads model weights on first use.
- `mqtt.py` — hand-rolled stdlib MQTT subscriber (no paho); the NAS is a subscriber, never a broker.
- `archive.py` — rebuilds per-day `manifest.json` / `summary.md` / `_READY.json` markers from the DB.
- `ffmpeg_tools.py` — all video/audio work (recording, sampling, contact sheets, person-filtered frame selection, clip extract/concat).
- `llm.py` — sync `urllib` client to llama.cpp `/chat/completions` (wrapped in `asyncio.to_thread`), base64 images, circuit breaker on repeated timeouts.
- `database.py` — SQLite WAL; idempotent `migrate()` at startup; short-lived connections per call (safe across threads).

## Behavior & config gotchas

- App starts with blank `RTSP_LOW_URL`/`RTSP_HIGH_URL` (recorder disabled) — use this for local UI/DB development.
- `RTSP_USERNAME`/`RTSP_PASSWORD` are merged into RTSP URLs via percent-encoding in `Settings.rtsp_*_url_for_ffmpeg`; URLs already embedding credentials are untouched.
- `ANALYSIS_IMAGE_MODE` default is now `frames` (multi-image); `contact_sheet` is the single-image mode for slow models.
- `ANALYSIS_STREAM_ROLE=high` analyzes the 4K stream directly and skips the low→high cross-reference.
- `CLIP_VIDEO_CODEC=copy` (default) preserves HEVC that browsers can't play; set `libx264`/`aac` for browser playback.
- Output layout in `NEXTCLOUD_OUTPUT_DIR`: `{YYYY-MM-DD}/{HHMMSS}_{slug}.mp4` + `.json`, plus per-day `summary.md`, `manifest.json`, and `_READY.json` (day complete). NAS never overwrites `analysis/`, `diary.json`, `diary.md`.
- Buffer segments are deleted after `RETENTION_HOURS`; saved clips/metadata are never auto-cleaned.
- Moment pipeline: keep threshold + confidence gate, `MOMENT_COOLDOWN_SECONDS`, per-period/per-day keep-best-N caps, verification frames before 4K clip extraction, atomic publication.
- Raw root-level `check*_*.onnx` files are gitignored scratch artifacts (person-filter model experiments) — ignore them.

## Deployment

- FreeBSD jail; see `docs/freebsd-jail.md` and `deploy/freebsd/` (`nas_video` and `llama_server` rc.d templates, mosquitto config).
- Board-primary mode: `docs/rv1106-mqtt.md`; local detector: `docs/daughter-detector.md`; tuning: `docs/tuning.md`, `docs/person-filter.md`.
