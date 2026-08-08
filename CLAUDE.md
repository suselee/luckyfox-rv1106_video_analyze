# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` contains a compact contributor guide; this file adds the architecture and command detail that needs multiple files to understand. Read both.

## Commands

```sh
uv sync --no-dev                      # install runtime (dev group is just pytest)
uv sync                               # include pytest for running tests
uv run pytest tests/                  # run the whole suite
uv run pytest tests/test_workers.py::test_append_daily_summary   # run one test
uv run nas-video                      # start web UI + background workers (blocks)
uv run nas-video-check                # preflight: .env, ffmpeg/ffprobe, paths, person filter
```

- Tests import `nas_video_summarizer` directly (no `conftest.py`, no `PYTHONPATH` tricks); they work because `[tool.uv] package = true` installs the package editable during `uv sync`. After editing `pyproject.toml` scripts/packaging, re-run `uv sync`.
- **No formatter, linter, or typechecker is configured. Do not add one without team discussion** (see `AGENTS.md`).
- Person-filter tests/features need `opencv` + `numpy`: `pip install -r requirements-filter.txt` locally, or system `py311-opencv`/`py311-numpy` + `uv venv --system-site-packages` on FreeBSD. Without them the filter and its tests degrade/skip rather than fail the app.
- `scripts/test_llm_compat.py` is a standalone NAS-side diagnostic (`python3 scripts/test_llm_compat.py <low-buffer-dir>`), not part of the pytest suite.

## Hard constraint: zero runtime dependencies

`pyproject.toml` `dependencies = []` is intentional and load-bearing. The target is a FreeBSD jail where Rust/native builds (Pydantic, httpx) are painful, so the HTTP server, llama.cpp client, and `.env` parser are all hand-rolled on the standard library. **Do not add FastAPI, Pydantic, httpx, requests, python-dotenv, etc.** `opencv`/`numpy` are the sole exception, and only for the optional person filter (imported lazily, always with a keep-all-frames fallback). `ffmpeg` and `ffprobe` are required external binaries on PATH and do all video work.

## Architecture

Single process, two concerns split across threads:

- **HTTP thread (main):** `app.py` runs a stdlib `ThreadingHTTPServer`. `RequestHandler` serves the vanilla-JS dashboard (`static/`), a JSON API (`/api/health`, `/api/moments`, favorite/delete), and moment videos with HTTP range/HEAD support for browser seeking.
- **Worker thread:** `app.py:WorkerRuntime` spins up a second thread with its own asyncio event loop and runs `workers.py:Supervisor`. `run()` installs SIGTERM/SIGINT handlers that call `server.shutdown()` so the `finally` block can cancel worker tasks and terminate child ffmpeg processes — otherwise `daemon(8)` would orphan the recorders (see the long comment in `app.py:run`).

`Supervisor` owns these concurrent asyncio loops, each writing status into `self.state` (surfaced at `/api/health`):

1. **Recorders** (`_recorder_loop`, one per low/high stream) — spawn `ffmpeg` to segment the RTSP stream into `BUFFER_DIR/<camera>/<role>/<camera>_<role>_<YYYYMMDDTHHMMSS>.mp4`. Disabled when the URL is blank; pause outside `RECORD_WINDOW_*`.
2. **Scanner** (`_scan_loop`) — discovers stable segment files and upserts them into the SQLite `segments` table.
3. **Analyzer** (`_analyzer_loop`) — the core pipeline; see below.
4. **Cleanup** (`_cleanup_loop`) — deletes buffer segments older than `RETENTION_HOURS`. Saved clips/metadata are never auto-deleted.

### Analyzer decision cascade (`workers.py`)

For each pending segment (`analysis_stream_role`, default `low`), in `_analyze_segment` → keep logic in `_analyzer_loop` → `_save_moment`:

1. Sample frames (motion-aware or even) → drop near-black frames (`filter_out_blank_frames`) → optional person filter (`filter_frames_by_person_detection`: skips no-person and confidently adult-only segments locally, before any LLM call).
2. Build either N `frames` or one `contact_sheet` and call `llm.py:LlamaAnalyzer.analyze` → `AnalysisResult` (the current default is `frames`).
3. Keep decision: `result.should_save(MOMENT_KEEP_THRESHOLD)` requires `keep` **and** `confidence >= threshold`, with a narrow "keep-consistency repair" for small VLMs that return `keep=false` while their text + local child evidence clearly describe the child (`AnalysisResult.keep_consistency_repaired`).
4. Gating before save: `MOMENT_COOLDOWN_SECONDS`, then per-period and per-day keep-best-N caps. Evictions are planned but only applied after the new clip is verified and registered successfully.
5. `_save_moment`: collect all matching 4K (`high`) segments by time overlap, verify the low-stream candidate, extract a staged high-stream clip on the output filesystem, verify that final clip, atomically publish it, then write metadata/summary and insert a `moments` row.

The design is deliberately **high-recall** while fail-closing final-source publication: candidate and final-clip verification both require visible-child evidence. Resilience: consecutive llama timeouts trip a circuit breaker (`LLAMA_CIRCUIT_BREAKER_*`); a single timeout can fall back to retrying frames as one contact sheet (`LLAMA_TIMEOUT_FALLBACK`).

### Supporting modules

- `config.py` — hand-written `.env` loader (handles quotes and `export`; shell env wins over `.env`) and the frozen `Settings` dataclass that is the single source of every tunable. RTSP credentials are merged into URLs via percent-encoding in `rtsp_low_url_for_ffmpeg` / `rtsp_high_url_for_ffmpeg`; URLs that already embed credentials are left alone.
- `database.py` — SQLite with WAL + 30s busy timeout, idempotent `migrate()` run at startup, tables `segments` / `moments` / `events`. Every method opens a short-lived connection (safe across the worker thread + HTTP threads).
- `llm.py` — synchronous `urllib` POST to the OpenAI-compatible `/chat/completions` (wrapped in `asyncio.to_thread`), base64 images, `response_format=json_object`, tolerant JSON extraction, and offset-snapping to real sampled-frame timestamps.
- `ffmpeg_tools.py` — every ffmpeg/ffprobe invocation: recorder command, frame sampling (incl. `motion_aware` scene detection), contact-sheet compositing (`xstack`), clip extraction (`-c copy` fast path vs. re-encode for browser-playable H.264), and the person-filter frame selection.
- `person_filter.py` — lazily-loaded OpenCV DNN person detection (YOLO family or MobileNet-SSD) plus a face/age model for adult-only skipping; downloads model weights on first use.

## Configuration & behavior notes

- The app **starts with blank `RTSP_LOW_URL`/`RTSP_HIGH_URL`** (recorder disabled) — use this for local UI/DB/analyzer development. Full env reference: `.env.example`; tuning guides: `docs/tuning.md`, `docs/person-filter.md`.
- Setting `ANALYSIS_STREAM_ROLE=high` analyzes the 4K stream directly and skips the low→high cross-reference in `_save_moment`.
- Saved clips default to `CLIP_VIDEO_CODEC=copy`, which is lossless/fast but may emit HEVC that browsers can't play; set `CLIP_VIDEO_CODEC=libx264` + `CLIP_AUDIO_CODEC=aac` for in-browser playback.
- Deployment is a FreeBSD jail under `daemon(8)`; see `docs/freebsd-jail.md` and the `deploy/freebsd/` rc.d templates (`nas_video`, `llama_server`).
