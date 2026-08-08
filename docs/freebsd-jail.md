# FreeBSD Jail Deployment

This app is meant to run inside a jail as the orchestrator. `ffmpeg` does the RTSP recording and clip cutting; Python handles state, queues, llama.cpp calls, and the web UI.

The runtime path deliberately avoids FastAPI, Pydantic, httpx, and other
packages that can force Rust or native extension builds on FreeBSD. The core
service uses Python's standard library and `ffmpeg`; the enabled-by-default
person filter additionally uses the OpenCV and NumPy packages supplied by
FreeBSD.

## Host clock and camera clock

The jail reads the FreeBSD host clock but should not try to change it. Enable
NTP on the FreeBSD host (not inside the jail):

```sh
sysrc ntpd_enable=YES
sysrc ntpd_sync_on_start=YES
service ntpd start
ntpq -p
date
```

The recorder uses the host wall clock for segment boundaries when
`SEGMENT_AT_CLOCKTIME=true`. The camera's on-screen timestamp is a separate
clock. If it is 23 seconds behind the host, set this in the jail's `.env`:

```text
CAMERA_TIME_OFFSET_SECONDS=-23
```

This changes displayed metadata and directory names only; it does not rewrite
the media PTS or alter relative clip extraction.

## 1. Packages

Install Python, SQLite support, `uv`, `ffmpeg`, and the person-filter runtime
packages in the jail.

```sh
pkg install python311 py311-sqlite3 py311-opencv py311-numpy ffmpeg
```

Python 3.11 or newer is supported. If your jail uses Python 3.12, install the
matching `python312`, `py312-sqlite3`, `py312-opencv`, and `py312-numpy`
packages and use `/usr/local/bin/python3.12` in the `uv venv` commands.

Install `uv` using your preferred FreeBSD package or Python user install. Confirm it is visible to the service user:

```sh
uv --version
ffmpeg -version
ffprobe -version
```

## 2. Storage Layout

Use disk-backed storage, not memory-backed storage.

Recommended mounts:

```text
/usr/local/mine_nas_ai_video       app checkout
/var/db/nas-video                  SQLite database and rolling buffer
/mnt/nextcloud-family-moments      shared output folder visible to Nextcloud
```

Expose `/mnt/nextcloud-family-moments` to Nextcloud as external storage. Avoid writing directly into Nextcloud's internal data directory.

## 3. App Setup

```sh
cd /usr/local/mine_nas_ai_video
uv venv --python /usr/local/bin/python3.11 --system-site-packages
uv sync --no-dev
cp .env.example .env
```

The `--system-site-packages` flag is required for the project environment to
see `py311-opencv` and `py311-numpy` installed by `pkg`. If `.venv` already
exists without that flag, recreate it once:

```sh
uv venv --clear --python /usr/local/bin/python3.11 --system-site-packages
uv sync --no-dev
```

Edit `.env`:

```text
DATA_DIR=/var/db/nas-video
BUFFER_DIR=/var/db/nas-video/buffer
DATABASE_PATH=/var/db/nas-video/app.sqlite3
NEXTCLOUD_OUTPUT_DIR=/mnt/nextcloud-family-moments
RTSP_USERNAME=camera-user
RTSP_PASSWORD=camera-password
RTSP_LOW_URL=rtsp://camera-ip/low-stream
RTSP_HIGH_URL=rtsp://camera-ip/4k-stream
LLAMA_BASE_URL=http://llama-jail-ip:8080/v1
LLAMA_MODEL=Qwen3-VL-2B-Instruct
SEGMENT_AT_CLOCKTIME=true
STREAM_ALIGNMENT_TOLERANCE_SECONDS=2
STREAM_ALIGNMENT_SAMPLE_COUNT=5
CAMERA_TIME_OFFSET_SECONDS=-23
```

For the board-primary RV1106 deployment, the NAS can run high-only. Keep a
small recording margin around the board's 07:00-21:00 detection window:

```text
RTSP_LOW_URL=
ANALYSIS_ENABLED=false
ANALYSIS_BACKEND=daughter_detector
ANALYSIS_STREAM_ROLE=high
RECORD_WINDOW_START=06:58
RECORD_WINDOW_END=21:02
RV1106_PROBABLE_POLICY=verify
RV1106_SAVE_WAIT_SECONDS=180
```

In this mode a blank low URL is expected: MQTT event processing remains active,
confirmed events use the 4K buffer directly, and only probable events load the
NAS detector for a five-frame verification.

If the RTSP password contains special characters, quote it in `.env`, for example `RTSP_PASSWORD="your#complex&password"`.

Run the preflight:

```sh
uv run nas-video-check
```

With `PERSON_FILTER_ENABLED=true`, preflight imports OpenCV, downloads the
configured YOLO model plus the local face/age models, and loads them before
reporting success. The models are stored under `DATA_DIR/person_filter_models`
by default, where the service user can write them. The face/age assets add
roughly 50 MB.

Start manually once:

```sh
uv run nas-video
```

Open `http://jail-ip:8000`.

After both streams have produced at least five stable segments, inspect the
alignment state:

```sh
fetch -qo - http://127.0.0.1:8000/api/health | \
  python3.11 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["stream_alignment"], indent=2))'
```

`stable` means recent low/high segment starts are within the configured
tolerance. `drifted` means recorder or camera timing needs investigation;
final high-stream verification remains enabled in either case.

## 4. rc.d Service

Copy the service template:

```sh
cp deploy/freebsd/nas_video /usr/local/etc/rc.d/nas_video
chmod +x /usr/local/etc/rc.d/nas_video
```

Add to `/etc/rc.conf` inside the jail:

```sh
nas_video_enable="YES"
nas_video_user="nasvideo"
nas_video_chdir="/usr/local/mine_nas_ai_video"
nas_video_uv="/usr/local/bin/uv"
nas_video_run_args="run nas-video"
```

Start it:

```sh
service nas_video start
service nas_video status
```

### llama.cpp server (rc.d)

If the local llama.cpp vision model runs in the same jail (or a
separate jail dedicated to inference), use the bundled rc.d template
to start `llama-server` under `daemon(8)`.

```sh
cp deploy/freebsd/llama_server /usr/local/etc/rc.d/llama_server
chmod +x /usr/local/etc/rc.d/llama_server
```

Add to `/etc/rc.conf` inside that jail:

```sh
llama_server_enable="YES"
llama_server_user="root"
llama_server_bin="/usr/local/bin/llama-server"
llama_server_model="/mnt/models/Qwen3-VL-2B-Instruct-UD-Q4_K_XL.gguf"
llama_server_mmproj="/mnt/models/mmproj-F16.gguf"
llama_server_threads="2"
llama_server_batch="256"
llama_server_ubatch="256"
llama_server_ctx="2048"
llama_server_host="0.0.0.0"
llama_server_port="8892"
```

Start it:

```sh
service llama_server start
service llama_server status
service llama_server stop
```

Point `nas_video` at it (in the recorder jail's `.env`):

```text
LLAMA_BASE_URL=http://llama-jail-ip:8892/v1
```

## 5. Operating Notes

- The low-resolution RTSP stream is analyzed for speed.
- The 4K RTSP stream is kept as the high-quality source for saved clips.
- Default analysis sends one chronological contact sheet to the vision model for each 120-second segment.
- Start with the camera low stream at 640x360. If the model is still too slow, try 352x288 before 320x240.
- Raw buffer files are deleted after `RETENTION_HOURS`.
- Saved clips and JSON metadata in `NEXTCLOUD_OUTPUT_DIR` are not deleted by buffer cleanup.
- If analysis is slow, increase `RETENTION_HOURS` before lowering model quality.
