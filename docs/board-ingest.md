# 板端 4K 剪辑 HTTP 上传 (board-primary 模式)

最终形态: 所有 4K 剪辑在 RV1106 板端完成 —— 板把 4K 主码流拉进内存环形
缓冲 (不落盘、不解码), 融合管线确认女儿事件后直接从缓冲切片, 以
`multipart/form-data` POST 到 NAS, NAS 只做 `-c copy` 重封装并发布到
Nextcloud 布局。本页描述 NAS 侧 `/api/ingest` 端点与板端 `[high]`/`[upload]`
配置之间的关系。

## 数据流

```
RV1106 板 (edge/rv1106/board_service)
  HighStream::feed_loop     RTSP 4K -> NALScanner -> VideoRing (内存, 不落盘)
  main.cpp                  FusionEvent confirmed/probable -> enqueue_event
  HighStream::make_clip     ring.cut() 关键帧对齐切片 + meta_json
  HighStream::upload_loop   有界重试队列 -> POST http://NAS:8000/api/ingest
                              multipart: meta=JSON, video=clip.h264|clip.hevc

NAS (src/nas_video_summarizer)
  app.RequestHandler._handle_board_ingest
    解析 multipart -> validate_meta -> event_key 幂等去重
    落盘排队 board_ingest_dir/pending/<key>/{meta.json, clip.h264}
    返回 202 {"accepted": true}
  workers._board_ingest_loop (每 5s)
    save_ingested_clip: ffmpeg -c copy -movflags +faststart .h264 -> .mp4
    -> NEXTCLOUD_OUTPUT_DIR/YYYY-MM-DD/HHMMSS_slug.mp4 + .json
    -> create_moment(analysis_backend=rv1106_edge, source_stream_role=board)
    -> rebuild_day_archive + ingested_clips 幂等落库
  失败重试 BOARD_INGEST_MAX_ATTEMPTS 次后移入 board_ingest_dir/failed/
```

## NAS 配置 (.env)

| 变量 | 默认 | 说明 |
|---|---|---|
| `BOARD_INGEST_ENABLED` | `false` | 启用 `/api/ingest` 与处理循环 |
| `BOARD_INGEST_DIR` | `./var/board_ingest` | 上传落盘队列 (pending/ failed/) |
| `BOARD_INGEST_MAX_BYTES` | `67108864` (64MB) | 单次上传上限, 超出回 413 |
| `BOARD_INGEST_MAX_ATTEMPTS` | `3` | 处理失败重试次数, 超出移入 failed/ |

同时按 AGENTS.md 建议: `RTSP_HIGH_URL` 留空 (NAS 不再自己录 4K),
`MQTT_ENABLED=false` (板端不再发 MQTT 事件, 改发剪辑)。

## 板端配置 (config.example.ini)

```ini
[high]
url = rtsp://camera/4k-stream
ring_mb = 64
upload_probable = false          ; true: probable 事件也切
[upload]
url = http://<nas>:8000/api/ingest
context_before_seconds = 5
context_after_seconds = 10
max_clip_seconds = 90
min_clip_seconds = 3
gap_limit_seconds = 2.5
max_retries = 3
retry_delay_seconds = 5
upload_timeout_seconds = 30
max_queue = 8
```

## 事件键 / 幂等

- `ingest_event_key()` 用 session_id + session_start + best_ts +
  clip 窗口 + identity 的 SHA-256, 截 24 字符。
- NAS 收到重复键: 若已入库 (ingested_clips.moment_id 非空) 返回
  `{"accepted": true, "duplicate": true}` 且不再处理; 若队列里已存在则
  直接返回 duplicate (同一网段重传不会产生两份)。
- 板端重试队列在 NAS 短暂不可用时会补传, NAS 靠键去重保证每条只入库一次。

## 剪辑元数据

板端 `meta_json()` 与 NAS 兼容字段:

```json
{
  "session_id": "1-3", "event": "confirmed", "identity": "confirmed",
  "track_id": 3, "ts": ..., "session_start": ..., "best_ts": ...,
  "score": 0.91, "face_score": ..., "person_score": ..., "activity_score": ...,
  "box": [..], "best_box": [..], "people_count": 1,
  "camera_id": "home-camera",
  "clip_start": ..., "clip_end": ..., "clip_bytes": ...,
  "codec": "H264", "source": "board_high_ring"
}
```

NAS 端产出 `HHMMSS_daughter-confirmed-by-rv1106.mp4` 同名 `.json`
(与原生 pipeline 同 schema_version=3), 每日 `manifest.json`/`summary.md`
自动重建, 供 Nextcloud 桌面端消费。

## 失效与运维

- `health /api/health` 的 `board_ingest` 段显示 enabled/done/failed 计数。
- 上传片段若 `ring.cut()` 拒绝 (老于缓冲窗口、断流 GAP、超长, 见
  `video_ring.h` CutResult), 板端只记日志, NAS 不会收到片段 —— 可在板端
  health JSON 的 `high_ring` 段看到 cut_reject 计数上涨。
- 若 ffmpeg 重封装失败 (损坏的裸流), 片段留在 pending, 重试
  `BOARD_INGEST_MAX_ATTEMPTS` 后移入 `failed/` 供人工排查。

## 并行保底期 (一刀切换)

板端与 NAS 端两条路径同时启用, 各自独立保存, Nextcloud 目录出现两套
（后缀相同的可能是同一事件的两种来源）。对拍 3-7 天后关闭 NAS 端
`RTSP_HIGH_URL`/`MQTT_ENABLED` 或直接移除旧路径, 数据以 `rv1106_edge`
(moments.analysis_backend) 为准。