# Mine NAS AI Video Architecture

> The diagram below documents the legacy `ANALYSIS_BACKEND=vlm` path. With
> `ANALYSIS_BACKEND=daughter_detector`, the main/candidate/final VLM stages are
> replaced by dense local OpenCV detection, temporal event grouping, and local
> final verification. Both backends share high-stream extraction, SQLite
> moments, quotas, and the Nextcloud archive. See
> [daughter-detector.md](daughter-detector.md) for the detector and desktop
> handoff contract.

> The recommended RV1106 production path is now board-primary and high-only on
> the NAS: the board analyzes the low stream during its active window, the NAS
> records only 4K, and MQTT `best_ts` is resolved directly against high-stream
> segments. Confirmed events publish directly; probable events optionally run
> a strict five-frame NAS verification. `ANALYSIS_ENABLED=false` disables the
> continuous analyzer without disabling MQTT event processing.

下面这张图对应当前 `main` 分支的实现。箭头表示数据流，虚线表示控制、状态或查询。

```mermaid
flowchart LR
    CAM[家庭摄像头\nRTSP low + RTSP high]
    LLAMA[llama.cpp server\nQwen3-VL-2B\nOpenAI-compatible API]
    NC[Nextcloud\n读取共享输出目录]
    USER[浏览器 / 管理员]

    subgraph JAIL[FreeBSD NAS jail]
        RC[rc.d: nas_video\ndaemon + uv run nas-video]

        subgraph APP[单个 Python 进程]
            HTTP[HTTP 线程\nThreadingHTTPServer\n静态页面 + REST API]
            RUNTIME[WorkerRuntime\n独立 asyncio event loop]

            subgraph SUP[Supervisor 并发任务]
                RLOW[low recorder\nffmpeg RTSP -> wall-clock segments]
                RHIGH[high recorder\nffmpeg RTSP -> wall-clock segments]
                SCAN[scanner\n每10秒发现稳定文件\n批量 upsert + 对齐监控]
                ANALYZE[analyzer\n按时间顺序处理 low segment]
                CLEAN[cleanup\n按 RETENTION_HOURS 清理原始 buffer]
            end
        end

        BUFFER[(BUFFER_DIR\nlow/high 120s mp4)]
        DB[(SQLite\nWAL + busy_timeout\nsegments / moments / events)]
        OUT[(NEXTCLOUD_OUTPUT_DIR\nYYYY-MM-DD/\nmp4 + json + summary.md)]
        MODELS[(person_filter_models\nYOLO + face + age)]
    end

    CAM -->|RTSP low| RLOW
    CAM -->|RTSP high| RHIGH
    RLOW --> BUFFER
    RHIGH --> BUFFER
    BUFFER -. stable files .-> SCAN
    SCAN --> DB
    CLEAN --> BUFFER
    CLEAN --> DB

    USER -->|HTTP :8000| HTTP
    HTTP --> DB
    HTTP --> OUT
    OUT --> NC

    DB -->|pending low, older than delay| ANALYZE
    BUFFER -->|low segment| ANALYZE
    ANALYZE --> MODELS
    SCAN -. recent low/high offsets .-> ALIGN[stream_alignment\nstable / drifted / insufficient]
    ALIGN -. /api/health .-> HTTP

    subgraph PIPE[单个片段的分析与保存流水线]
        SAMPLE[ffmpeg 抽样\n默认本地最多12帧]
        BLANK[黑帧过滤]
        PERSON[YOLO + face/age\n无人物 / adult-only 可直接跳过]
        MAIN[主 VLM #1\n4张 384px\nkeep + offsets + JSON]
        GATE[keep/confidence\n冷却 + 时段/每日配额]
        CAND[候选验证 VLM #2\n低流候选附近3张 512px]
        SOURCE[按时间找全部重叠 high segments\n跨段时 concat]
        STAGE[ffmpeg 生成隐藏暂存高流视频\n同一输出文件系统]
        FINAL[最终验证 VLM #3\n暂存视频前/中/后3帧\n本地检测 + VLM]
        PUBLISH[验证通过后原子 rename\n写 mp4/json/summary.md\n写 moments 记录]
    end

    ANALYZE --> SAMPLE --> BLANK --> PERSON --> MAIN
    MAIN -->|keep=false| SKIP[analysis-skip\nmark processed]
    MAIN -->|keep=true| GATE
    GATE -->|cooldown/cap blocked| SKIP
    GATE --> CAND -->|通过| SOURCE
    CAND -->|失败| SKIP
    SOURCE --> STAGE --> FINAL
    FINAL -->|失败/超时/无孩子| DISCARD[删除暂存文件\n不发布、不淘汰旧视频]
    FINAL -->|通过| PUBLISH --> OUT
    PUBLISH --> DB

    MAIN -. timeout fallback .-> LLAMA
    CAND --> LLAMA
    FINAL --> LLAMA
    LLAMA --> MAIN
    LLAMA --> CAND
    LLAMA --> FINAL

    ANALYZE -. timeout 计数 .-> CIRCUIT[连续超时达到阈值\n暂停分析5分钟\n/api/health 显示 circuit-open]
    CIRCUIT -. resume .-> ANALYZE
```

## 如何阅读

1. **录像层**：摄像头同时输出 low 和 high 两路 RTSP。两个 ffmpeg recorder 各自按 FreeBSD 墙上时钟切成约120秒文件。`RECORD_WINDOW_START/END` 外会停止拉流。
2. **索引层**：scanner 只把稳定的 mp4 登记到 SQLite；analyzer 只处理 `analysis_stream_role=low` 的待处理片段，并故意等待 `ANALYSIS_DELAY_SECONDS`。
3. **本地预过滤层**：OpenCV DNN 运行 YOLO、脸部检测和年龄分类。黑帧、无人物、确定只有成人的片段不会调用 VLM。
4. **主判断层**：Qwen3-VL 查看低流抽样帧，返回 `keep`、置信度、标题、摘要和片段偏移。
5. **候选确认层**：主判断为正面后，重新抽取候选附近的高分辨率帧，进行第二次 VLM 确认。
6. **最终视频层**：根据 low 的时间偏移找到所有重叠 high 文件，生成隐藏暂存视频，再从暂存视频抽取前、中、后三帧进行第三次确认。只有这里通过后才发布到 Nextcloud 目录。
7. **持久化层**：发布后写视频、JSON 元数据、每日 `summary.md`，再写 SQLite `moments` 记录。配额淘汰也延迟到新视频成功登记后执行。
8. **时间监控层**：scanner 用最近 low/high 片段的开始时间计算中位数 offset。它监控 recorder 边界是否同步，不把摄像头 OSD 水印误认为 RTSP 元数据。

## 当前一次正面片段的 VLM 成本

正常情况下，一个最终保存的片段会产生三次 VLM 请求：

| 阶段 | 输入 | 目的 |
| --- | --- | --- |
| 主分析 | 4 张 384px low 帧 | 判断是否值得保存、确定时间范围 |
| 候选验证 | 3 张 512px low 候选帧 | 防止主模型误判时间或人物 |
| 最终验证 | 暂存 high 视频的 3 帧 | 确认实际准备发布的视频不是空片段 |

主分析为负面的片段不会进入后两步。VLM 超时会先尝试联系表回退；连续超时会触发 analyzer 熔断，避免 NAS 无限堆积请求。

## 主要目录和记录

```text
BUFFER_DIR/
  <camera>/low/*.mp4       # 分析源，过 RETENTION_HOURS 后清理
  <camera>/high/*.mp4      # 原始高质量源，过 RETENTION_HOURS 后清理

NEXTCLOUD_OUTPUT_DIR/
  2026-07-15/
    083351_child-walking.mp4
    083351_child-walking.json
    summary.md

DATABASE_PATH
  segments                    # 原始片段索引与分析状态
  moments                     # 已发布视频记录
  events                      # skip/error/verification/cap 事件
```

`CAMERA_TIME_OFFSET_SECONDS` 只修正展示时间、文件名、JSON 和配额日期，不改变 ffmpeg 在源文件内的相对截取偏移。对于示例中摄像头时间比 NAS 慢约23秒的情况，应设置为 `-23`。
