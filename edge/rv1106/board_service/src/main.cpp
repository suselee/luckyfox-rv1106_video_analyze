// daughter_watch: RV1106 low-stream daughter detector.
// RockIVA provides low-cost person/face detection and stable object ids;
// RetinaFace + MobileFaceNet are scheduled only when a track needs identity.

#include <algorithm>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <map>

#include "config.h"
#include "facedb.h"
#include "face_detect.h"
#include "face_recog.h"
#include "h264_source.h"
#include "high_stream.h"
#include "mpp_decoder.h"
#include "mqtt_publisher.h"
#include "rockiva_detector.h"
#include "schedule.h"
#include "system_monitor.h"
#include "track_fusion.h"

using namespace dw;

static volatile int g_running = 1;
static void on_signal(int) { g_running = 0; }

static HighStream* g_high = NULL;  // 4K 高码流链路 (可选)

// 与 time_util.h 的 dw::now_seconds() 同源同义 (gettimeofday)。

static int h264_nal_type(const uint8_t* data, int len) {
    int off = 0;
    if (len >= 4 && data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1) off = 4;
    else if (len >= 3 && data[0] == 0 && data[1] == 0 && data[2] == 1) off = 3;
    return off < len ? data[off] & 0x1F : -1;
}

// One scheduled face-recognition job: a region of the frame (normalized
// coordinates) plus the track the recognized identity should be applied to.
struct FaceRoi {
    uint32_t track_id;
    float x1, y1, x2, y2;
    bool rockiva_anchored;
    bool from_full_frame;
};

// Largest/clearest face sample seen per track. 识别时始终用轨迹内最大最清晰的
// 人脸裁块: 人走近时脸变大, 缓存随之更新, 避免小脸/运动模糊拉低相似度。
struct BestFace {
    std::vector<uint8_t> crop;   // RGB 裁块 (frame 分辨率, box+margin 区域)
    int crop_w = 0, crop_h = 0;  // 裁块尺寸 (像素)
    float face_x1, face_y1, face_x2, face_y2;  // 人脸框, 相对裁块归一化
    int face_w = 0, face_h = 0;  // 人脸尺寸 (frame 像素)
    float score = 0.0f;          // retinaface 检测分数
    double ts = 0.0;             // 存储时刻
};

// Crop the region out of the full RGB frame, run RetinaFace on the crop
// (the detector letterboxes any input size to 320x320, so small faces are
// upscaled and become detectable), then map the detections back to
// full-frame normalized coordinates.
static std::vector<FaceBox> detect_faces_in_region(
        FaceDetector& detector, const std::vector<uint8_t>& rgb,
        int frame_w, int frame_h, const FaceRoi& roi, float margin,
        float det_score) {
    float roi_w = roi.x2 - roi.x1;
    float roi_h = roi.y2 - roi.y1;
    float x1 = std::max(0.0f, roi.x1 - roi_w * margin);
    float y1 = std::max(0.0f, roi.y1 - roi_h * margin);
    float x2 = std::min(1.0f, roi.x2 + roi_w * margin);
    float y2 = std::min(1.0f, roi.y2 + roi_h * margin);
    int px1 = (int)(x1 * frame_w), py1 = (int)(y1 * frame_h);
    int px2 = std::min(frame_w, (int)(x2 * frame_w + 0.9999f));
    int py2 = std::min(frame_h, (int)(y2 * frame_h + 0.9999f));
    int cw = px2 - px1, ch = py2 - py1;
    std::vector<FaceBox> empty;
    if (cw < 16 || ch < 16) return empty;

    std::vector<uint8_t> crop((size_t)cw * ch * 3);
    for (int y = 0; y < ch; ++y)
        memcpy(crop.data() + (size_t)y * cw * 3,
               rgb.data() + ((size_t)(py1 + y) * frame_w + px1) * 3,
               (size_t)cw * 3);

    std::vector<FaceBox> faces = detector.detect(crop.data(), cw, ch, det_score);
    float region_w = x2 - x1, region_h = y2 - y1;
    for (size_t i = 0; i < faces.size(); ++i) {
        FaceBox& f = faces[i];
        f.x1 = x1 + f.x1 * region_w; f.x2 = x1 + f.x2 * region_w;
        f.y1 = y1 + f.y1 * region_h; f.y2 = y1 + f.y2 * region_h;
        for (int k = 0; k < 5; ++k) {
            f.lmk[k * 2 + 0] = x1 + f.lmk[k * 2 + 0] * region_w;
            f.lmk[k * 2 + 1] = y1 + f.lmk[k * 2 + 1] * region_h;
        }
    }
    return faces;
}

static float face_roi_iou(const FaceBox& face, const FaceRoi& roi) {
    float x1 = std::max(face.x1, roi.x1);
    float y1 = std::max(face.y1, roi.y1);
    float x2 = std::min(face.x2, roi.x2);
    float y2 = std::min(face.y2, roi.y2);
    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float face_area = std::max(0.0f, face.x2 - face.x1) *
                      std::max(0.0f, face.y2 - face.y1);
    float roi_area = std::max(0.0f, roi.x2 - roi.x1) *
                     std::max(0.0f, roi.y2 - roi.y1);
    float total = face_area + roi_area - intersection;
    return total > 0 ? intersection / total : 0;
}

// RetinaFace may return more than one face from an expanded crop. Only accept
// the detection that still maps to the scheduled track. For RockIVA-anchored
// jobs, also require geometric agreement with the original face box.
static int select_face_for_job(const std::vector<FaceBox>& faces,
                               const FaceRoi& job,
                               const TrackFusion& fusion) {
    int best = -1;
    float best_rank = -1e9f;
    float anchor_cx = (job.x1 + job.x2) * 0.5f;
    float anchor_cy = (job.y1 + job.y2) * 0.5f;
    float anchor_diag = sqrtf(
        (job.x2 - job.x1) * (job.x2 - job.x1) +
        (job.y2 - job.y1) * (job.y2 - job.y1));
    anchor_diag = std::max(0.01f, anchor_diag);
    for (size_t i = 0; i < faces.size(); ++i) {
        float cx = (faces[i].x1 + faces[i].x2) * 0.5f;
        float cy = (faces[i].y1 + faces[i].y2) * 0.5f;
        if (fusion.track_for_face(cx, cy) != job.track_id) continue;

        float dx = cx - anchor_cx;
        float dy = cy - anchor_cy;
        float distance = sqrtf(dx * dx + dy * dy) / anchor_diag;
        float overlap = face_roi_iou(faces[i], job);
        if (job.rockiva_anchored) {
            bool center_in_anchor = cx >= job.x1 && cx <= job.x2 &&
                                    cy >= job.y1 && cy <= job.y2;
            if (!center_in_anchor && overlap < 0.05f) continue;
        }
        float rank = faces[i].score - distance * 0.25f;
        if (job.rockiva_anchored) rank += overlap * 3.0f;
        if (rank > best_rank) {
            best_rank = rank;
            best = (int)i;
        }
    }
    return best;
}

static std::string event_payload(const FusionEvent& event, const std::string& camera,
                                 long sequence, int width, int height,
                                 const char* pipeline) {
    int x = (int)(event.box.x1 * width);
    int y = (int)(event.box.y1 * height);
    int w = (int)((event.box.x2 - event.box.x1) * width);
    int h = (int)((event.box.y2 - event.box.y1) * height);
    int best_x = (int)(event.best_box.x1 * width);
    int best_y = (int)(event.best_box.y1 * height);
    int best_w = (int)((event.best_box.x2 - event.best_box.x1) * width);
    int best_h = (int)((event.best_box.y2 - event.best_box.y1) * height);
    char payload[1024];
    snprintf(payload, sizeof(payload),
             "{\"ts\":%.3f,\"score\":%.4f,\"camera_id\":\"%s\","
             "\"box\":[%d,%d,%d,%d],\"best_box\":[%d,%d,%d,%d],"
             "\"frame_width\":%d,\"frame_height\":%d,"
             "\"seq\":%ld,\"event\":\"%s\","
             "\"session_id\":\"%s\",\"session_start_ts\":%.3f,"
             "\"track_id\":%u,\"identity\":\"%s\",\"face_score\":%.4f,"
             "\"person_score\":%.4f,\"activity_score\":%.4f,"
             "\"best_ts\":%.3f,\"people_count\":%d,\"pipeline\":\"%s\"}",
             event.timestamp, event.score, camera.c_str(), x, y, w, h,
             best_x, best_y, best_w, best_h, width, height, sequence,
             event.event.c_str(), event.session_id.c_str(), event.session_start,
             event.track_id, event.identity.c_str(), event.face_score,
             event.person_score, event.activity_score, event.best_timestamp,
             event.people_count, pipeline);
    return payload;
}

static void publish_fusion_events(
        MqttPublisher& mqtt, const std::string& topic, int qos,
        const std::string& camera, long& sequence,
        const std::vector<FusionEvent>& events, int width, int height,
        const char* pipeline) {
    for (size_t i = 0; i < events.size(); ++i) {
        sequence++;
        std::string payload = event_payload(
            events[i], camera, sequence, width, height, pipeline);
        bool sent = mqtt.publish(topic, payload, qos);
        printf("[EVENT] %s track=%u identity=%s score=%.3f MQTT=%s\n",
               events[i].event.c_str(), events[i].track_id,
               events[i].identity.c_str(), events[i].score,
               sent ? "OK" : "FAIL");
        // 高码流链路: 板端直接从 4K 环形缓冲切片段上传 NAS (新建通道)
        if (g_high) g_high->enqueue_event(events[i]);
    }
}

static bool publish_legacy_face(MqttPublisher& mqtt, const std::string& topic, int qos,
                                const std::string& camera, long sequence, double now,
                                float score, const FaceBox& face, int width, int height) {
    char payload[640];
    snprintf(payload, sizeof(payload),
             "{\"ts\":%.3f,\"score\":%.4f,\"camera_id\":\"%s\","
             "\"box\":[%d,%d,%d,%d],\"seq\":%ld,\"event\":\"hit\","
             "\"frame_width\":%d,\"frame_height\":%d,"
             "\"identity\":\"confirmed\",\"face_score\":%.4f,"
             "\"best_ts\":%.3f,\"pipeline\":\"face_only_guard\"}",
             now, score, camera.c_str(), (int)(face.x1 * width), (int)(face.y1 * height),
             (int)((face.x2 - face.x1) * width), (int)((face.y2 - face.y1) * height),
             sequence, width, height, score, now);
    return mqtt.publish(topic, payload, qos);
}

int main(int argc, char* argv[]) {
    // 行缓冲: 日志立即落盘, 便于板端实时诊断 (默认全缓冲在低频输出时
    // 可延迟数小时, 曾导致 HEALTH/FFDBG 长期不可见)。
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    if (argc < 2) {
        printf("Usage: %s <config.ini>\n", argv[0]);
        return 1;
    }
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);

    Config cfg;
    if (!cfg.load(argv[1])) {
        printf("[ERR] cannot load config: %s\n", argv[1]);
        return 1;
    }

    std::string rtsp_url = cfg.get("rtsp.url");
    bool rtsp_stdin_mode = cfg.get_bool("rtsp.stdin_mode", false);
    int rtsp_w = cfg.get_int("rtsp.width", 640);
    int rtsp_h = cfg.get_int("rtsp.height", 360);
    bool keyframes_only = cfg.get_bool("rtsp.keyframes_only", false);
    std::string pipeline_mode = cfg.get("pipeline.mode", "fusion");
    double person_fps = cfg.get_double("pipeline.person_scan_fps", 1.0);

    std::string det_path = cfg.get("model.detector");
    std::string rec_path = cfg.get("model.recognizer");
    std::string db_path = cfg.get("model.facedb");
    std::string rockiva_dir = cfg.get("model.rockiva_dir", "/root/daughter_watch/models/rockiva");

    float threshold = (float)cfg.get_double("recognize.threshold", 0.35);
    float high_threshold = (float)cfg.get_double("recognize.high_threshold", 0.55);
    int min_face = cfg.get_int("recognize.min_face", 24);
    float det_score = (float)cfg.get_double("recognize.det_score", 0.50);
    float roi_det_score = (float)cfg.get_double("recognize.roi_det_score", 0.40);

    std::string mqtt_host = cfg.get("mqtt.host", "127.0.0.1");
    int mqtt_port = cfg.get_int("mqtt.port", 1883);
    std::string mqtt_user = cfg.get("mqtt.username");
    std::string mqtt_pass = cfg.get("mqtt.password");
    std::string hit_topic = cfg.get("mqtt.topic", "homecam/daughter/hit");
    std::string status_topic = cfg.get("mqtt.status_topic", "homecam/daughter/status");
    std::string mqtt_cid = cfg.get("mqtt.client_id", "rv1106");
    int mqtt_qos = cfg.get_int("mqtt.qos", 1);
    std::string camera_id = cfg.get("meta.camera_id", "home-camera");

    std::string schedule_start_text = cfg.get("schedule.start", "07:00");
    std::string schedule_end_text = cfg.get("schedule.end", "21:00");
    ActiveSchedule schedule;
    schedule.enabled = cfg.get_bool("schedule.enabled", true);
    schedule.utc_offset_minutes = cfg.get_int("schedule.utc_offset_minutes", 480);
    if (!parse_hhmm(schedule_start_text, &schedule.start_minute) ||
        !parse_hhmm(schedule_end_text, &schedule.end_minute)) {
        printf("[ERR] schedule.start/end must use HH:MM\n");
        return 1;
    }

    if (rtsp_url.empty() || det_path.empty() || rec_path.empty() || db_path.empty()) {
        printf("[ERR] rtsp.url and all face model paths are required\n");
        return 1;
    }

    FaceDetector face_detector;
    FaceRecognizer recognizer;
    FaceDB db;
    if (!face_detector.init(det_path.c_str()) || !recognizer.init(rec_path.c_str()) ||
        !db.load(db_path.c_str()) || db.empty()) {
        printf("[ERR] face detector/recognizer/database initialization failed\n");
        return 1;
    }
    printf("[INIT] facedb=%d dim=%d threshold=%.3f high=%.3f\n",
           db.count(), db.dim(), threshold, high_threshold);

    // 全帧检测用独立 FaceDetector 实例: 与 ROI 检测器互不干扰 (rknn 上下文独立)
    FaceDetector ff_detector;
    if (!ff_detector.init(det_path.c_str())) {
        printf("[ERR] full-frame face detector init failed\n");
        return 1;
    }

    RockIvaDetector iva;
    bool fusion_enabled = pipeline_mode == "fusion" &&
        iva.init(rockiva_dir, rtsp_w, rtsp_h,
                 cfg.get_int("rockiva.person_score", 45),
                 cfg.get_int("rockiva.face_score", 45));
    if (pipeline_mode == "fusion" && !fusion_enabled)
        printf("[WARN] RockIVA unavailable (error=%d); starting face-only fallback\n", iva.last_error());

    FusionConfig fusion_cfg;
    fusion_cfg.probable_min_seconds = cfg.get_double("pipeline.probable_min_seconds", 10.0);
    fusion_cfg.probable_min_observations = cfg.get_int("pipeline.probable_min_observations", 8);
    fusion_cfg.child_max_height_ratio = cfg.get_double("pipeline.child_max_height_ratio", 0.40);
    fusion_cfg.relative_child_height_ratio = cfg.get_double("pipeline.relative_child_height_ratio", 0.65);
    fusion_cfg.face_check_interval_seconds = cfg.get_double("pipeline.face_check_interval_seconds", 1.0);
    fusion_cfg.face_hit_window_seconds = cfg.get_double("pipeline.face_hit_window_seconds", 6.0);
    fusion_cfg.confirmed_ttl_seconds = cfg.get_double("pipeline.confirmed_ttl_seconds", 8.0);
    fusion_cfg.track_lost_seconds = cfg.get_double("pipeline.track_lost_seconds", 6.0);
    fusion_cfg.probable_hold_seconds = cfg.get_double("pipeline.probable_hold_seconds", 3.0);
    fusion_cfg.confirm_child_hold_seconds = cfg.get_double(
        "pipeline.confirm_child_hold_seconds", 60.0);
    fusion_cfg.mqtt_update_seconds = cfg.get_double("pipeline.mqtt_update_seconds", 15.0);
    fusion_cfg.probable_min_activity = (float)cfg.get_double(
        "pipeline.probable_min_activity", 0.20);
    fusion_cfg.face_threshold = threshold;
    fusion_cfg.face_high_threshold = high_threshold;
    TrackFusion fusion(fusion_cfg);
    std::map<uint32_t, BestFace> best_faces;
    int max_face_rois = cfg.get_int("pipeline.max_face_rois_per_scan", 2);
    bool debug_dump_crops = cfg.get_bool("pipeline.debug_dump_crops", false);
    float face_roi_margin = (float)cfg.get_double("pipeline.face_roi_margin", 0.50);
    float head_roi_ratio = (float)cfg.get_double("pipeline.head_roi_ratio", 0.55);

    MqttPublisher mqtt;
    if (!mqtt.connect(mqtt_host, mqtt_port, mqtt_cid, mqtt_user, mqtt_pass))
        printf("[WARN] MQTT initial connection failed; publish will retry\n");

    // ---- 4K 高码流链路 (环形缓冲 + 切片上传) ----
    HighStream* high = NULL;
    HighStreamConfig high_cfg;
    high_cfg.rtsp_url = cfg.get("high.url");
    high_cfg.bind_ip = cfg.get("high.bind_ip");
    high_cfg.pipe_path = cfg.get("high.pipe_path");
    high_cfg.audio_pipe_path = cfg.get("high.audio_pipe_path");
    high_cfg.ring_mb = (size_t)cfg.get_int("high.ring_mb", 64);
    high_cfg.audio_enabled = cfg.get_bool("high.audio_enabled", true);
    high_cfg.audio_ring_mb = (size_t)cfg.get_int("high.audio_ring_mb", 2);
    high_cfg.upload_url = cfg.get("upload.url");
    high_cfg.upload_probable = cfg.get_bool("high.upload_probable", false);
    high_cfg.context_before = cfg.get_double("upload.context_before_seconds", 5.0);
    high_cfg.context_after = cfg.get_double("upload.context_after_seconds", 10.0);
    high_cfg.max_clip_seconds = cfg.get_double("upload.max_clip_seconds", 90.0);
    high_cfg.min_clip_seconds = cfg.get_double("upload.min_clip_seconds", 3.0);
    high_cfg.gap_limit = cfg.get_double("upload.gap_limit_seconds", 2.5);
    high_cfg.max_retries = cfg.get_int("upload.max_retries", 3);
    high_cfg.retry_delay = cfg.get_double("upload.retry_delay_seconds", 5.0);
    high_cfg.upload_timeout = cfg.get_double("upload.upload_timeout_seconds", 30.0);
    high_cfg.max_queue = cfg.get_int("upload.max_queue", 8);
    high_cfg.min_interval_seconds = cfg.get_double("upload.min_interval_seconds", 300.0);
    high_cfg.camera_id = camera_id;
    if (!high_cfg.rtsp_url.empty() && !high_cfg.upload_url.empty()) {
        high_cfg.enabled = true;
        high = new HighStream(high_cfg);
        if (high->start()) {
            g_high = high;
            printf("[INIT] high-stream enabled: ring=%zuMB upload=%s\n",
                   high_cfg.ring_mb, high_cfg.upload_url.c_str());
        } else {
            delete high;
            high = NULL;
            printf("[WARN] high-stream failed to start; MQTT-only mode\n");
        }
    }

    H264Source src;
    MppDecoder decoder(rtsp_w, rtsp_h);
    bool stream_active = false;

    SystemMonitor monitor;
    monitor.sample();
    PerformanceGuard guard(
        cfg.get_double("guard.max_cpu_percent", 65.0),
        cfg.get_int("guard.min_available_memory_mb", 80) * 1024L,
        cfg.get_double("guard.max_temperature_c", 75.0),
        cfg.get_double("guard.max_detector_p95_ms", 150.0));

    std::vector<uint8_t> chunk(256 * 1024);
    std::vector<uint8_t> rgb;
    std::vector<uint8_t> compact_nv12;
    std::vector<float> embedding;
    std::vector<double> detector_latencies;
    uint64_t pts = 0;
    uint32_t iva_frame_id = 0;
    long sequence = 0;
    long decoded_frames = 0;
    long session_frames_baseline = 0;
    long session_feeds = 0;
    long session_idrs = 0;
    double last_diag_log_ = -1e9;
    long scanned_frames = 0;
    long reconnects = 0;
    int idle_chunk_reads = 0;        // 连续无数据次数 (3s poll 超时/次)
    const int STALL_LIMIT = 20;      // 60s 无数据才强制重连 (期间每 3s 保活探测)
                                     // 摄像头惩罚模式会发几秒停几秒, 等待 > 重连
    unsigned long long rockiva_face_detections = 0;
    unsigned long long face_scan_attempts = 0;
    unsigned long long full_frame_scans = 0;
    unsigned long long full_frame_detections = 0;
    unsigned long long roi_scans = 0;
    unsigned long long retinaface_detections = 0;
    unsigned long long eligible_face_detections = 0;
    unsigned long long face_track_matches = 0;
    unsigned long long embedding_successes = 0;
    unsigned long long similarity_samples = 0;
    unsigned long long threshold_hits = 0;
    unsigned long long high_threshold_hits = 0;
    float max_face_similarity = -1.0f;
    int iva_failures = 0;
    double last_scan = -1e9;
    double last_face_fallback = -1e9;
    double last_face_hit = -1e9;
    double last_health = -1e9;
    double last_dbg = -1e9;
    double last_crop_dump = -1e9;
    double last_full_frame = -1e9;
    double last_feed_ts = -1e9;
    int reconnect_wait = 2;

    printf("[RUN] %s %dx%d H264; source 5fps, person scan %.2ffps; schedule=%s %s-%s UTC%+dmin\n",
           fusion_enabled ? "rockiva_fusion_v1" : "face_only", rtsp_w, rtsp_h,
           person_fps, schedule.enabled ? "on" : "off",
           schedule_start_text.c_str(), schedule_end_text.c_str(),
           schedule.utc_offset_minutes);

    while (g_running) {
        double loop_now = now_seconds();
        bool schedule_active = schedule_active_at(loop_now, schedule);
        if (!schedule_active) {
            if (stream_active || fusion.active_tracks() > 0) {
                std::vector<FusionEvent> ending = fusion.finish_sessions(loop_now);
                publish_fusion_events(
                    mqtt, hit_topic, mqtt_qos, camera_id, sequence, ending,
                    rtsp_w, rtsp_h,
                    fusion_enabled ? "rockiva_fusion_v1" : "face_only");
                if (stream_active) {
                    src.close();
                    decoder.deinit();
                }
                stream_active = false;
                if (g_high) {
                    g_high->stop();
                    printf("[SCHEDULE] high-stream stopped\n");
                }
                reconnect_wait = 2;
                last_health = -1e9;
                printf("[SCHEDULE] active window ended; RTSP and decoder stopped\n");
            } else if (g_high && g_high->running()) {
                // 窗口外启动 (初始 stream_active=false) 也要停掉 4K 链路,
                // 避免窗口外反复重连摄像头刷日志。
                g_high->stop();
                printf("[SCHEDULE] high-stream stopped (window closed)\n");
            }
            if (loop_now - last_health >= 60.0) {
                last_health = loop_now;
                SystemStats stats = monitor.sample();
                char high_json[2048];
                high_json[0] = '\0';
                if (g_high) g_high->status_json(high_json, sizeof(high_json));
                char status[2048];
                snprintf(status, sizeof(status),
                         "{\"ts\":%.3f,\"camera_id\":\"%s\",\"pipeline\":\"sleeping\","
                         "\"schedule_active\":false,\"active_window_start\":\"%s\","
                         "\"active_window_end\":\"%s\",\"utc_offset_minutes\":%d,"
                         "\"guard_level\":0,\"cpu_percent\":%.1f,"
                         "\"available_memory_mb\":%.1f,\"temperature_c\":%.1f,"
                         "\"detector_p95_ms\":0.0,\"person_scan_fps\":0.0,"
                         "\"active_tracks\":0,\"confirmed_tracks\":0,"
                         "\"probable_tracks\":0,\"confirmed_sessions\":%d,"
                         "\"decoded_frames\":%ld,\"scanned_frames\":%ld,"
                         "\"rtsp_reconnects\":%ld%s}",
                         loop_now, camera_id.c_str(), schedule_start_text.c_str(),
                         schedule_end_text.c_str(), schedule.utc_offset_minutes,
                         stats.cpu_percent, stats.available_memory_kb / 1024.0,
                         stats.temperature_c, fusion.confirmed_sessions(),
                         decoded_frames, scanned_frames, reconnects,
                         high_json);
                if (!status_topic.empty()) mqtt.publish(status_topic, status, 0);
                printf("[HEALTH] sleeping cpu=%.1f%% mem=%.1fMB temp=%.1fC\n",
                       stats.cpu_percent, stats.available_memory_kb / 1024.0,
                       stats.temperature_c);
            }
            sleep(1);
            continue;
        }

        if (!stream_active) {
            // 摄像头固件对"历史高频重连的客户端"有持久配额 (每会话只推
            // SPS/PPS + 几 KB 数据), 与协议细节无关 (已对齐 ffprobe 全部
            // 握手特征仍被配额)。stdin 模式: 由外部 ffprobe 拉流, 摄像头
            // 对 ffprobe 无配额, 女儿_watch 从管道读 Annex-B。
            if (rtsp_stdin_mode) {
                if (!src.open_stdin() || !decoder.init()) {
                    reconnects++;
                    printf("[WARN] stdin/decoder init failed; retry in %ds\n",
                           reconnect_wait);
                    sleep(reconnect_wait);
                    reconnect_wait = std::min(30, reconnect_wait * 2);
                    continue;
                }
                stream_active = true;
                reconnect_wait = 2;
                session_frames_baseline = decoded_frames;
                session_feeds = 0;
                session_idrs = 0;
                last_scan = -1e9;
                printf("[SCHEDULE] stdin stream ready (ffprobe pipe)\n");
            } else if (!src.open(rtsp_url, true) || !decoder.init()) {
                reconnects++;
                src.close();
                decoder.deinit();
                printf("[WARN] RTSP or decoder initialization failed; retrying in %ds\n",
                       reconnect_wait);
                sleep(reconnect_wait);
                reconnect_wait = std::min(30, reconnect_wait * 2);
                continue;
            }
            stream_active = true;
            if (g_high) {
                if (!g_high->running()) g_high->start();
                printf("[SCHEDULE] high-stream (re)started\n");
            }
            reconnect_wait = 2;
            session_frames_baseline = decoded_frames;
            session_feeds = 0;
            session_idrs = 0;
            last_scan = -1e9;
            last_face_fallback = -1e9;
            last_health = -1e9;
            printf("[SCHEDULE] active window started; RTSP and decoder running\n");
        }

        int n = src.read_chunk(chunk.data(), (int)chunk.size());
        if (n == -2) {
            // EOF: 摄像头主动断开。
            // 本会话拿到过帧 → 快速重连 (5s) 保持数据连续;
            // 没拿到帧就被踢 → 低频重连 (30s), 避免高频新连接
            // 触发摄像头侧惩罚 (曾实测连接周期被压到 2-3s)。
            if (!g_running) break;
            reconnects++;
            if (now_seconds() - last_diag_log_ > 30.0) {
                last_diag_log_ = now_seconds();
            }
            src.close();
            decoder.deinit();
            stream_active = false;
            idle_chunk_reads = 0;
            if (decoded_frames > session_frames_baseline) {
                reconnect_wait = 2;
                sleep(5);
            } else {
                if (now_seconds() - last_diag_log_ > 30.0) {
                    last_diag_log_ = now_seconds();
                    printf("[DIAG] EOF, session frames=%ld, backoff=%d\n",
                           decoded_frames - session_frames_baseline, reconnect_wait);
                }
                // 低频重连: 摄像头对高频新连接进入惩罚模式
                // (每个会话只给几秒数据就断)。30s 间隔让惩罚窗口过期。
                sleep(30);
            }
            continue;
        }
        if (n == -3) {
            // 音频包/RTCP: 数据在流动, 不计 stall, 不喂解码器
            idle_chunk_reads = 0;
            continue;
        }
        if (n < 0) {
            if (!g_running) break;
            reconnects++;
            src.close();
            decoder.deinit();
            stream_active = false;
            idle_chunk_reads = 0;
            sleep(reconnect_wait);
            reconnect_wait = std::min(30, reconnect_wait * 2);
            continue;
        }
        if (n == 0) {
            // RTP 无数据 (3s poll 超时)。摄像头对历史高频重连的客户端
            // 有配额: 每会话推几 KB 后静默。注意: 不能在此发 GET_PARAMETER
            // 保活 —— 它与 RTP 数据共用一个 TCP 连接, rtsp_req 逐字节读
            // 响应时会吞掉 RTP 帧破坏读流 (曾实测数据流被持续破坏)。
            // 保活改用单向 RTCP RR (maybe_send_rtcp, 不读响应)。
            if (++idle_chunk_reads >= STALL_LIMIT) {
                reconnects++;
                if (now_seconds() - last_diag_log_ > 30.0) {
                    last_diag_log_ = now_seconds();
                    printf("[DIAG] STALL: %d x 3s poll timeout, no data; reconnect\n",
                           idle_chunk_reads);
                }
                src.close();
                decoder.deinit();
                stream_active = false;
                idle_chunk_reads = 0;
                sleep(30);
                continue;
            }
            continue;
        }
        idle_chunk_reads = 0;
        if (n > 0) {
            int nal_type = h264_nal_type(chunk.data(), n);
            // keyframes_only: 关键帧必喂; 中间帧按 1s 节流抽样,
            // 把有效分析帧率从 GOP(约 0.5fps)提到 ~1fps, 让 person
            // 扫描/轨迹融合回到全帧时代的节奏, 同时 CPU 只有全帧的 ~1/10。
            bool feed = !keyframes_only || nal_type == 5 || nal_type == 6 ||
                        nal_type == 7 || nal_type == 8 || nal_type == 9;
            if (!feed && keyframes_only && now_seconds() - last_feed_ts >= 1.0) {
                feed = true;
                last_feed_ts = now_seconds();
            }
            if (feed) {
                decoder.send(chunk.data(), n, pts++, true);
                session_feeds++;
                if (nal_type == 5) session_idrs++;
            }
        }

        while (true) {
            Nv12Frame frame;
            if (!decoder.get_frame(frame, 0)) break;
            decoded_frames++;
            double now = now_seconds();
            int level = guard.level();
            double effective_fps = level == 0 ? person_fps : (level == 1 ? 0.5 : 1.0);
            if (effective_fps <= 0) effective_fps = 0.5;
            if (now - last_scan < 1.0 / effective_fps) continue;
            last_scan = now;
            scanned_frames++;

            if (fusion_enabled && level < 2 && frame.yuv420_layout()) {
                IvaResult objects;
                double begin = now_seconds();
                bool ok = false;
                if (frame.data_fd() >= 0) {
                    ok = iva.detect_fd(++iva_frame_id, frame.data_fd(),
                                       frame.yuv420_layout(), frame.width(),
                                       frame.height(), objects);
                } else if (frame.physical_addr()) {
                    ok = iva.detect_physical(++iva_frame_id, frame.physical_addr(),
                                             frame.yuv420_layout(),
                                             frame.width(), frame.height(), objects);
                } else if (frame.copy_nv12(compact_nv12)) {
                    ok = iva.detect_nv12(++iva_frame_id, compact_nv12.data(), frame.width(), frame.height(), objects);
                }
                double latency = (now_seconds() - begin) * 1000.0;
                detector_latencies.push_back(latency);
                if (detector_latencies.size() > 180) detector_latencies.erase(detector_latencies.begin());
                if (!ok) {
                    printf("[ROCKIVA] detect failed error=%d\n", iva.last_error());
                    objects = IvaResult();
                    if (++iva_failures >= 3) {
                        fusion_enabled = false;
                        printf("[GUARD] RockIVA disabled after 3 consecutive failures; face-only fallback\n");
                    }
                } else {
                    iva_failures = 0;
                }
                fusion.observe(now, objects);
                rockiva_face_detections += objects.faces.size();

                if (now - last_dbg >= 10.0) {
                    last_dbg = now;
                    std::vector<TrackSnapshot> dbg_snaps = fusion.snapshot(now);
                    int child_like = 0;
                    for (size_t i = 0; i < dbg_snaps.size(); ++i)
                        if (dbg_snaps[i].child_like) child_like++;
                    int overlap = 0;
                    for (size_t i = 0; i < objects.faces.size(); ++i) {
                        const IvaObject& f = objects.faces[i];
                        float cx = (f.x1 + f.x2) * 0.5f;
                        float cy = (f.y1 + f.y2) * 0.5f;
                        if (fusion.track_for_face(cx, cy)) overlap++;
                    }
                    printf("[DBG] t=%.1f iva_faces=%zu persons=%zu tracks=%zu child_like=%d iva_overlap=%d\n",
                           now, objects.faces.size(), objects.people.size(),
                           dbg_snaps.size(), child_like, overlap);
                    for (size_t i = 0; i < objects.faces.size() && i < 3; ++i) {
                        const IvaObject& f = objects.faces[i];
                        printf("[DBG]   iva_face#%zu box=(%.3f,%.3f,%.3f,%.3f) size=%dx%dpx score=%.2f\n",
                               i, f.x1, f.y1, f.x2, f.y2,
                               (int)((f.x2 - f.x1) * frame.width()),
                               (int)((f.y2 - f.y1) * frame.height()), f.score);
                    }
                    for (size_t i = 0; i < objects.people.size() && i < 2; ++i) {
                        const IvaObject& p = objects.people[i];
                        printf("[DBG]   person#%zu box=(%.3f,%.3f,%.3f,%.3f) size=%dx%dpx score=%.2f\n",
                               i, p.x1, p.y1, p.x2, p.y2,
                               (int)((p.x2 - p.x1) * frame.width()),
                               (int)((p.y2 - p.y1) * frame.height()), p.score);
                    }
                }

                // 全帧检测 (独立于 jobs, 1s 节流): 脸保持全帧自然尺寸
                // (640x360 → letterbox 320, 脸约 20-40px), 检测可靠; ROCKIVA
                // 小裁块路径会把脸放大 4-8x 导致检测丢失。直接识别并计入融合。
                if (now - last_full_frame >= 1.0 && frame.to_rgb(rgb)) {
                    last_full_frame = now;
                    full_frame_scans++;
                    std::vector<FaceBox> all_faces = ff_detector.detect(
                        rgb.data(), frame.width(), frame.height(), det_score);
                    full_frame_detections += all_faces.size();
                    if (all_faces.empty())
                        printf("[FFDBG] t=%.1f no-face-in-full-frame\n", now);
                    for (size_t fi = 0; fi < all_faces.size(); ++fi) {
                        const FaceBox& f = all_faces[fi];
                        int fw_px = (int)((f.x2 - f.x1) * frame.width());
                        int fh_px = (int)((f.y2 - f.y1) * frame.height());
                        if (fw_px < min_face || fh_px < min_face) continue;
                        float cx = (f.x1 + f.x2) * 0.5f;
                        float cy = (f.y1 + f.y2) * 0.5f;
                        uint32_t track = fusion.track_for_face(cx, cy);
                        if (!track || !fusion.should_check_face(track, now)) continue;
                        fusion.mark_face_checked(track, now);
                        roi_scans++;
                        eligible_face_detections++;
                        face_track_matches++;
                        std::vector<float> embedding;
                        if (!recognizer.extract(rgb.data(), frame.width(),
                                                frame.height(), f, embedding)) {
                            printf("[FFDBG] t=%.1f face=%ux%u track=%u embed-fail\n",
                                   now, fw_px, fh_px, track);
                            continue;
                        }
                        embedding_successes++;
                        float similarity = db.best_similarity(embedding);
                        similarity_samples++;
                        max_face_similarity = std::max(max_face_similarity, similarity);
                        if (similarity >= threshold) threshold_hits++;
                        if (similarity >= high_threshold) high_threshold_hits++;
                        printf("[FACE] t=%.1f track=%u roi=full det=%.3f size=%dx%d "
                               "similarity=%.4f result=%s\n",
                               now, track, f.score, fw_px, fh_px, similarity,
                               similarity >= high_threshold ? "high-hit" :
                               (similarity >= threshold ? "hit" : "below-threshold"));
                        fusion.apply_face_score(track, similarity, now);
                    }
                }

                // Schedule recognition jobs: RockIVA face boxes anchored to
                // tracks first (precise attribution, works even when the
                // child is held), then head regions of due child-like tracks.
                std::vector<TrackSnapshot> snaps = fusion.snapshot(now);
                std::vector<FaceRoi> jobs;
                std::vector<uint32_t> roi_tracks;
                for (size_t i = 0; i < objects.faces.size(); ++i) {
                    const IvaObject& iva_face = objects.faces[i];
                    float cx = (iva_face.x1 + iva_face.x2) * 0.5f;
                    float cy = (iva_face.y1 + iva_face.y2) * 0.5f;
                    uint32_t track = fusion.track_for_face(cx, cy);
                    if (!track || !fusion.should_check_face(track, now)) continue;
                    if (std::find(roi_tracks.begin(), roi_tracks.end(), track) != roi_tracks.end())
                        continue;
                    FaceRoi job;
                    job.track_id = track;
                    job.x1 = iva_face.x1; job.y1 = iva_face.y1;
                    job.x2 = iva_face.x2; job.y2 = iva_face.y2;
                    job.rockiva_anchored = true;
                    job.from_full_frame = false;
                    jobs.push_back(job);
                    roi_tracks.push_back(track);
                }
                for (size_t i = 0; i < snaps.size(); ++i) {
                    const TrackSnapshot& snap = snaps[i];
                    if (!snap.child_like || snap.ambiguous) continue;
                    if (!fusion.should_check_face(snap.id, now)) continue;
                    if (std::find(roi_tracks.begin(), roi_tracks.end(), snap.id) != roi_tracks.end())
                        continue;
                    FaceRoi job;
                    job.track_id = snap.id;
                    job.x1 = snap.box.x1;
                    job.y1 = snap.box.y1;
                    job.x2 = snap.box.x2;
                    job.y2 = snap.box.y1 + (snap.box.y2 - snap.box.y1) * head_roi_ratio;
                    job.rockiva_anchored = false;
                    job.from_full_frame = false;
                    jobs.push_back(job);
                    roi_tracks.push_back(snap.id);
                }

                if (!jobs.empty()) {
                    face_scan_attempts++;
                    if (best_faces.size() > 32) {
                        for (auto it = best_faces.begin(); it != best_faces.end();) {
                            if (now - it->second.ts > 90.0) it = best_faces.erase(it);
                            else ++it;
                        }
                    }
                    if (frame.to_rgb(rgb)) {
                        int budget = max_face_rois;
                        for (size_t i = 0; i < jobs.size() && budget > 0; ++i, --budget) {
                            const FaceRoi& job = jobs[i];
                            fusion.mark_face_checked(job.track_id, now);
                            roi_scans++;
                            std::vector<FaceBox> faces = detect_faces_in_region(
                                face_detector, rgb, frame.width(), frame.height(),
                                job, face_roi_margin, roi_det_score);
                            retinaface_detections += faces.size();

                            // 调试: 把 ROI 检测器输入裁块落盘 (PPM, RGB), 用于离线
                            // 对照检测器行为; 每 5 秒最多一张, 仅 iva 任务。
                            // 必须放在任何 selection/min_face 过滤之前: 检测空结果
                            // 正是要抓的现象。
                            if (debug_dump_crops && job.rockiva_anchored &&
                                now - last_crop_dump >= 5.0) {
                                last_crop_dump = now;
                                float roi_w = job.x2 - job.x1, roi_h = job.y2 - job.y1;
                                float rx1 = std::max(0.0f, job.x1 - roi_w * face_roi_margin);
                                float ry1 = std::max(0.0f, job.y1 - roi_h * face_roi_margin);
                                float rx2 = std::min(1.0f, job.x2 + roi_w * face_roi_margin);
                                float ry2 = std::min(1.0f, job.y2 + roi_h * face_roi_margin);
                                int px1 = (int)(rx1 * frame.width());
                                int py1 = (int)(ry1 * frame.height());
                                int px2 = std::min(frame.width(),
                                                   (int)(rx2 * frame.width() + 0.9999f));
                                int py2 = std::min(frame.height(),
                                                   (int)(ry2 * frame.height() + 0.9999f));
                                int cw = px2 - px1, ch = py2 - py1;
                                if (cw >= 16 && ch >= 16) {
                                    char path[96];
                                    snprintf(path, sizeof(path), "/tmp/roi_%.0f_%u.ppm",
                                             now, job.track_id);
                                    FILE* pf = fopen(path, "wb");
                                    if (pf) {
                                        fprintf(pf, "P6\n%d %d\n255\n", cw, ch);
                                        for (int yy = 0; yy < ch; ++yy)
                                            fwrite(rgb.data() +
                                                       ((size_t)(py1 + yy) * frame.width() + px1) * 3,
                                                   1, (size_t)cw * 3, pf);
                                        fclose(pf);
                                        printf("[DETDBG] saved %s crop=%dx%d track=%u raw_faces=%zu\n",
                                               path, cw, ch, job.track_id, faces.size());
                                    }
                                }
                            }

                            int selected = select_face_for_job(faces, job, fusion);
                            if (selected < 0) continue;
                            const FaceBox& face = faces[(size_t)selected];
                            int face_w = (int)((face.x2 - face.x1) * frame.width());
                            int face_h = (int)((face.y2 - face.y1) * frame.height());
                            if (face_w < min_face || face_h < min_face) continue;
                            eligible_face_detections++;
                            face_track_matches++;
                            const char* roi_label = job.rockiva_anchored ? "iva" : "head";

                            // 与 detect_faces_in_region 一致的裁块区域 (box+margin),
                            // 更新该轨迹最大/最清晰人脸缓存。
                            float roi_w = job.x2 - job.x1, roi_h = job.y2 - job.y1;
                            float rx1 = std::max(0.0f, job.x1 - roi_w * face_roi_margin);
                            float ry1 = std::max(0.0f, job.y1 - roi_h * face_roi_margin);
                            float rx2 = std::min(1.0f, job.x2 + roi_w * face_roi_margin);
                            float ry2 = std::min(1.0f, job.y2 + roi_h * face_roi_margin);
                            int px1 = (int)(rx1 * frame.width());
                            int py1 = (int)(ry1 * frame.height());
                            int px2 = std::min(frame.width(), (int)(rx2 * frame.width() + 0.9999f));
                            int py2 = std::min(frame.height(), (int)(ry2 * frame.height() + 0.9999f));
                            int cw = px2 - px1, ch = py2 - py1;
                            BestFace& bf = best_faces[job.track_id];
                            if (cw >= 16 && ch >= 16) {
                                bool replace = bf.crop_w <= 0;
                                if (!replace) {
                                    long long cur_area = (long long)face_w * face_h;
                                    long long old_area = (long long)bf.face_w * bf.face_h;
                                    if (cur_area > old_area) replace = true;
                                    else if (face.score > bf.score + 0.05f &&
                                             cur_area >= old_area * 85 / 100) replace = true;
                                }
                                if (replace) {
                                    bf.crop.resize((size_t)cw * ch * 3);
                                    for (int yy = 0; yy < ch; ++yy)
                                        memcpy(bf.crop.data() + (size_t)yy * cw * 3,
                                               rgb.data() + ((size_t)(py1 + yy) * frame.width() + px1) * 3,
                                               (size_t)cw * 3);
                                    bf.crop_w = cw; bf.crop_h = ch;
                                    bf.face_x1 = (face.x1 * frame.width() - px1) / (float)cw;
                                    bf.face_y1 = (face.y1 * frame.height() - py1) / (float)ch;
                                    bf.face_x2 = (face.x2 * frame.width() - px1) / (float)cw;
                                    bf.face_y2 = (face.y2 * frame.height() - py1) / (float)ch;
                                    bf.face_w = face_w; bf.face_h = face_h;
                                    bf.score = face.score;
                                    bf.ts = now;
                                }
                            }

                            // 用该轨迹当前最优人脸裁块做识别
                            if (bf.crop_w <= 0) continue;
                            FaceBox cf;
                            cf.x1 = bf.face_x1; cf.y1 = bf.face_y1;
                            cf.x2 = bf.face_x2; cf.y2 = bf.face_y2;
                            cf.score = bf.score;
                            if (!recognizer.extract_crop(bf.crop.data(), bf.crop_w, bf.crop_h, cf, embedding)) {
                                printf("[FACE] t=%.1f track=%u roi=%s det=%.3f size=%dx%d result=embedding-failed\n",
                                       now, job.track_id, roi_label, bf.score, bf.face_w, bf.face_h);
                                continue;
                            }
                            embedding_successes++;
                            float similarity = db.best_similarity(embedding);
                            similarity_samples++;
                            max_face_similarity = std::max(max_face_similarity, similarity);
                            if (similarity >= threshold) threshold_hits++;
                            if (similarity >= high_threshold) high_threshold_hits++;
                            printf("[FACE] t=%.1f track=%u roi=%s det=%.3f size=%dx%d similarity=%.4f result=%s\n",
                                   now, job.track_id, roi_label,
                                   bf.score, bf.face_w, bf.face_h, similarity,
                                   similarity >= high_threshold ? "high-hit" :
                                   (similarity >= threshold ? "hit" : "below-threshold"));
                            fusion.apply_face_score(job.track_id, similarity, now);
                        }
                    }
                }

                std::vector<FusionEvent> events = fusion.collect_events(now);
                publish_fusion_events(
                    mqtt, hit_topic, mqtt_qos, camera_id, sequence, events,
                    frame.width(), frame.height(), "rockiva_fusion_v1");
            } else if (now - last_face_fallback >= 1.0 && frame.to_rgb(rgb)) {
                last_face_fallback = now;
                std::vector<FaceBox> faces = face_detector.detect(
                    rgb.data(), frame.width(), frame.height(), det_score);
                float best = -1;
                FaceBox best_face = {};
                for (size_t i = 0; i < faces.size(); ++i) {
                    int fw = (int)((faces[i].x2 - faces[i].x1) * frame.width());
                    int fh = (int)((faces[i].y2 - faces[i].y1) * frame.height());
                    if (fw < min_face || fh < min_face) continue;
                    if (!recognizer.extract(rgb.data(), frame.width(), frame.height(), faces[i], embedding)) continue;
                    float similarity = db.best_similarity(embedding);
                    if (similarity > best) { best = similarity; best_face = faces[i]; }
                }
                if (best >= threshold && now - last_face_hit >= 10.0) {
                    last_face_hit = now;
                    publish_legacy_face(mqtt, hit_topic, mqtt_qos, camera_id,
                                        ++sequence, now, best, best_face,
                                        frame.width(), frame.height());
                }
            }

            if (now - last_health >= 60.0) {
                last_health = now;
                SystemStats stats = monitor.sample();
                double p95 = SystemMonitor::percentile95(detector_latencies);
                int level = guard.update(stats, p95);
                char high_json[2048];
                high_json[0] = '\0';
                if (g_high) g_high->status_json(high_json, sizeof(high_json));
                char status[3072];
                snprintf(status, sizeof(status),
                         "{\"ts\":%.3f,\"camera_id\":\"%s\",\"pipeline\":\"%s\","
                         "\"schedule_active\":true,\"active_window_start\":\"%s\","
                         "\"active_window_end\":\"%s\",\"utc_offset_minutes\":%d,"
                         "\"guard_level\":%d,\"cpu_percent\":%.1f,"
                         "\"available_memory_mb\":%.1f,\"temperature_c\":%.1f,"
                         "\"detector_p95_ms\":%.1f,\"person_scan_fps\":%.2f,"
                         "\"active_tracks\":%d,\"confirmed_tracks\":%d,"
                         "\"probable_tracks\":%d,\"confirmed_sessions\":%d,"
                         "\"rockiva_face_detections\":%llu,\"face_scan_attempts\":%llu,"
                         "\"roi_scans\":%llu,"
                         "\"full_frame_scans\":%llu,\"full_frame_detections\":%llu,"
                         "\"retinaface_detections\":%llu,\"eligible_face_detections\":%llu,"
                         "\"face_track_matches\":%llu,\"embedding_successes\":%llu,"
                         "\"similarity_samples\":%llu,\"max_face_similarity\":%.4f,"
                         "\"face_threshold_hits\":%llu,\"face_high_threshold_hits\":%llu,"
                         "\"decoded_frames\":%ld,"
                         "\"scanned_frames\":%ld,\"rtsp_reconnects\":%ld%s}",
                          now, camera_id.c_str(), fusion_enabled ? "rockiva_fusion_v1" : "face_only",
                          schedule_start_text.c_str(), schedule_end_text.c_str(),
                          full_frame_scans, full_frame_detections,
                          schedule.utc_offset_minutes,
                          level, stats.cpu_percent, stats.available_memory_kb / 1024.0,
                          stats.temperature_c, p95,
                          level == 0 ? person_fps : (level == 1 ? 0.5 : 1.0),
                          fusion.active_tracks(), fusion.confirmed_tracks(),
                          fusion.probable_tracks(), fusion.confirmed_sessions(),
                          rockiva_face_detections, face_scan_attempts, roi_scans,
                          retinaface_detections,
                          eligible_face_detections, face_track_matches,
                          embedding_successes, similarity_samples,
                          max_face_similarity, threshold_hits, high_threshold_hits,
                          decoded_frames, scanned_frames, reconnects, high_json);
                if (!status_topic.empty()) mqtt.publish(status_topic, status, 0);
                printf("[HEALTH] cpu=%.1f%% mem=%.1fMB temp=%.1fC p95=%.1fms guard=%d "
                       "dec=%ld scan=%ld rkf=%llu fscan=%llu roi=%llu det=%llu elig=%llu emb=%llu sim=%llu maxsim=%.4f\n",
                       stats.cpu_percent, stats.available_memory_kb / 1024.0,
                       stats.temperature_c, p95, level,
                       decoded_frames, scanned_frames, rockiva_face_detections,
                       face_scan_attempts, roi_scans, retinaface_detections,
                       eligible_face_detections, embedding_successes, similarity_samples,
                       max_face_similarity);
                detector_latencies.clear();
            }
        }
    }

    printf("[SHUTDOWN] cleaning up\n");
    std::vector<FusionEvent> ending = fusion.finish_sessions(now_seconds());
    publish_fusion_events(
        mqtt, hit_topic, mqtt_qos, camera_id, sequence, ending,
        rtsp_w, rtsp_h,
        fusion_enabled ? "rockiva_fusion_v1" : "face_only");
    if (g_high) {
        g_high->stop();
        delete g_high;
        g_high = NULL;
    }
    src.close();
    decoder.deinit();
    mqtt.disconnect();
    iva.destroy();
    recognizer.destroy();
    face_detector.destroy();
    return 0;
}
