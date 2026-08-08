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

#include "config.h"
#include "facedb.h"
#include "face_detect.h"
#include "face_recog.h"
#include "h264_source.h"
#include "mpp_decoder.h"
#include "mqtt_publisher.h"
#include "rockiva_detector.h"
#include "schedule.h"
#include "system_monitor.h"
#include "track_fusion.h"

using namespace dw;

static volatile int g_running = 1;
static void on_signal(int) { g_running = 0; }

static double now_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

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
    fusion_cfg.mqtt_update_seconds = cfg.get_double("pipeline.mqtt_update_seconds", 15.0);
    fusion_cfg.face_threshold = threshold;
    fusion_cfg.face_high_threshold = high_threshold;
    TrackFusion fusion(fusion_cfg);
    int max_face_rois = cfg.get_int("pipeline.max_face_rois_per_scan", 2);
    float face_roi_margin = (float)cfg.get_double("pipeline.face_roi_margin", 0.50);
    float head_roi_ratio = (float)cfg.get_double("pipeline.head_roi_ratio", 0.55);

    MqttPublisher mqtt;
    if (!mqtt.connect(mqtt_host, mqtt_port, mqtt_cid, mqtt_user, mqtt_pass))
        printf("[WARN] MQTT initial connection failed; publish will retry\n");

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
    long scanned_frames = 0;
    long reconnects = 0;
    unsigned long long rockiva_face_detections = 0;
    unsigned long long face_scan_attempts = 0;
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
                reconnect_wait = 2;
                last_health = -1e9;
                printf("[SCHEDULE] active window ended; RTSP and decoder stopped\n");
            }
            if (loop_now - last_health >= 60.0) {
                last_health = loop_now;
                SystemStats stats = monitor.sample();
                char status[1536];
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
                         "\"rtsp_reconnects\":%ld}",
                         loop_now, camera_id.c_str(), schedule_start_text.c_str(),
                         schedule_end_text.c_str(), schedule.utc_offset_minutes,
                         stats.cpu_percent, stats.available_memory_kb / 1024.0,
                         stats.temperature_c, fusion.confirmed_sessions(),
                         decoded_frames, scanned_frames, reconnects);
                if (!status_topic.empty()) mqtt.publish(status_topic, status, 0);
                printf("[HEALTH] sleeping cpu=%.1f%% mem=%.1fMB temp=%.1fC\n",
                       stats.cpu_percent, stats.available_memory_kb / 1024.0,
                       stats.temperature_c);
            }
            sleep(1);
            continue;
        }

        if (!stream_active) {
            if (!src.open(rtsp_url) || !decoder.init()) {
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
            reconnect_wait = 2;
            last_scan = -1e9;
            last_face_fallback = -1e9;
            last_health = -1e9;
            printf("[SCHEDULE] active window started; RTSP and decoder running\n");
        }

        int n = src.read_chunk(chunk.data(), (int)chunk.size());
        if (n < 0) {
            if (!g_running) break;
            reconnects++;
            src.close();
            decoder.deinit();
            stream_active = false;
            sleep(reconnect_wait);
            reconnect_wait = std::min(30, reconnect_wait * 2);
            continue;
        }
        if (n > 0) {
            int nal_type = h264_nal_type(chunk.data(), n);
            bool feed = !keyframes_only || nal_type == 5 || nal_type == 6 ||
                        nal_type == 7 || nal_type == 8 || nal_type == 9;
            if (feed) decoder.send(chunk.data(), n, pts++, true);
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
                    jobs.push_back(job);
                    roi_tracks.push_back(snap.id);
                }

                if (!jobs.empty()) {
                    face_scan_attempts++;
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
                            int selected = select_face_for_job(faces, job, fusion);
                            if (selected < 0) continue;
                            const FaceBox& face = faces[(size_t)selected];
                            int face_w = (int)((face.x2 - face.x1) * frame.width());
                            int face_h = (int)((face.y2 - face.y1) * frame.height());
                            if (face_w < min_face || face_h < min_face) continue;
                            eligible_face_detections++;
                            face_track_matches++;
                            if (!recognizer.extract(rgb.data(), frame.width(), frame.height(), face, embedding)) {
                                printf("[FACE] t=%.1f track=%u roi=%s det=%.3f size=%dx%d result=embedding-failed\n",
                                       now, job.track_id,
                                       job.rockiva_anchored ? "iva" : "head",
                                       face.score, face_w, face_h);
                                continue;
                            }
                            embedding_successes++;
                            float similarity = db.best_similarity(embedding);
                            similarity_samples++;
                            max_face_similarity = std::max(max_face_similarity, similarity);
                            if (similarity >= threshold) threshold_hits++;
                            if (similarity >= high_threshold) high_threshold_hits++;
                            printf("[FACE] t=%.1f track=%u roi=%s det=%.3f size=%dx%d similarity=%.4f result=%s\n",
                                   now, job.track_id,
                                   job.rockiva_anchored ? "iva" : "head",
                                   face.score, face_w, face_h, similarity,
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
                char status[2048];
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
                         "\"retinaface_detections\":%llu,\"eligible_face_detections\":%llu,"
                         "\"face_track_matches\":%llu,\"embedding_successes\":%llu,"
                         "\"similarity_samples\":%llu,\"max_face_similarity\":%.4f,"
                         "\"face_threshold_hits\":%llu,\"face_high_threshold_hits\":%llu,"
                         "\"decoded_frames\":%ld,"
                         "\"scanned_frames\":%ld,\"rtsp_reconnects\":%ld}",
                          now, camera_id.c_str(), fusion_enabled ? "rockiva_fusion_v1" : "face_only",
                          schedule_start_text.c_str(), schedule_end_text.c_str(),
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
                          decoded_frames, scanned_frames, reconnects);
                if (!status_topic.empty()) mqtt.publish(status_topic, status, 0);
                printf("[HEALTH] cpu=%.1f%% mem=%.1fMB temp=%.1fC p95=%.1fms guard=%d\n",
                       stats.cpu_percent, stats.available_memory_kb / 1024.0,
                       stats.temperature_c, p95, level);
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
    src.close();
    decoder.deinit();
    mqtt.disconnect();
    iva.destroy();
    recognizer.destroy();
    face_detector.destroy();
    return 0;
}
