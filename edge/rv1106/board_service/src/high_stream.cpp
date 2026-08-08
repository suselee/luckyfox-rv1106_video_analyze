#include "high_stream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>

#include "h264_source.h"
#include "http_uploader.h"
#include "nal_stats.h"
#include "time_util.h"

namespace dw {

// ---- 线程启动 ---------------------------------------------------------------
void* HighStream::feed_thread_fn(void* arg) {
    ((HighStream*)arg)->feed_loop();
    return NULL;
}

void* HighStream::upload_thread_fn(void* arg) {
    ((HighStream*)arg)->upload_loop();
    return NULL;
}

bool HighStream::start() {
    if (cfg_.rtsp_url.empty() || cfg_.upload_url.empty()) return false;
    ring_.resize(cfg_.ring_mb * 1024 * 1024);
    running_ = true;
    if (pthread_create(&feed_th_, NULL, feed_thread_fn, this) != 0) {
        running_ = false;
        return false;
    }
    if (pthread_create(&upload_th_, NULL, upload_thread_fn, this) != 0) {
        running_ = false;
        pthread_join(feed_th_, NULL);
        return false;
    }
    return true;
}

void HighStream::stop() {
    if (!running_) return;
    running_ = false;
    // 唤醒上传线程
    pthread_mutex_lock(&up_mu_);
    pthread_cond_broadcast(&up_cond_);
    pthread_mutex_unlock(&up_mu_);
    if (feed_th_) pthread_join(feed_th_, NULL);
    if (upload_th_) pthread_join(upload_th_, NULL);
    feed_th_ = 0;
    upload_th_ = 0;
}

// ---- 事件入队 ---------------------------------------------------------------
void HighStream::enqueue_event(const FusionEvent& ev) {
    pthread_mutex_lock(&ev_mu_);
    ev_queue_.push_back(ev);
    if (ev_queue_.size() > 16) ev_queue_.erase(ev_queue_.begin());
    pthread_mutex_unlock(&ev_mu_);
}

// ---- 4K 拉流线程: RTSP -> NAL -> 环形缓冲 -----------------------------------
void HighStream::feed_loop() {
    H264Source src;
    NalScanner scanner;
    const std::string* codec_holder = NULL;
    std::string codec_storage;

    scanner.set_callback([&](const uint8_t* nal, size_t len, double ts) {
        uint8_t type = 0;
        if (*codec_holder == "H265") {
            if (len >= 2) type = (uint8_t)((nal[0] >> 1) & 0x3F);
        } else if (len >= 1) {
            type = (uint8_t)(nal[0] & 0x1F);
        }
        ring_.push(nal, len, type, classify_nal(*codec_holder, type), ts);
    });

    std::vector<uint8_t> chunk(512 * 1024);
    int reconnect_wait = 2;

    while (running_) {
        // 处理队列里的融合事件 (切片/上传)
        drain_events();

        if (!src.is_open()) {
            if (!src.open(cfg_.rtsp_url)) {
                printf("[HIGH] RTSP open failed; retry in %ds\n", reconnect_wait);
                sleep(reconnect_wait);
                reconnect_wait = std::min(30, reconnect_wait * 2);
                continue;
            }
            codec_storage = src.codec();
            codec_holder = &codec_storage;
            reconnect_wait = 2;
            printf("[HIGH] 4K stream connected, codec=%s, ring=%zu MB\n",
                   codec_storage.c_str(), cfg_.ring_mb);
        }

        int n = src.read_chunk(chunk.data(), (int)chunk.size());
        if (n < 0) {
            printf("[HIGH] stream error; closing\n");
            src.close();
            reconnect_wait = 2;
            continue;
        }
        if (n > 0) {
            scanner.feed(chunk.data(), (size_t)n, now_seconds());
        }
    }
    src.close();
    printf("[HIGH] feed thread stopped\n");
}

// ---- 事件处理: 切片 + 入上传队列 ----------------------------------------------
void HighStream::drain_events() {
    std::vector<FusionEvent> evs;
    pthread_mutex_lock(&ev_mu_);
    evs.swap(ev_queue_);
    pthread_mutex_unlock(&ev_mu_);
    for (size_t i = 0; i < evs.size(); ++i) make_clip(evs[i]);
}

void HighStream::make_clip(const FusionEvent& ev) {
    bool do_cut = ev.event == "confirmed" ||
                  (cfg_.upload_probable && ev.event == "probable");
    if (!do_cut) return;

    // 事件窗口: session 开始前 context_before, 峰值时刻后 context_after;
    // 超出 max_clip_seconds 时压缩后窗 (最少 min_clip_seconds)。
    double t0 = ev.session_start;
    double t1 = ev.best_timestamp > 0 ? ev.best_timestamp : ev.timestamp;
    double dur = t1 - t0;
    double max_after = cfg_.max_clip_seconds - dur - cfg_.context_before;
    double after = std::max(cfg_.min_clip_seconds,
                            std::min(cfg_.context_after, max_after));
    if (max_after < cfg_.min_clip_seconds) {
        printf("[HIGH] window too long (%.1fs), skip session=%s\n",
               dur + cfg_.context_before + cfg_.context_after,
               ev.session_id.c_str());
        return;
    }
    double before = cfg_.context_before;

    size_t max_bytes = (size_t)(cfg_.max_clip_seconds * 2 * 1024 * 1024);
    std::vector<uint8_t> clip;
    CutResult rc = ring_.cut(t0, t1, before, after, cfg_.gap_limit,
                             max_bytes, clip);
    if (rc != CutResult::OK) {
        stats_.cut_reject++;
        printf("[HIGH] cut rejected (%d) session=%s window=[%.1f,%.1f] ring=[%.1f,%.1f]\n",
               (int)rc, ev.session_id.c_str(), t0 - before, t1 + after,
               ring_.oldest_ts(), ring_.newest_ts());
        return;
    }
    stats_.cut_ok++;

    std::string meta = meta_json(ev, t0 - before, t1 + after, clip.size());
    PendingClip pc;
    pc.data.swap(clip);
    pc.meta_json.swap(meta);
    pc.session_id = ev.session_id;
    pc.next_ts = now_seconds();

    pthread_mutex_lock(&up_mu_);
    if ((int)up_queue_.size() >= cfg_.max_queue) {
        up_queue_.pop_front();
        stats_.queue_drops++;
        printf("[HIGH] upload queue full; dropped oldest clip\n");
    }
    up_queue_.push_back(std::move(pc));
    pthread_cond_signal(&up_cond_);
    pthread_mutex_unlock(&up_mu_);
}

std::string HighStream::meta_json(const FusionEvent& ev, double clip_start,
                                  double clip_end, size_t clip_bytes) {
    char buf[1024];
    snprintf(buf, sizeof(buf),
             "{\"session_id\":\"%s\",\"event\":\"%s\",\"identity\":\"%s\","
             "\"track_id\":%u,\"ts\":%.3f,\"session_start\":%.3f,"
             "\"best_ts\":%.3f,\"score\":%.4f,\"face_score\":%.4f,"
             "\"person_score\":%.4f,\"activity_score\":%.4f,"
             "\"box\":[%.4f,%.4f,%.4f,%.4f],\"best_box\":[%.4f,%.4f,%.4f,%.4f],"
             "\"people_count\":%d,\"camera_id\":\"%s\","
             "\"clip_start\":%.3f,\"clip_end\":%.3f,\"clip_bytes\":%zu,"
             "\"source\":\"board_high_ring\"}",
             ev.session_id.c_str(), ev.event.c_str(), ev.identity.c_str(),
             ev.track_id, ev.timestamp, ev.session_start, ev.best_timestamp,
             ev.score, ev.face_score, ev.person_score, ev.activity_score,
             ev.box.x1, ev.box.y1, ev.box.x2, ev.box.y2,
             ev.best_box.x1, ev.best_box.y1, ev.best_box.x2, ev.best_box.y2,
             ev.people_count, cfg_.camera_id.c_str(),
             clip_start, clip_end, clip_bytes);
    return std::string(buf);
}

// ---- 上传线程: 有界重试队列 ---------------------------------------------------
void HighStream::upload_loop() {
    HttpUploader uploader;
    const std::string boundary = "dwclip-boundary";

    while (running_) {
        pthread_mutex_lock(&up_mu_);
        while (running_ && up_queue_.empty())
            pthread_cond_wait(&up_cond_, &up_mu_);
        if (!running_) {
            pthread_mutex_unlock(&up_mu_);
            break;
        }
        PendingClip pc = std::move(up_queue_.front());
        up_queue_.pop_front();
        pthread_mutex_unlock(&up_mu_);

        if (pc.next_ts > now_seconds()) {
            usleep((useconds_t)((pc.next_ts - now_seconds()) * 1e6));
        }

        std::vector<uint8_t> body = HttpUploader::build_multipart(
            boundary,
            {{"meta", pc.meta_json}},
            pc.data.data(), pc.data.size(), "clip.h264");

        HttpUploader::Response resp;
        bool ok = uploader.post(cfg_.upload_url, body, cfg_.upload_timeout, resp);
        if (ok) {
            stats_.uploads_ok++;
            stats_.last_upload_ts = now_seconds();
            printf("[HIGH] upload OK session=%s bytes=%zu status=%d\n",
                   pc.session_id.c_str(), pc.data.size(), resp.status);
        } else {
            pc.attempts++;
            if (pc.attempts <= cfg_.max_retries) {
                pc.next_ts = now_seconds() + cfg_.retry_delay * pc.attempts;
                pthread_mutex_lock(&up_mu_);
                up_queue_.push_back(std::move(pc));
                pthread_mutex_unlock(&up_mu_);
                printf("[HIGH] upload failed (retry %d/%d)\n",
                       pc.attempts, cfg_.max_retries);
            } else {
                stats_.uploads_failed++;
                printf("[HIGH] upload FAILED permanently\n");
            }
        }
    }
    printf("[HIGH] upload thread stopped\n");
}

// ---- 状态 ----------------------------------------------------------------
void HighStream::snapshot(UploadStats& out) {
    out = stats_;
}

void HighStream::status_json(char* buf, size_t n) {
    pthread_mutex_lock(&up_mu_);
    size_t pending = up_queue_.size();
    pthread_mutex_unlock(&up_mu_);
    snprintf(buf, n,
             ",\"high_ring\":{\"enabled\":%s,\"fill_percent\":%.1f,"
             "\"nals\":%zu,\"oldest_age_s\":%.0f,\"dropped_nals\":%llu,"
             "\"uploads_ok\":%llu,\"uploads_failed\":%llu,"
             "\"cut_ok\":%llu,\"cut_reject\":%llu,\"queue_drops\":%llu,"
             "\"pending\":%zu}",
             cfg_.enabled ? "true" : "false",
             ring_.capacity() ? 100.0 * (double)ring_.used_bytes() /
                                    (double)ring_.capacity()
                              : 0.0,
             ring_.nal_count(),
             ring_.newest_ts() - ring_.oldest_ts(),
             ring_.dropped_nals(), stats_.uploads_ok, stats_.uploads_failed,
             stats_.cut_ok, stats_.cut_reject, stats_.queue_drops,
             pending);
}

} // namespace dw