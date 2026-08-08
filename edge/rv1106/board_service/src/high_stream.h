#pragma once
#include <pthread.h>
#include <stdint.h>
#include <deque>
#include <string>
#include <vector>

#include "track_fusion.h"  // FusionEvent
#include "video_ring.h"

namespace dw {

struct HighStreamConfig {
    bool enabled = false;
    std::string rtsp_url;        // 4K 主码流
    size_t ring_mb = 64;         // 环形缓冲大小
    std::string upload_url;      // http://nas-host:port/api/ingest
    double context_before = 5.0; // 事件前保留
    double context_after = 10.0; // 事件后保留
    double max_clip_seconds = 90.0;
    double gap_limit = 2.5;      // 断流判定 (秒)
    int max_retries = 3;
    double retry_delay = 5.0;
    double upload_timeout = 30.0;
    int max_queue = 8;           // 上传待发队列上限 (个)
    bool upload_probable = false; // false: 只切 confirmed
    std::string camera_id;
    double min_clip_seconds = 3.0;
};

struct UploadStats {
    unsigned long long uploads_ok = 0;
    unsigned long long uploads_failed = 0;
    unsigned long long ring_miss = 0;     // 事件窗口已出缓冲
    unsigned long long cut_ok = 0;
    unsigned long long cut_reject = 0;    // GAP/NO_PARAMS 等
    unsigned long long queue_drops = 0;
    double last_upload_ts = 0.0;
};

// 4K 高码流链路: 专用线程拉 RTSP -> 内存环形缓冲 (不解码);
// 收到融合管线 confirmed/probable 事件 -> 环形缓冲切片 -> 队列上传 NAS。
// enqueue_event 可从任意线程调用 (内部加锁); 其余状态仅本模块线程访问。
class HighStream {
public:
    explicit HighStream(const HighStreamConfig& cfg) : cfg_(cfg), ring_(0) {}
    ~HighStream() { stop(); }

    bool start();
    void stop();

    // 线程安全: 把融合事件送入切片队列 (主线程调用)。
    void enqueue_event(const FusionEvent& ev);

    // 线程安全: 状态快照 (供 status JSON / 日志)。
    void snapshot(UploadStats& out);
    void status_json(char* buf, size_t n);  // 追加到 health payload

    bool running() const { return running_; }

private:
    struct PendingClip {
        std::vector<uint8_t> data;
        std::string meta_json;
        std::string session_id;
        int attempts = 0;
        double next_ts = 0.0;
    };

    static void* feed_thread_fn(void* arg);
    static void* upload_thread_fn(void* arg);
    void feed_loop();
    void upload_loop();
    void drain_events();
    void make_clip(const FusionEvent& ev);
    std::string meta_json(const FusionEvent& ev, double clip_start,
                          double clip_end, size_t clip_bytes);

    HighStreamConfig cfg_;
    VideoRing ring_;
    pthread_t feed_th_ = 0;
    pthread_t upload_th_ = 0;
    volatile bool running_ = false;

    pthread_mutex_t ev_mu_ = PTHREAD_MUTEX_INITIALIZER;
    std::vector<FusionEvent> ev_queue_;

    pthread_mutex_t up_mu_ = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t up_cond_ = PTHREAD_COND_INITIALIZER;
    std::deque<PendingClip> up_queue_;

    UploadStats stats_;
};

} // namespace dw