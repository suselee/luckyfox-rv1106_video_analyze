#pragma once
#include <pthread.h>
#include <atomic>
#include <stdint.h>
#include <deque>
#include <string>
#include <vector>

#include "audio_ring.h"
#include "track_fusion.h"  // FusionEvent
#include "video_ring.h"

namespace dw {

struct HighStreamConfig {
    bool enabled = false;
    std::string rtsp_url;        // 4K 主码流
    std::string bind_ip;         // 可选: 绑定的源地址 (换身份避开摄像头配额)
    std::string pipe_path;       // 可选: FIFO 路径 (ffmpeg 拉流输出, 摄像头无配额)
    std::string audio_pipe_path; // 可选: 音频 FIFO 路径 (ADTS AAC)
    size_t ring_mb = 64;         // 环形缓冲大小
    bool audio_enabled = true;   // 尝试协商音频轨 (无音频时静默降级)
    size_t audio_ring_mb = 2;    // 音频环形缓冲大小 (G711 8kHz ≈ 8KB/s)
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
    double min_interval_seconds = 300.0; // 两次切片上传之间最短间隔 (全局冷却)
    // Q2 尾部裁剪: clip 末端收到 last_active_ts + 该秒数, 去掉人走后的静默。
    double tail_trim_seconds = 2.0;
};

struct UploadStats {
    unsigned long long uploads_ok = 0;
    unsigned long long uploads_failed = 0;
    unsigned long long ring_miss = 0;     // 事件窗口已出缓冲
    unsigned long long cut_ok = 0;
    unsigned long long cut_reject = 0;    // GAP/NO_PARAMS 等
    unsigned long long cut_skipped = 0;   // 冷却窗口内跳过 (重复抑制)
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

    // 窗口外暂停: 保持线程与 FIFO fd 存活, 仅停止读取。
    // 管道背压让 ffmpeg-high 写端自然阻塞休眠 (不会 SIGPIPE 死亡);
    // resume() 后从暂停点继续, 无需重连/重新握手。
    void pause();
    void resume();
    bool paused() const { return paused_; }

private:
    struct PendingClip {
        std::vector<uint8_t> data;
        std::string meta_json;
        std::string session_id;
        std::string clip_name;  // clip.h264 / clip.hevc
        std::vector<uint8_t> audio;
        std::string audio_name; // clip.g711a / clip.g711u / clip.adts (空=无音频)
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
    static std::string audio_clip_name(const std::string& audio_codec);

    HighStreamConfig cfg_;
    VideoRing ring_;
    AudioRing audio_ring_;
    std::string codec_;  // "H264"/"H265" (feed 线程写, 单线程访问)
    std::string audio_codec_; // "PCMU"/"PCMA"/"AAC" (feed 线程写)
    int audio_rate_ = 8000;
    int audio_channels_ = 1;
    unsigned long long audio_chunks_ = 0;
    pthread_t feed_th_ = 0;
    pthread_t upload_th_ = 0;
    volatile bool running_ = false;
    std::atomic<bool> paused_{false};

    pthread_mutex_t ev_mu_ = PTHREAD_MUTEX_INITIALIZER;
    std::vector<FusionEvent> ev_queue_;

    pthread_mutex_t up_mu_ = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t up_cond_ = PTHREAD_COND_INITIALIZER;
    std::deque<PendingClip> up_queue_;

    UploadStats stats_;
    double last_cut_ts_ = 0.0;   // 上次成功切片时间 (全局冷却, feed 线程独占)
};

} // namespace dw