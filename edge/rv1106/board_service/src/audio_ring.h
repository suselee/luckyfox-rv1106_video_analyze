// 音频环形缓冲: 与 VideoRing 同构的字节环, 按墙钟 ts 索引, 切片时按窗口截取。
// PCMU/PCMA 为原始 PCM, AAC 为 ADTS 帧; 本类对内容透明, 只按时间窗口拷贝字节。
// 只被高码流 feed 线程读写; 不用锁。
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <deque>
#include <vector>

namespace dw {

struct AudioMeta {
    uint32_t off;
    uint32_t size;
    double   ts;   // 墙钟秒 (now_seconds)
};

class AudioRing {
public:
    explicit AudioRing(size_t capacity_bytes = 0) : cap_(capacity_bytes) { reset(); }

    void resize(size_t capacity_bytes) {
        cap_ = capacity_bytes;
        reset();
    }

    void reset();

    // 追加一段音频字节 (len > capacity 时丢弃并计数)。
    void push(const uint8_t* data, size_t len, double ts);

    // 截取 [lo_ts, hi_ts] 内的字节; 窗口内最大间隔超过 gap_limit 视为断流。
    // 与视频 cut 共用窗口, 起点/终点精确按 ts (音频无需关键帧对齐)。
    // 返回 0=成功, 1=窗口超出缓冲, 2=断流, 3=超 max_bytes。
    int cut(double lo_ts, double hi_ts, double gap_limit, size_t max_bytes,
            std::vector<uint8_t>& out) const;

    size_t capacity() const { return cap_; }
    size_t used_bytes() const { return used_; }
    size_t chunk_count() const { return metas_.size(); }
    double oldest_ts() const { return metas_.empty() ? 0.0 : metas_.front().ts; }
    double newest_ts() const { return metas_.empty() ? 0.0 : metas_.back().ts; }
    unsigned long long dropped_chunks() const { return dropped_; }

private:
    void write_at(size_t off, const uint8_t* src, size_t n);
    void read_at(size_t off, size_t n, std::vector<uint8_t>& out) const;

    std::vector<uint8_t> buf_;
    size_t cap_;
    size_t head_ = 0;
    size_t used_ = 0;
    unsigned long long dropped_ = 0;
    std::deque<AudioMeta> metas_;
};

} // namespace dw
