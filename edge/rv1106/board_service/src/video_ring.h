#pragma once
#include <stdint.h>
#include <deque>
#include <vector>

#include "nal_util.h"

namespace dw {

// 一条 NAL 的元数据: 数据在 buf_ 环形区内的偏移 + 接收时刻。
struct RingMeta {
    uint32_t off;
    uint32_t size;
    double   ts;   // 墙钟秒 (now_seconds)
    uint8_t  type; // 码流 NAL type (已按 codec 归一)
    bool     key;
    bool     ps;
};

enum class CutResult {
    OK,
    EMPTY_WINDOW,  // 环形缓冲为空或窗口超出缓冲
    NO_KEYFRAME,   // 窗口之前没有关键帧可对齐
    NO_PARAMS,     // 窗口内没有参数集 (无法解码)
    GAP,           // 窗口内出现断流 (>gap_limit)
    TOO_BIG,       // 切片超过 max_bytes
};

// 内存环形缓冲: 高码流 (4K) 原始 NAL 透传存放, 按需覆盖最旧数据。
// 只被单个线程 (高码流线程) 读写; 不用锁。
class VideoRing {
public:
    explicit VideoRing(size_t capacity_bytes = 0) : cap_(capacity_bytes) { reset(); }

    // 重设容量 (分配新缓冲, 清空数据)。避免拷贝大缓冲。
    void resize(size_t capacity_bytes) {
        cap_ = capacity_bytes;
        reset();
    }

    void reset();

    // 追加一条 NAL (len > capacity 时丢弃并计数)。
    void push(const uint8_t* nal, size_t len, uint8_t type, const NalKind& kind,
              double ts);

    // 组装 [t0 - before, t1 + after] 的完整码流片段 (Annex-B 4 字节起始码):
    // 起点回退到窗口前最近关键帧 (并带上其参数集), 终点延伸到下一个关键帧。
    // gap_limit: 窗口内相邻 NAL 间隔超过该值视为断流, 返回 GAP。
    CutResult cut(double t0, double t1, double before, double after,
                  double gap_limit, size_t max_bytes,
                  std::vector<uint8_t>& out) const;

    size_t capacity() const { return cap_; }
    size_t used_bytes() const { return used_; }
    size_t nal_count() const { return metas_.size(); }
    double oldest_ts() const { return metas_.empty() ? 0.0 : metas_.front().ts; }
    double newest_ts() const { return metas_.empty() ? 0.0 : metas_.back().ts; }
    unsigned long long dropped_nals() const { return dropped_; }

private:
    void evict_until(size_t needed);
    void write_at(size_t off, const uint8_t* src, size_t n);
    void read_at(size_t off, size_t n, std::vector<uint8_t>& out) const;
    // 从 idx 处向前 (更早) 回溯: 找到最近的 key 元数据; 若找到了还继续吞掉其前面
    // 连续的 ps 元数据 (参数集紧跟关键帧前)。
    static size_t snap_start(const std::deque<RingMeta>& m, size_t lo, size_t& start_ps);

    std::vector<uint8_t> buf_;
    size_t cap_;
    size_t head_ = 0;   // 最旧存活数据偏移
    size_t used_ = 0;   // 存活字节数
    unsigned long long dropped_ = 0;
    std::deque<RingMeta> metas_;
};

} // namespace dw