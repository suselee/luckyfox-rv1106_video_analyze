#include "video_ring.h"

#include <stdio.h>
#include <string.h>

namespace dw {

void VideoRing::reset() {
    buf_.assign(cap_, 0);
    head_ = 0;
    used_ = 0;
    metas_.clear();
    dropped_ = 0;
}

void VideoRing::write_at(size_t off, const uint8_t* src, size_t n) {
    if (off + n <= cap_) {
        memcpy(buf_.data() + off, src, n);
        return;
    }
    size_t first = cap_ - off;
    memcpy(buf_.data() + off, src, first);
    memcpy(buf_.data(), src + first, n - first);
}

void VideoRing::read_at(size_t off, size_t n, std::vector<uint8_t>& out) const {
    if (off + n <= cap_) {
        out.insert(out.end(), buf_.data() + off, buf_.data() + off + n);
        return;
    }
    size_t first = cap_ - off;
    out.insert(out.end(), buf_.data() + off, buf_.data() + off + first);
    out.insert(out.end(), buf_.data(), buf_.data() + (n - first));
}

void VideoRing::evict_until(size_t needed) {
    while (used_ + needed > cap_ && !metas_.empty()) {
        const RingMeta& m = metas_.front();
        head_ = (head_ + m.size) % cap_;
        used_ -= m.size;
        metas_.pop_front();
    }
}

void VideoRing::push(const uint8_t* nal, size_t len, uint8_t type,
                     const NalKind& kind, double ts) {
    if (len > cap_) {
        dropped_++;
        printf("[RING] NAL too large (%zu B > %zu B cap); dropped\n", len, cap_);
        return;
    }
    evict_until(len);
    if (used_ + len > cap_) {  // 极端: 单 NAL 仍放不下 (理论上不出现)
        dropped_++;
        return;
    }
    write_at((head_ + used_) % cap_, nal, len);
    RingMeta m;
    m.off  = (uint32_t)((head_ + used_) % cap_);
    m.size = (uint32_t)len;
    m.ts   = ts;
    m.type = type;
    m.key  = kind.key;
    m.ps   = kind.ps;
    metas_.push_back(m);
    used_ += len;
}

// 从 lo 回溯返回关键帧下标 (含); 若其前紧邻参数集, 一并吞入 (start_ps)。
size_t VideoRing::snap_start(const std::deque<RingMeta>& m, size_t lo,
                             size_t& start_ps) {
    if (lo >= m.size()) return (size_t)-1;
    // 向前走到"前一个元素是关键帧"的位置
    size_t i = lo;
    while (i > 0 && !m[i - 1].key) --i;
    if (i == 0 && !m[0].key) return (size_t)-1;  // 窗口前没有关键帧
    size_t key = (i == 0) ? 0 : i - 1;           // 关键帧下标
    size_t ps = key;
    while (ps > 0 && m[ps - 1].ps) --ps;         // 吞掉关键帧前的参数集
    start_ps = ps;
    return key;
}

CutResult VideoRing::cut(double t0, double t1, double before, double after,
                         double gap_limit, size_t max_bytes,
                         std::vector<uint8_t>& out) const {
    out.clear();
    if (metas_.empty()) return CutResult::EMPTY_WINDOW;

    double lo_ts = t0 - before;
    double hi_ts = t1 + after;

    // 线性扫描定位 [lo_idx, hi_idx] (按 ts 单调递增)
    size_t lo_idx = metas_.size();
    size_t hi_idx = metas_.size();
    for (size_t i = 0; i < metas_.size(); ++i) {
        if (lo_idx == metas_.size() && metas_[i].ts >= lo_ts) lo_idx = i;
        if (metas_[i].ts <= hi_ts) hi_idx = i;
    }
    if (lo_idx == metas_.size() || hi_idx == metas_.size() || lo_idx > hi_idx)
        return CutResult::EMPTY_WINDOW;

    // 起点: 窗口前最近关键帧 (+ 其参数集)
    size_t start_ps = 0;
    size_t start = snap_start(metas_, lo_idx, start_ps);
    if (start == (size_t)-1) return CutResult::NO_KEYFRAME;

    // 终点: 延伸到窗口后的下一个关键帧 (含), 使片段在关键帧边界收尾
    size_t end = hi_idx;
    while (end + 1 < metas_.size() && !metas_[end + 1].key) ++end;

    // 断流检测: 窗口内相邻 NAL 间隔 > gap_limit
    for (size_t i = start + 1; i <= end; ++i) {
        if (metas_[i].ts - metas_[i - 1].ts > gap_limit)
            return CutResult::GAP;
    }

    // 参数集检查: 窗口内必须含参数集, 否则解码器无法起步
    bool has_ps = false;
    for (size_t i = start_ps; i <= end; ++i)
        if (metas_[i].ps) { has_ps = true; break; }
    if (!has_ps) return CutResult::NO_PARAMS;

    // 尺寸预估 (含 4 字节起始码)
    size_t total = 0;
    for (size_t i = start_ps; i <= end; ++i) total += metas_[i].size + 4;
    if (total > max_bytes) return CutResult::TOO_BIG;

    out.reserve(total);
    const uint8_t sc[4] = {0x00, 0x00, 0x00, 0x01};
    for (size_t i = start_ps; i <= end; ++i) {
        const RingMeta& m = metas_[i];
        out.insert(out.end(), sc, sc + 4);
        read_at(m.off, m.size, out);
    }
    return CutResult::OK;
}

} // namespace dw