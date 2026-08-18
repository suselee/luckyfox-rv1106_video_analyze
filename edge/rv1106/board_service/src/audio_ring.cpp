#include "audio_ring.h"

#include <stdio.h>
#include <string.h>

namespace dw {

void AudioRing::reset() {
    buf_.assign(cap_, 0);
    head_ = 0;
    used_ = 0;
    metas_.clear();
    dropped_ = 0;
}

void AudioRing::write_at(size_t off, const uint8_t* src, size_t n) {
    if (off + n <= cap_) {
        memcpy(buf_.data() + off, src, n);
        return;
    }
    size_t first = cap_ - off;
    memcpy(buf_.data() + off, src, first);
    memcpy(buf_.data(), src + first, n - first);
}

void AudioRing::read_at(size_t off, size_t n, std::vector<uint8_t>& out) const {
    if (off + n <= cap_) {
        out.insert(out.end(), buf_.data() + off, buf_.data() + off + n);
        return;
    }
    size_t first = cap_ - off;
    out.insert(out.end(), buf_.data() + off, buf_.data() + off + first);
    out.insert(out.end(), buf_.data(), buf_.data() + (n - first));
}

void AudioRing::push(const uint8_t* data, size_t len, double ts) {
    if (len == 0) return;
    if (len > cap_) {
        dropped_++;
        printf("[AUDIO] chunk too large (%zu B > %zu B cap); dropped\n", len, cap_);
        return;
    }
    while (used_ + len > cap_ && !metas_.empty()) {
        const AudioMeta& m = metas_.front();
        head_ = (head_ + m.size) % cap_;
        used_ -= m.size;
        metas_.pop_front();
    }
    if (used_ + len > cap_) {
        dropped_++;
        return;
    }
    write_at((head_ + used_) % cap_, data, len);
    AudioMeta m;
    m.off  = (uint32_t)((head_ + used_) % cap_);
    m.size = (uint32_t)len;
    m.ts   = ts;
    metas_.push_back(m);
    used_ += len;
}

int AudioRing::cut(double lo_ts, double hi_ts, double gap_limit,
                   size_t max_bytes, std::vector<uint8_t>& out) const {
    out.clear();
    if (metas_.empty()) return 1;

    size_t lo_idx = metas_.size();
    size_t hi_idx = metas_.size();
    for (size_t i = 0; i < metas_.size(); ++i) {
        if (lo_idx == metas_.size() && metas_[i].ts >= lo_ts) lo_idx = i;
        if (metas_[i].ts <= hi_ts) hi_idx = i;
    }
    if (lo_idx == metas_.size() || hi_idx == metas_.size() || lo_idx > hi_idx)
        return 1;

    // 断流检测: 窗口内相邻块间隔 > gap_limit
    for (size_t i = lo_idx + 1; i <= hi_idx; ++i) {
        if (metas_[i].ts - metas_[i - 1].ts > gap_limit) return 2;
    }

    size_t total = 0;
    for (size_t i = lo_idx; i <= hi_idx; ++i) total += metas_[i].size;
    if (total > max_bytes) return 3;

    out.reserve(total);
    for (size_t i = lo_idx; i <= hi_idx; ++i) {
        const AudioMeta& m = metas_[i];
        read_at(m.off, m.size, out);
    }
    return 0;
}

} // namespace dw
