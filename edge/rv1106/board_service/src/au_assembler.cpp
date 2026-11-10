#include "au_assembler.h"

#include <string.h>
#include <string>
#include <algorithm>

namespace dw {

namespace {
const size_t kNoPos = (size_t)-1;

// NAL 起始码长度 (3 或 4), 用于定位 header 字节
inline size_t start_code_len(const std::vector<uint8_t>& nal) {
    if (nal.size() >= 4 && nal[0] == 0 && nal[1] == 0 && nal[2] == 0 && nal[3] == 1) return 4;
    if (nal.size() >= 3 && nal[0] == 0 && nal[1] == 0 && nal[2] == 1) return 3;
    return 0;
}
const size_t kMaxAuBytes = 512 * 1024;   // 单 AU 上限 (4K 高分片段的 I 帧也够用)

// 在 [pos, end) 里找下一个起始码, 返回起始码首字节位置; 未找到返回 npos。
size_t find_start_code(const std::vector<uint8_t>& buf, size_t from, size_t* sc_len) {
    for (size_t i = from; i + 3 <= buf.size(); ++i) {
        if (buf[i] == 0 && buf[i + 1] == 0) {
            if (buf[i + 2] == 1) { *sc_len = 3; return i; }
            if (i + 4 <= buf.size() && buf[i + 2] == 0 && buf[i + 3] == 1) {
                *sc_len = 4;
                return i;
            }
        }
    }
    return kNoPos;
}

inline int nal_type_of(uint8_t header) { return header & 0x1f; }
} // namespace

void AuAssembler::flush(std::vector<std::vector<uint8_t>>& out) {
    // 尾部可能还压着最后一个 NAL (没有后续起始码结尾): 把它当作完整 NAL 收进 AU
    size_t sc_len = 0;
    size_t start = find_start_code(pending_, 0, &sc_len);
    if (start != kNoPos) {
        std::vector<uint8_t> nal(pending_.begin() + start, pending_.end());
        pending_.clear();
        int type = nal_type_of(nal[sc_len]);
        if (type == 7) last_sps_ = nal;
        else if (type == 8) last_pps_ = nal;
        au_nals_.push_back(std::move(nal));
        if (type == 1 || type == 5) {
            au_has_vcl_ = true;
            if (type == 5) au_has_idr_ = true;
        }
    }
    pending_.clear();
    close_au(out);
}

void AuAssembler::reset() {
    pending_.clear();
    au_nals_.clear();
    last_sps_.clear();
    last_pps_.clear();
    au_has_vcl_ = false;
    au_has_idr_ = false;
}

void AuAssembler::append_nals(std::vector<uint8_t>& dst,
                              const std::vector<std::vector<uint8_t>>& nals) const {
    for (size_t i = 0; i < nals.size(); ++i)
        dst.insert(dst.end(), nals[i].begin(), nals[i].end());
}

void AuAssembler::close_au(std::vector<std::vector<uint8_t>>& out) {
    if (!au_nals_.empty()) {
        ++aus_seen_;
        if (au_has_idr_) {
            ++idr_aus_;
            // 补齐参数集: 有些摄像头把 SPS/PPS 单独放在上一个 AU 里
            bool has_sps = false, has_pps = false;
            for (size_t i = 0; i < au_nals_.size(); ++i) {
                size_t sc = start_code_len(au_nals_[i]);
                if (sc == 0 || au_nals_[i].size() <= sc) continue;
                int t = nal_type_of(au_nals_[i][sc]);
                if (t == 7) has_sps = true;
                if (t == 8) has_pps = true;
            }
            std::vector<uint8_t> au;
            au.reserve(4096);
            if (!has_sps && !last_sps_.empty()) au.insert(au.end(), last_sps_.begin(), last_sps_.end());
            if (!has_pps && !last_pps_.empty()) au.insert(au.end(), last_pps_.begin(), last_pps_.end());
            append_nals(au, au_nals_);
            if (!au.empty()) out.push_back(std::move(au));
        }
    }
    au_nals_.clear();
    au_has_vcl_ = false;
    au_has_idr_ = false;
}

void AuAssembler::feed(const uint8_t* data, int len,
                       std::vector<std::vector<uint8_t>>& out) {
    if (!data || len <= 0) return;
    pending_.insert(pending_.end(), data, data + len);

    size_t search_from = 0;
    while (true) {
        size_t sc_len = 0;
        size_t start = find_start_code(pending_, search_from, &sc_len);
        if (start == kNoPos) {
            // 保留最后 3 字节, 防止起始码被切在两个块之间
            if (pending_.size() > 3) pending_.erase(pending_.begin(), pending_.end() - 3);
            return;
        }
        size_t nal_begin = start + sc_len;
        size_t next_len = 0;
        size_t next = find_start_code(pending_, nal_begin, &next_len);
        if (next == kNoPos) {
            // 当前 NAL 还没收完: 丢掉起始码之前的垃圾字节后等待
            if (start > 0) pending_.erase(pending_.begin(), pending_.begin() + start);
            return;
        }
        // 完整 NAL: [start, next)
        std::vector<uint8_t> nal(pending_.begin() + start, pending_.begin() + next);
        pending_.erase(pending_.begin(), pending_.begin() + next);
        search_from = 0;

        int type = nal_type_of(nal[sc_len]);

        // AU 边界: 已有 VCL 后又来一个 VCL, 或非 VCL(SPS/PPS/SEI/AUD) 出现在 VCL 之后
        bool is_vcl = (type == 1 || type == 5);
        if (au_has_vcl_ && (!is_vcl || is_vcl)) {
            // 任何新 NAL 在 VCL 之后都意味着上一个 AU 结束 (简化且安全的判断)
            close_au(out);
        }

        if (type == 7) last_sps_ = nal;
        else if (type == 8) last_pps_ = nal;

        au_nals_.push_back(std::move(nal));
        if (is_vcl) {
            au_has_vcl_ = true;
            if (type == 5) au_has_idr_ = true;
        }

        size_t au_bytes = 0;
        for (size_t i = 0; i < au_nals_.size(); ++i) au_bytes += au_nals_[i].size();
        if (au_bytes > kMaxAuBytes) close_au(out);   // 防御: 异常码流不无限增长
    }
}

} // namespace dw
