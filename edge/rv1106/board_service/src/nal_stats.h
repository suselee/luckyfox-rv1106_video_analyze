#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "nal_util.h"

namespace dw {

// 逐 NAL 解析: 跨 chunk 缓冲当前 NAL, 遇到下一个起始码时整段回调。
// 供探针/环形缓冲/切片共用; 纯逻辑, 无 IO, 可在主机单元测试。
class NalScanner {
public:
    typedef std::function<void(const uint8_t* nal, size_t len, double ts)> Callback;
    static const size_t MAX_NAL_BYTES = 4 * 1024 * 1024;

    void feed(const uint8_t* data, size_t len, double ts);

    // 流末尾: 把残留的未闭合 NAL 也回调一次。
    void flush(double ts);

    void set_callback(const Callback& cb) { cb_ = cb; }
    size_t pending_bytes() const { return cur_.size(); }

private:
    void emit(double ts);

    std::vector<uint8_t> cur_;
    Callback cb_;
};

// 探针统计: 码流层面的总量/GOP/参数集/窗口码率。
struct NalStats {
    const std::string* codec = nullptr;
    double window_secs = 5.0;
    unsigned long long nals = 0;
    unsigned long long key_count = 0;
    unsigned long long vcl_count = 0;
    unsigned long long bytes = 0;
    unsigned long long sps_count = 0, pps_count = 0, vps_count = 0;
    unsigned long long sps_bytes = 0, pps_bytes = 0, vps_bytes = 0;
    unsigned long long sei_count = 0;
    std::vector<double> gop_secs;    // 相邻关键帧间隔
    std::vector<double> window_kbps; // 每窗口码率
    double last_key_ts = -1.0;
    double window_start = 0.0;
    unsigned long long window_bytes = 0;
    unsigned long long gaps = 0;
    double last_data_ts = -1.0;

    void on_nal(const uint8_t* nal, size_t len, double ts);
    void on_chunk(size_t /*len*/, double /*ts*/) {}
    void print_summary(double total_secs) const;
};

} // namespace dw