#pragma once
#include <cstdint>
#include <string>

namespace dw {

// 码流无关的 NAL 分类 (H.264 Annex-B / H.265 Annex-B 共用)。
// H264: nal_type = header & 0x1F;  H265: nal_type = (header[0] >> 1) & 0x3F
struct NalKind {
    bool key;  // 关键帧 (H264: 5=IDR; H265: 16..21=IRAP)
    bool ps;   // 参数集 (H264: 7/8 = SPS/PPS; H265: 32/33/34 = VPS/SPS/PPS)
    bool vcl;  // 视频编码层切片 (逐个 = 一帧)
};

inline NalKind classify_nal(const std::string& codec, uint8_t type) {
    NalKind k = {false, false, false};
    if (codec == "H264") {
        k.ps  = type == 7 || type == 8;
        k.key = type == 5;
        k.vcl = type >= 1 && type <= 5;
    } else if (codec == "H265") {
        k.ps  = type == 32 || type == 33 || type == 34;
        k.key = type >= 16 && type <= 21;
        k.vcl = type <= 31;
    }
    return k;
}

// 日志用的人可读名称
inline const char* nal_name(const std::string& codec, uint8_t type) {
    if (codec == "H264") {
        switch (type) {
            case 1: return "SLICE"; case 5: return "IDR"; case 6: return "SEI";
            case 7: return "SPS";   case 8: return "PPS"; case 9: return "AUD";
            default: return "H264-other";
        }
    }
    switch (type) {
        case 0:  return "TRAIL_N"; case 1: return "TRAIL_R";
        case 19: return "IDR_W_RADL"; case 20: return "IDR_N_LP";
        case 21: return "CRA_NUT"; case 32: return "VPS"; case 33: return "SPS";
        case 34: return "PPS"; case 35: return "AUD"; case 39: return "SEI";
        default: return "H265-other";
    }
}

} // namespace dw