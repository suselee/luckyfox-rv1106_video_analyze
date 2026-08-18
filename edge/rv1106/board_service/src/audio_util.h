// 音频辅助: RTSP SDP 音频段解析 + AAC AudioSpecificConfig / ADTS 头构建。
// 纯函数, 无板端依赖, 主机侧可单测 (tests/audio_test.cpp)。
#pragma once
#include <ctype.h>
#include <stdint.h>
#include <string>
#include <vector>

namespace dw {

// SDP m=audio 段的解析结果。
struct SdpAudio {
    bool     ok = false;
    std::string codec;        // "PCMU" / "PCMA" / "AAC" (mpeg4-generic) / 其他
    int      rate = 8000;     // 采样率 (rtpmap)
    int      channels = 1;    // 声道数 (rtpmap)
    std::string control;      // 媒体级 a=control (SETUP URL)
    std::string fmtp;         // a=fmtp 全值 (AAC 用 config=xxxx)
};

// 从 SDP 文本解析第一个 m=audio 媒体段。
// 支持 rtpmap 静态/动态 PT (PCMU/PCMA/MPEG4-GENERIC), 忽略其余编码。
SdpAudio parse_audio_track(const std::string& sdp);

// 解析 a=fmtp 里的 config=hex (RFC 3640 AudioSpecificConfig, 通常 2 字节):
// 输出 object_type / sampling_frequency_index / channel_configuration。
bool parse_audio_specific_config(const std::string& fmtp,
                                 int& object_type,
                                 int& sampling_frequency_index,
                                 int& channel_configuration);

// 构建 7 字节 ADTS 头 (AAC LC, 无 CRC)。
// profile: object_type - 1 (AAC LC=2 → 1)。
void build_adts_header(uint8_t out[7], int profile,
                       int sampling_frequency_index, int channel_configuration,
                       int frame_length);

// 把一包 RTP 音频 payload 追加到 out (PCMU/PCMA 透传, AAC 包 ADTS)。
// object_type/sf_index/chan_cfg 为 AAC ASC 参数 (来自 fmtp config=)。
// 返回追加的字节数 (0 = 不识别/无效包)。
size_t append_audio_payload(std::vector<uint8_t>& out,
                            const SdpAudio& audio,
                            int object_type, int sf_index, int chan_cfg,
                            const uint8_t* payload, size_t len);

// 十六进制字符 → 值 (解析 config= 用)。
inline int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

} // namespace dw
