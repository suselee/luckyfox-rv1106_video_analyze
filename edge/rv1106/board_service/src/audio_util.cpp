#include "audio_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace dw {

static std::string trim_line(const std::string& value) {
    size_t begin = 0;
    while (begin < value.size() &&
           (value[begin] == ' ' || value[begin] == '\t' ||
            value[begin] == '\r' || value[begin] == '\n')) {
        begin++;
    }
    size_t end = value.size();
    while (end > begin &&
           (value[end - 1] == ' ' || value[end - 1] == '\t' ||
            value[end - 1] == '\r' || value[end - 1] == '\n')) {
        end--;
    }
    return value.substr(begin, end - begin);
}

static std::string lower_ascii(std::string value) {
    for (char& c : value) c = (char)tolower((unsigned char)c);
    return value;
}

SdpAudio parse_audio_track(const std::string& sdp) {
    SdpAudio out;
    bool in_audio = false;
    bool saw_audio_m = false;
    int m_pt = -1;
    size_t pos = 0;

    while (pos <= sdp.size()) {
        size_t end = sdp.find('\n', pos);
        if (end == std::string::npos) end = sdp.size();
        std::string line = trim_line(sdp.substr(pos, end - pos));
        std::string lower = lower_ascii(line);

        if (lower.compare(0, 2, "m=") == 0) {
            if (in_audio) break;
            in_audio = lower.compare(0, 8, "m=audio ") == 0 ||
                       lower.compare(0, 7, "m=audio") == 0;
            if (in_audio) {
                saw_audio_m = true;
                // m=audio 0 RTP/AVP 8  -> 静态 PT=8 (PCMA/8000)
                size_t avp = lower.find("rtp/avp");
                if (avp != std::string::npos) {
                    size_t pt_start = lower.find_first_not_of(" \t", avp + 7);
                    if (pt_start != std::string::npos)
                        m_pt = atoi(lower.c_str() + pt_start);
                }
            }
        } else if (in_audio) {
            if (lower.compare(0, 9, "a=rtpmap:") == 0) {
                size_t space = lower.find(' ');
                if (space != std::string::npos) {
                    size_t slash = lower.find('/', space + 1);
                    std::string name = lower.substr(
                        space + 1,
                        slash == std::string::npos ? std::string::npos
                                                   : slash - space - 1);
                    if (name == "pcma") out.codec = "PCMA";
                    else if (name == "pcmu") out.codec = "PCMU";
                    else if (name == "mpeg4-generic") out.codec = "AAC";
                    // 其他编码 (G726/G722/MP2...) 不识别, out.codec 留空
                    if (slash != std::string::npos && !out.codec.empty()) {
                        size_t slash2 = lower.find('/', slash + 1);
                        out.rate = atoi(lower.c_str() + slash + 1);
                        out.channels =
                            slash2 == std::string::npos
                                ? 1
                                : atoi(lower.c_str() + slash2 + 1);
                        if (out.rate <= 0) out.rate = 8000;
                        if (out.channels <= 0) out.channels = 1;
                    }
                }
            } else if (lower.compare(0, 10, "a=control:") == 0) {
                out.control = trim_line(line.substr(10));
            } else if (lower.compare(0, 7, "a=fmtp:") == 0) {
                out.fmtp = trim_line(line.substr(7));
            }
        }

        if (end == sdp.size()) break;
        pos = end + 1;
    }

    // 静态 PT 回退 (RFC 3551): rtpmap 缺失时按 m= 行的 PT 推断 G711。
    if (saw_audio_m && out.codec.empty()) {
        if (m_pt == 8) {
            out.codec = "PCMA";
            out.rate = 8000;
            out.channels = 1;
        } else if (m_pt == 0) {
            out.codec = "PCMU";
            out.rate = 8000;
            out.channels = 1;
        }
    }

    out.ok = !out.codec.empty();
    return out;
}

bool parse_audio_specific_config(const std::string& fmtp,
                                 int& object_type,
                                 int& sampling_frequency_index,
                                 int& channel_configuration) {
    // a=fmtp:96 streamtype=5; profile-level-id=1; mode=AAC-hbr;
    //         sizelength=13; indexlength=3; indexdeltalength=3; config=1210
    size_t pos = fmtp.find("config=");
    if (pos == std::string::npos) {
        std::string lower = lower_ascii(fmtp);
        pos = lower.find("config=");
        if (pos == std::string::npos) return false;
    }
    pos += 7;
    uint8_t bytes[4] = {0, 0, 0, 0};
    int n = 0;
    while (pos < fmtp.size() && n < 4) {
        int hi = hex_digit(fmtp[pos]);
        int lo = pos + 1 < fmtp.size() ? hex_digit(fmtp[pos + 1]) : -1;
        if (hi < 0) break;
        bytes[n++] = (uint8_t)((hi << 4) | (lo < 0 ? 0 : lo));
        pos += 2;
    }
    if (n < 1) return false;

    object_type = (bytes[0] >> 3) & 0x1F;
    sampling_frequency_index = (int)(((bytes[0] & 0x07) << 1) |
                                     ((n > 1 ? bytes[1] : 0) >> 7));
    channel_configuration = (int)((n > 1 ? bytes[1] : 0) >> 3) & 0x0F;
    return true;
}

void build_adts_header(uint8_t out[7], int profile,
                       int sampling_frequency_index, int channel_configuration,
                       int frame_length) {
    // 保护不足: frame_length <= 8191 (13 bit)。
    if (frame_length > 8191) frame_length = 8191;
    out[0] = 0xFF;                                     // syncword 高 8 位
    out[1] = 0xF1;                                     // syncword 低 4 + MPEG-4(00) + layer(00) + 无 CRC(1)
    out[2] = (uint8_t)(((profile & 0x03) << 6) |       // profile (2 bit)
                       ((sampling_frequency_index & 0x0F) << 2) |  // sf idx (4 bit)
                       ((channel_configuration >> 2) & 0x01));     // chan cfg 高 1 bit
    out[3] = (uint8_t)(((channel_configuration & 0x03) << 6) |     // chan cfg 低 2 bit
                       ((frame_length >> 11) & 0x03));             // frame len 高 2 bit
    out[4] = (uint8_t)((frame_length >> 3) & 0xFF);                // frame len 中 8 bit
    out[5] = (uint8_t)(((frame_length & 0x07) << 5) | 0x1F);       // frame len 低 3 + buffer fullness 高 5
    out[6] = 0xFC;                                     // buffer fullness 低 6 + num_raw_blocks(00)
}

size_t append_audio_payload(std::vector<uint8_t>& out,
                            const SdpAudio& audio,
                            int object_type, int sf_index, int chan_cfg,
                            const uint8_t* payload, size_t len) {
    if (!audio.ok || len == 0 || payload == NULL) return 0;

    if (audio.codec == "PCMU" || audio.codec == "PCMA") {
        out.insert(out.end(), payload, payload + len);
        return len;
    }
    if (audio.codec != "AAC") return 0;

#ifdef DW_DEBUG_AUDIO
    {
        static int dbg_count = 0;
        if (dbg_count < 3) {
            printf("[AUDIO-DBG] pkt len=%zu hex:", len);
            for (size_t i = 0; i < len && i < 64; ++i)
                printf("%02x ", payload[i]);
            printf("\n");
            dbg_count++;
        }
    }
#endif

    // RFC 3640: 2 字节 AU-headers-length (bit 数, 通常 16), 随后 AU-header。
    if (len < 4) return 0;
    int au_headers_len = ((int)payload[0] << 8) | payload[1];
    if (au_headers_len == 0) {
        // 无 AU 头: 整包视为单个 AU (部分设备的简化封装)。
        int profile = object_type > 0 ? object_type - 1 : 1;
        uint8_t adts[7];
        build_adts_header(adts, profile, sf_index, chan_cfg, (int)len - 2 + 7);
        out.insert(out.end(), adts, adts + 7);
        out.insert(out.end(), payload + 2, payload + len);
        return (size_t)len - 2 + 7;
    }
    int au_headers_bytes = (au_headers_len + 7) / 8;
    if (au_headers_bytes > (int)len - 2 || au_headers_bytes < 2) return 0;

    // 每个 AU-header 16 bit: [AU-size 13 bit][AU-index 3 bit]。
    // RFC 3640 规定 AU-size 单位为 bit, 但部分设备 (TP-LINK 实测) 直接按字节
    // 填入; 用"所有 AU 大小之和 == 剩余数据长度"的一致性校验来兼容两种。
    int n_au = au_headers_bytes / 2;
    std::vector<int> au_sizes;
    au_sizes.reserve((size_t)n_au);
    int pos = 2;
    for (int i = 0; i < n_au && pos + 1 < (int)len; ++i, pos += 2) {
        au_sizes.push_back(((int)(payload[pos] & 0xFF) << 5) |
                           ((payload[pos + 1] >> 3) & 0x1F));
    }

    int data_total = (int)len - 2 - au_headers_bytes;
    int sum_bits = 0, sum_bytes = 0;
    for (size_t i = 0; i < au_sizes.size(); ++i) {
        sum_bits += (au_sizes[i] + 7) / 8;
        sum_bytes += au_sizes[i];
    }
    bool size_in_bytes = (sum_bytes == data_total) && (sum_bits != data_total);
    bool size_in_bits = (sum_bits == data_total);
    if (!size_in_bits && !size_in_bytes) return 0;

    int data_pos = 2 + au_headers_bytes;
    int profile = object_type > 0 ? object_type - 1 : 1;
    size_t before = out.size();
    for (size_t i = 0; i < au_sizes.size(); ++i) {
        int au_size_bytes = size_in_bytes ? au_sizes[i] : (au_sizes[i] + 7) / 8;
        if (au_size_bytes <= 0 || data_pos + au_size_bytes > (int)len) break;
        uint8_t adts[7];
        build_adts_header(adts, profile, sf_index, chan_cfg,
                          au_size_bytes + 7);
        out.insert(out.end(), adts, adts + 7);
        out.insert(out.end(), payload + data_pos, payload + data_pos + au_size_bytes);
        data_pos += au_size_bytes;
    }
    return out.size() - before;
}

} // namespace dw
