#include "h264_source.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <poll.h>
#include <time.h>
#include <ctype.h>

namespace dw {

// ---- URL 解析 ------------------------------------------------------------
void H264Source::parse_url(const std::string& url) {
    // rtsp://[user:pass@]host[:port]/path
    const char* p = url.c_str();
    if (strncmp(p, "rtsp://", 7) == 0) p += 7;
    else if (strncmp(p, "rtsps://", 8) == 0) p += 8;

    // user:pass@
    const char* at = strchr(p, '@');
    if (at) {
        const char* colon = (const char*)memchr(p, ':', at - p);
        if (colon) {
            user_.assign(p, colon - p);
            pass_.assign(colon + 1, at - colon - 1);
        } else {
            user_.assign(p, at - p);
        }
        p = at + 1;
    }

    // host[:port]
    const char* slash = strchr(p, '/');
    const char* col   = (const char*)memchr(p, ':', slash ? (slash - p) : strlen(p));
    if (col) {
        host_.assign(p, col - p);
        if (slash)
            port_ = atoi(col + 1);
        else
            port_ = atoi(col + 1); // col is the last colon before end - but this shouldn't happen with RTSP urls
        p = slash ? slash : "";
    } else if (slash) {
        host_.assign(p, slash - p);
        p = slash;
    } else {
        host_.assign(p);
        p = "";
    }
    if (*p == '/') path_ = p; else path_ = "/";
}

// ---- Base64 ----------------------------------------------------------------
std::string H264Source::base64_encode(const uint8_t* data, int len) {
    static const char T[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (int i = 0; i < len; i += 3) {
        uint32_t v = (uint32_t)data[i] << 16;
        if (i + 1 < len) v |= (uint32_t)data[i + 1] << 8;
        if (i + 2 < len) v |= (uint32_t)data[i + 2];
        out.push_back(T[(v >> 18) & 0x3F]);
        out.push_back(T[(v >> 12) & 0x3F]);
        out.push_back((i + 1 < len) ? T[(v >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < len) ? T[v & 0x3F] : '=');
    }
    return out;
}

std::string H264Source::basic_auth() const {
    if (user_.empty()) return "";
    std::string cred = user_ + ":" + pass_;
    return "Authorization: Basic " + base64_encode((const uint8_t*)cred.data(), cred.size()) + "\r\n";
}

// ---- TCP ------------------------------------------------------------------
bool H264Source::tcp_connect() {
    tcp_close();
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(port_);
    struct addrinfo* res = nullptr;
    if (::getaddrinfo(host_.c_str(), port_str.c_str(), &hints, &res) != 0 || !res)
        return false;

    int fd = -1;
    for (struct addrinfo* r = res; r; r = r->ai_next) {
        fd = ::socket(r->ai_family, r->ai_socktype, r->ai_protocol);
        if (fd < 0) continue;
        if (!bind_ip_.empty()) {
            struct addrinfo bh, *bres = nullptr;
            memset(&bh, 0, sizeof(bh));
            bh.ai_family = AF_INET;
            bh.ai_socktype = SOCK_STREAM;
            if (::getaddrinfo(bind_ip_.c_str(), "0", &bh, &bres) == 0 && bres) {
                ::bind(fd, bres->ai_addr, bres->ai_addrlen);
                ::freeaddrinfo(bres);
            }
        }
        if (::connect(fd, r->ai_addr, r->ai_addrlen) == 0) break;
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(res);
    if (fd < 0) return false;

    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    // 大接收缓冲: 本进程主循环有 NPU 检测等耗时环节, 读速率远低于
    // 摄像头推流速率 (35 包/s); 小缓冲会让摄像头发送阻塞, 固件随后
    // 静默踢断会话 (IDR 分片只发前几片, 板端永远组不齐关键帧)。
    int rcvbuf = 512 * 1024;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    int sndbuf = 128 * 1024;
    ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

    // 接收超时: 用手动 poll() 替代 SO_RCVTIMEO, 可移植性更好
    sock_ = fd;
    return true;
}

bool H264Source::tcp_send(const void* buf, int len) {
    const uint8_t* p = (const uint8_t*)buf;
    while (len > 0) {
        ssize_t n = ::send(sock_, p, (size_t)len, MSG_NOSIGNAL);
        if (n <= 0) return false;
        p   += n;
        len -= (int)n;
    }
    return true;
}

int H264Source::tcp_recv(uint8_t* buf, int len, int timeout_ms) {
    struct pollfd pfd;
    pfd.fd      = sock_;
    pfd.events  = POLLIN;
    int ret = ::poll(&pfd, 1, timeout_ms);
    if (ret < 0) return -1;
    if (ret == 0) return 0;
    ssize_t n = ::recv(sock_, buf, (size_t)len, 0);
    if (n < 0) return -1;
    if (n == 0) return -2;  // EOF: 对端主动关闭 (摄像头周期踢连接 ~7-9s)
    return (int)n;
}

void H264Source::tcp_close() {
    if (sock_ >= 0) { ::close(sock_); sock_ = -1; }
}

// ---- RTSP 协议 -------------------------------------------------------------
int H264Source::rtsp_req(const std::string& method, const std::string& url,
                         const std::string& extra_hdr,
                         std::string& out_headers, std::string& out_body) {
    cseq_++;
    out_headers.clear();
    out_body.clear();

    // 与 ffprobe 一致: 首次不带 Authorization, 收到 401 后带 Basic 重试。
    // 摄像头固件疑似按"是否走过 401 认证流程"区分正规客户端:
    // 直接带 auth 的会话被按短配额推流 (~3KB 后静默)。
    bool auth_done = false;
    for (int attempt = 0; attempt < 2; attempt++) {
        std::string auth = auth_done ? basic_auth() : "";
        char req[1024];
        int n = snprintf(req, sizeof(req),
                         "%s %s RTSP/1.0\r\nCSeq: %d\r\n%s%s"
                         "User-Agent: Lavf61.1.100\r\n\r\n",
                         method.c_str(), url.c_str(), cseq_,
                         auth.c_str(), extra_hdr.c_str());
        if (!tcp_send(req, n)) return -1;

        // 读响应头 (双 CRLF 截止)
        std::string hdr;
        size_t cap = 4096;
        hdr.reserve(cap);
        uint8_t byte;
        int crlf = 0;
        while (hdr.size() < cap) {
            if (tcp_recv(&byte, 1, timeout_) <= 0) return -1;
            hdr.push_back((char)byte);
            if (byte == '\r' || byte == '\n') {
                crlf++;
                if (byte == '\n' && crlf >= 4) break;  // \r\n\r\n → 4
            } else {
                crlf = 0;
            }
        }
        out_headers = hdr;

        // 解析 status code
        if (hdr.size() < 12) return -1;
        int sc = atoi(hdr.c_str() + 9);  // "RTSP/1.0 xxx"
        if (sc <= 0) return -1;

        if (sc == 401 && !auth_done && !user_.empty()) {
            auth_done = true;
            continue;
        }

        // 读 body (如果有 Content-Length)
        auto cl_pos = hdr.find("Content-Length:");
        if (cl_pos == std::string::npos) cl_pos = hdr.find("content-length:");
        if (cl_pos != std::string::npos) {
            int cl = atoi(hdr.c_str() + cl_pos + 15);
            if (cl > 0 && cl < 65536) {
                out_body.resize((size_t)cl);
                for (int i = 0; i < cl; ) {
                    int r = tcp_recv((uint8_t*)out_body.data() + i, cl - i,
                                     timeout_);
                    if (r <= 0) return -1;
                    i += r;
                }
            }
        }
        return sc;
    }
    return -1;
}

// ---- Session / Transport 解析辅助 -----------------------------------------
static std::string hdr_val(const std::string& hdrs, const char* key) {
    // 大小写不敏感查找, 返回 : 后到 \r\n 之间的值(去首尾空白)
    std::string lk(key);
    for (char& c : lk) c = tolower(c);
    std::string lh(hdrs);
    for (char& c : lh) c = tolower(c);
    size_t pos = lh.find(lk);
    if (pos == std::string::npos) return "";
    pos = lh.find(':', pos);
    if (pos == std::string::npos) return "";
    size_t end = lh.find("\r\n", pos);
    if (end == std::string::npos) return "";
    pos++;
    while (pos < end && (lh[pos] == ' ' || lh[pos] == '\t')) pos++;
    size_t r = end;
    while (r > pos && (lh[r-1] == ' ' || lh[r-1] == '\t')) r--;
    return hdrs.substr(pos, r - pos);
}

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

// Locate the first video media section and return its codec/control value.
// The media-level a=control line, not the session-level a=control:*, is the
// URL that must be used by SETUP.
static bool parse_video_track(const std::string& sdp,
                              std::string& codec,
                              std::string& control) {
    codec.clear();
    control.clear();
    bool in_video = false;
    bool saw_video = false;
    size_t pos = 0;

    while (pos <= sdp.size()) {
        size_t end = sdp.find('\n', pos);
        if (end == std::string::npos) end = sdp.size();
        std::string line = trim_line(sdp.substr(pos, end - pos));
        std::string lower = lower_ascii(line);

        if (lower.compare(0, 2, "m=") == 0) {
            if (in_video) break;
            in_video = lower.compare(0, 8, "m=video ") == 0;
            if (in_video) saw_video = true;
        } else if (in_video) {
            if (lower.compare(0, 9, "a=rtpmap:") == 0) {
                size_t space = lower.find(' ');
                if (space != std::string::npos) {
                    size_t slash = lower.find('/', space + 1);
                    std::string name = lower.substr(
                        space + 1,
                        slash == std::string::npos ? std::string::npos
                                                   : slash - space - 1);
                    if (name == "h264") codec = "H264";
                    else if (codec.empty() && (name == "h265" || name == "hevc"))
                        codec = "H265";
                }
            } else if (lower.compare(0, 10, "a=control:") == 0) {
                control = trim_line(line.substr(10));
            }
        }

        if (end == sdp.size()) break;
        pos = end + 1;
    }
    return saw_video;
}

static std::string rtsp_origin(const std::string& url) {
    size_t scheme = url.find("://");
    if (scheme == std::string::npos) return "";
    size_t path = url.find('/', scheme + 3);
    return path == std::string::npos ? url : url.substr(0, path);
}

static std::string resolve_control_url(const std::string& base,
                                       const std::string& content_base,
                                       const std::string& control) {
    if (control.empty() || control == "*") return base;
    std::string lower = lower_ascii(control);
    if (lower.compare(0, 7, "rtsp://") == 0 ||
        lower.compare(0, 8, "rtsps://") == 0) {
        return control;
    }
    if (control[0] == '/') return rtsp_origin(base) + control;

    std::string parent = content_base.empty() ? base : content_base;
    if (!parent.empty() && parent[parent.size() - 1] != '/') parent += '/';
    return parent + control;
}

bool H264Source::rtsp_handshake() {
    std::string base = "rtsp://" + host_;
    if (port_ != 554) base += ":" + std::to_string(port_);
    base += path_;

    std::string hdrs, body;

    // OPTIONS
    if (rtsp_req("OPTIONS", base, "", hdrs, body) < 0) {
        printf("[RTSP] OPTIONS failed\n");
        return false;
    }

    // DESCRIBE
    if (rtsp_req("DESCRIBE", base, "Accept: application/sdp\r\n", hdrs, body) != 200) {
        printf("[RTSP] DESCRIBE failed\n");
        return false;
    }

    std::string codec, control;
    if (!parse_video_track(body, codec, control)) {
        printf("[RTSP] SDP has no video media section (body=%zu bytes)\n",
               body.size());
        return false;
    }
    if (codec != "H264" && codec != "H265") {
        printf("[RTSP] unsupported video codec: %s (need H264/H265)\n",
               codec.empty() ? "unknown" : codec.c_str());
        return false;
    }
    codec_ = codec;

    // 解析音频轨 (必须在后续 SETUP 请求覆盖 body 之前)。
    audio_ = want_audio_ ? parse_audio_track(body) : SdpAudio();
    // (音频协商日志已静默, 避免重连时刷日志)

    std::string content_base = hdr_val(hdrs, "Content-Base");
    if (content_base.empty()) content_base = hdr_val(hdrs, "Content-Location");
    std::string ctrl_url = resolve_control_url(base, content_base, control);

    // SETUP the SDP-advertised video track over RTSP interleaved TCP.
    // 必须带 unicast: 摄像头 (TP-LINK) 对无 unicast 的 SETUP 走兼容路径,
    // 会话被快速踢断 (实测 ffprobe 带 unicast 60s 连续, 本客户端不带 2-3s 断)。
    int sc = rtsp_req("SETUP", ctrl_url,
                      "Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n",
                      hdrs, body);
    if (sc != 200) {
        printf("[RTSP] SETUP failed (status=%d)\n", sc);
        return false;
    }

    session_ = hdr_val(hdrs, "Session");
    if (session_.empty()) {
        printf("[RTSP] SETUP: no Session header\n");
        return false;
    }
    // Session 值可能带 timeout 后缀, 取分号前的部分
    size_t semi = session_.find(';');
    if (semi != std::string::npos) session_.resize(semi);

    // 可选: SETUP 音频轨 (interleaved=2-3)。摄像头无音频/不支持时静默降级。
    if (audio_.ok) {
        std::string audio_ctrl = resolve_control_url(base, content_base,
                                                     audio_.control);
        int sc2 = rtsp_req("SETUP", audio_ctrl,
                           "Transport: RTP/AVP/TCP;unicast;interleaved=2-3\r\n"
                           "Session: " + session_ + "\r\n",
                           hdrs, body);
        if (sc2 != 200) {
            printf("[RTSP] audio SETUP failed (status=%d); continuing video-only\n", sc2);
            audio_ = SdpAudio();
        } else if (audio_.codec == "AAC") {
            int ot, sf, ch;
            if (!parse_audio_specific_config(audio_.fmtp, ot, sf, ch)) {
                printf("[RTSP] AAC fmtp missing config; audio disabled\n");
                audio_ = SdpAudio();
            } else {
                audio_object_type_ = ot;
                audio_sf_index_ = sf;
                audio_chan_cfg_ = ch;
            }
        }
    }

    // PLAY: 用 SDP 的 Content-Base (含端口与尾斜杠, 与 ffprobe 对齐)。
    // 摄像头对 URL 与 Content-Base 不一致的会话疑似走"短配额"路径
    // (每会话只推 ~3KB 就静默)。
    std::string play_url = content_base.empty() ? base : content_base;
    if (!play_url.empty() && play_url[play_url.size() - 1] != '/')
        play_url += '/';
    if (rtsp_req("PLAY", play_url,
                 "Session: " + session_ + "\r\nRange: npt=0.000-\r\n",
                 hdrs, body) != 200) {
        printf("[RTSP] PLAY failed\n");
        return false;
    }

    // 成功路径静默 (摄像头 ~7-9s 周期踢连接, 每次重连都打会刷爆日志);
    // 失败路径 (上面各 printf) 保留诊断信息。
    return true;
}

// ---- RTP / H.264 -----------------------------------------------------------
int H264Source::read_rtp_packet(uint8_t* buf, int max_len, int& ch) {
    // 读 interleaved 帧头: $ <ch> <len_hi> <len_lo>
    uint8_t ih[4];
    for (int i = 0; i < 4; ) {
        int r = tcp_recv(ih + i, 4 - i, timeout_);
        if (r <= 0) return r; // 0=超时, -1=错误
        i += r;
    }
    if (ih[0] != '$') { printf("[RTP] bad interleaved magic 0x%02x\n", ih[0]); return -1; }
    ch = ih[1];
    uint16_t pl = ((uint16_t)ih[2] << 8) | ih[3];
    if ((int)pl > max_len) {
        printf("[RTP] oversize payload %u (buffer=%d)\n", pl, max_len);
        return -1;
    }

    for (int i = 0; i < (int)pl; ) {
        int r = tcp_recv(buf + i, pl - i, timeout_);
        if (r <= 0) return r;
        i += r;
    }

    if (ch == RTP_CHANNEL || ch == RTP_AUDIO_CHANNEL) {
        // 有效的 RTP 包
        return (int)pl;
    } else if (ch == RTCP_CHANNEL || ch == RTCP_AUDIO_CHANNEL) {
        // RTCP (SR/RR), 忽略, 继续读下一个
        return -3;
    }
    printf("[RTP] unexpected interleaved channel %d; ignored\n", ch);
    return 0;
}

// 剥 RTP 固定头/CSRC/扩展/填充, 返回 payload 长度; 无效返回 -1。
int H264Source::strip_rtp_header(const uint8_t* buf, int pl) const {
    if (pl < 12 || (buf[0] >> 6) != 2) {
        printf("[RTP] invalid header (len=%d)\n", pl);
        return -1;
    }
    int header_len = 12 + 4 * (buf[0] & 0x0F);
    if (header_len > pl) return -1;
    if (buf[0] & 0x10) {  // X: header extension present
        if (header_len + 4 > pl) return -1;
        int ext_words = ((int)buf[header_len + 2] << 8) | buf[header_len + 3];
        header_len += 4 + ext_words * 4;
        if (header_len > pl) return -1;
    }
    int payload_end = pl;
    if (buf[0] & 0x20) {  // P: trailing padding present
        int padding = buf[pl - 1];
        if (padding <= 0 || padding > pl - header_len) return -1;
        payload_end -= padding;
    }
    if (payload_end <= header_len) return -1;
    return payload_end - header_len;
}

void H264Source::append_nal(const uint8_t* data, int len) {
    const uint8_t start[] = {0x00, 0x00, 0x00, 0x01};
    out_buf_.insert(out_buf_.end(), start, start + 4);
    out_buf_.insert(out_buf_.end(), data, data + len);
}

void H264Source::depacketize_h264(const uint8_t* payload, int len) {
    // RTP payload header (RFC 6184): [NAL_HDR] [+ 1-byte FU header if NAL type >= 24]
    if (len < 1) return;
    uint8_t nal_hdr  = payload[0];
    uint8_t nal_type = nal_hdr & 0x1F;

    if (nal_type >= 1 && nal_type <= 23) {
        // 单 NAL 单元
        append_nal(payload, len);
    } else if (nal_type == 24) {
        // STAP-A (聚合包)
        int off = 1;
        while (off + 2 <= len) {
            uint16_t sz = ((uint16_t)payload[off] << 8) | payload[off + 1];
            off += 2;
            if (off + sz > len) break;
            append_nal(payload + off, sz);
            off += sz;
        }
    } else if (nal_type == 28) {
        // FU-A (分片)
        if (len < 2) return;
        if (frag_.buf.size() + (size_t)len > 16 * 1024 * 1024) {
            frag_.active = false;
            frag_.buf.clear();
            return;
        }
        uint8_t fu_hdr  = payload[1];
        bool    start   = (fu_hdr & 0x80) != 0;  // S bit
        bool    end     = (fu_hdr & 0x40) != 0;  // E bit
        uint8_t fu_type = fu_hdr & 0x1F;

        if (start) {
            frag_.buf.clear();
            frag_.active   = true;
            frag_.nal_type = fu_type;
            // 重构 NAL 头: (F|NRI 来自 original) | type
            uint8_t nh = (nal_hdr & 0xE0) | fu_type;
            frag_.buf.push_back(nh);
        }
        if (!frag_.active || fu_type != frag_.nal_type) {
            frag_.active = false;
            return;
        }
        frag_.buf.insert(frag_.buf.end(), payload + 2, payload + len);
        if (end) {
            append_nal(frag_.buf.data(), (int)frag_.buf.size());
            frag_.buf.clear();
            frag_.active = false;
        }
    }
    // 其他 NAL 类型 (0, 25-31) 忽略
}

// RFC 7798 HEVC RTP 解封装。
// 16-bit payload header: [F(1)][Type(6)][LayerId(6)][TID(3)]
// 假设无 DONL 字段 (sprop-max-don-diff=0, 绝大多数编码器如此)。
void H264Source::depacketize_h265(const uint8_t* payload, int len) {
    if (len < 2) return;
    uint8_t type = (payload[0] >> 1) & 0x3F;

    if (type < 48) {
        // 单 NAL 单元 (0~47): payload header 即 NAL header
        append_nal(payload, len);
    } else if (type == 48) {
        // AP (聚合包): 2 字节 payload header 后是 [2B size + NAL] 序列
        int off = 2;
        while (off + 2 <= len) {
            uint16_t sz = ((uint16_t)payload[off] << 8) | payload[off + 1];
            off += 2;
            if (off + sz > len) break;
            append_nal(payload + off, sz);
            off += sz;
        }
    } else if (type == 49) {
        // FU: 2 字节 payload header + 1 字节 FU header [S(1)][E(1)][FuType(6)]
        if (len < 3) return;
        // 防护: 4K IDR 帧的 FU 分片可达 MB 级; 异常流 (无 E 位) 会让
        // frag_ 无限增长耗尽内存。超过 16MB 直接丢弃该帧序列。
        if (frag_.buf.size() + (size_t)len > 16 * 1024 * 1024) {
            frag_.active = false;
            frag_.buf.clear();
            return;
        }
        uint8_t fu_hdr  = payload[2];
        bool    start   = (fu_hdr & 0x80) != 0;
        bool    end     = (fu_hdr & 0x40) != 0;
        uint8_t fu_type = fu_hdr & 0x3F;

        if (start) {
            frag_.buf.clear();
            // 重构原始 NAL header:
            // byte0 = F(来自 payload[0]) | FuType | LayerId LSB
            // byte1 = LayerId 高 5 位 | TID (即原 payload[1], 不变)
            uint8_t b0 = (payload[0] & 0x80) | (uint8_t)(fu_type << 1) |
                         (payload[0] & 0x01);
            frag_.buf.push_back(b0);
            frag_.buf.push_back(payload[1]);
            frag_.active   = true;
            frag_.nal_type = fu_type;
        }
        if (!frag_.active || (frag_.nal_type & 0xFF) != fu_type) {
            frag_.active = false;
            return;
        }
        frag_.buf.insert(frag_.buf.end(), payload + 3, payload + len);
        if (end) {
            append_nal(frag_.buf.data(), (int)frag_.buf.size());
            frag_.buf.clear();
            frag_.active = false;
        }
    }
}

// ---- 音频输出 ---------------------------------------------------------------
int H264Source::drain_audio_outbuf(uint8_t* buf, int max_len) {
    int avail = (int)audio_out_buf_.size() - audio_out_off_;
    if (avail <= 0) return 0;
    int n = avail < max_len ? avail : max_len;
    memcpy(buf, audio_out_buf_.data() + audio_out_off_, n);
    audio_out_off_ += n;
    if (audio_out_off_ >= (int)audio_out_buf_.size()) {
        audio_out_buf_.clear();
        audio_out_off_ = 0;
    }
    return n;
}

// RTP 音频解包: PCMU/PCMA 透传 payload; AAC (mpeg4-generic, RFC 3640)
// 剥 AU 头并逐个 AU 包成 ADTS 帧。
void H264Source::depacketize_audio(const uint8_t* payload, int len) {
    if (len <= 0 || !audio_.ok) return;
    append_audio_payload(audio_out_buf_, audio_,
                         audio_object_type_, audio_sf_index_, audio_chan_cfg_,
                         payload, (size_t)len);
}

// ---- 公共接口 ----------------------------------------------------------------
int H264Source::drain_outbuf(uint8_t* buf, int max_len) {
    int avail = (int)out_buf_.size() - out_off_;
    if (avail <= 0) return 0;
    int n = avail < max_len ? avail : max_len;
    memcpy(buf, out_buf_.data() + out_off_, n);
    out_off_ += n;
    if (out_off_ >= (int)out_buf_.size()) {
        out_buf_.clear();
        out_off_ = 0;
    }
    return n;
}

bool H264Source::open(const std::string& rtsp_url, bool with_audio) {
    close();
    parse_url(rtsp_url);
    // (连接日志已静默, 避免重连时刷日志)

    audio_ = SdpAudio();
    want_audio_ = with_audio;

    if (!tcp_connect()) {
        printf("[RTSP] TCP connect failed\n");
        return false;
    }

    // 音频协商需要 DESCRIBE 返回的 SDP 文本; 这里重读一份 body 供解析。
    if (!rtsp_handshake()) {
        tcp_close();
        return false;
    }

    out_buf_.clear();
    out_off_ = 0;
    audio_out_buf_.clear();
    audio_out_off_ = 0;
    frag_ = RtpFrag{};
    return true;
}

bool H264Source::open_stdin() {
    close();
    stdin_mode_ = true;
    sock_ = 0;  // fd 0 视为数据源
    codec_ = "H264";
    return true;
}

bool H264Source::open_pipe(const std::string& path) {
    close();
    int fd = ::open(path.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd < 0) {
        printf("[PIPE] open %s failed\n", path.c_str());
        return false;
    }
    stdin_mode_ = true;
    pipe_fd_ = fd;
    sock_ = fd;  // is_open() 复用 sock_ 判定
    codec_ = "H265";
    return true;
}

int H264Source::read_chunk(uint8_t* buf, int max_len) {
    if (stdin_mode_) {
        // ffprobe 管道: Annex-B 裸流, 直接透传
        int rfd = pipe_fd_ >= 0 ? pipe_fd_ : 0;
        struct pollfd pfd;
        pfd.fd = rfd;
        pfd.events = POLLIN;
        int ret = ::poll(&pfd, 1, timeout_);
        if (ret <= 0) return ret;  // 0=超时, -1=错误
        ssize_t n = ::read(rfd, buf, (size_t)max_len);
        if (n < 0) return -1;
        if (n == 0) return -2;  // ffprobe 退出 (管道 EOF)
        return (int)n;
    }

    int n = drain_outbuf(buf, max_len);
    if (n > 0) return n;

    // 输出缓冲空了, 从 socket 读 RTP 包。
    // 音频包 (channel 2) 会进入音频缓冲, 视频包继续解封装。
    int ch = -1;
    int pl = read_rtp_packet(buf, max_len, ch); // 复用 buf 做临时接收
    if (pl < 0) return pl;
    if (pl == 0) return 0; // 超时
    if (ch == RTP_AUDIO_CHANNEL) {
        int payload_len = strip_rtp_header(buf, pl);
        if (payload_len > 0) {
            int off = pl - payload_len;
            depacketize_audio(buf + off, payload_len);
        }
        // 音频包不是"无数据": 返回 -3 让调用方不把它计为 stall,
        // 避免密集音频包 (20ms 间隔) 误触 STALL_LIMIT 重连。
        return -3;
    }

    // 解析 RTP seq/SSRC 供 RTCP RR 使用。
    if (pl >= 12 && (buf[0] >> 6) == 2) {
        uint16_t seq = ((uint16_t)buf[2] << 8) | buf[3];
        uint32_t ssrc = ((uint32_t)buf[8] << 24) | ((uint32_t)buf[9] << 16) |
                        ((uint32_t)buf[10] << 8) | buf[11];
        if (!got_rtp_ || ssrc != ssrc_peer_) {
            ssrc_peer_ = ssrc;
            got_rtp_ = true;
            seq_cycles_ = 0;
        }
        if (seq < seq_last_ && (uint16_t)(seq_last_ - seq) > 0x7000)
            seq_cycles_++;
        seq_last_ = seq;
        ext_seq_ = (seq_cycles_ << 16) | seq;
        // 收到数据后立即补发 RR (5s 节流): 摄像头在等首个 RR 确认
        // 客户端活跃, 发晚了会话被踢 (实测 stream2 只给 2s 数据)。
        maybe_send_rtcp();
    }

    // Strip the RTP fixed header, CSRC list, optional extension and padding.
    // read_rtp_packet() returns the complete RTP packet carried by the RTSP
    // interleaved frame; RFC 6184 depacketization must see only its payload.
    int payload_len = strip_rtp_header(buf, pl);
    if (payload_len < 0) return 0;
    int header_len = pl - payload_len;

    if (codec_ == "H265")
        depacketize_h265(buf + header_len, payload_len);
    else
        depacketize_h264(buf + header_len, payload_len);
    return drain_outbuf(buf, max_len);
}

// RTCP RR (接收者报告): 摄像头要求定期报告, 否则踢连接。
// 通过 interleaved channel 1 (视频 RTCP) 发送, 每 2s 一次。
void H264Source::maybe_send_rtcp() {
    if (sock_ < 0 || session_.empty() || !got_rtp_) return;
    double now = now_seconds();
    if (now - last_rtcp_send_ < 2.0) return;
    last_rtcp_send_ = now;

    uint8_t rtcp[32];
    memset(rtcp, 0, sizeof(rtcp));
    rtcp[0] = 0x81;   // V=2, RC=1
    rtcp[1] = 201;    // PT=RR
    rtcp[2] = 0;      // length = 7 words
    rtcp[3] = 7;
    rtcp[4]  = (uint8_t)(ssrc_local_ >> 24);
    rtcp[5]  = (uint8_t)(ssrc_local_ >> 16);
    rtcp[6]  = (uint8_t)(ssrc_local_ >> 8);
    rtcp[7]  = (uint8_t)ssrc_local_;
    rtcp[8]  = (uint8_t)(ssrc_peer_ >> 24);
    rtcp[9]  = (uint8_t)(ssrc_peer_ >> 16);
    rtcp[10] = (uint8_t)(ssrc_peer_ >> 8);
    rtcp[11] = (uint8_t)ssrc_peer_;
    // fraction lost / cumulative lost = 0 (12-15)
    rtcp[16] = (uint8_t)(ext_seq_ >> 24);
    rtcp[17] = (uint8_t)(ext_seq_ >> 16);
    rtcp[18] = (uint8_t)(ext_seq_ >> 8);
    rtcp[19] = (uint8_t)ext_seq_;
    // jitter / LSR / DLSR = 0 (20-31)

    uint8_t frame[4 + 32];
    frame[0] = '$';
    frame[1] = RTCP_CHANNEL;
    frame[2] = 0;
    frame[3] = sizeof(rtcp);
    memcpy(frame + 4, rtcp, sizeof(rtcp));
    tcp_send(frame, sizeof(frame));
}

// RTSP GET_PARAMETER 保活探测: 返回 true=会话活着 (200 OK)。
// 摄像头进入"惩罚模式"时会对连接发几秒数据后静默 (不 FIN);
// 此时不要立刻重连 (新连接只会再次被惩罚), 而是保活等待。
bool H264Source::keepalive() {
    if (sock_ < 0 || session_.empty()) return false;
    std::string base = "rtsp://" + host_;
    if (port_ != 554) base += ":" + std::to_string(port_);
    base += path_;
    std::string hdrs, body;
    int sc = rtsp_req("GET_PARAMETER", base, "Session: " + session_ + "\r\n",
                      hdrs, body);
    if (sc != 200) printf("[RTSP] GET_PARAMETER status=%d\n", sc);
    return sc == 200;
}

bool H264Source::replay() {
    if (sock_ < 0 || session_.empty()) return false;
    std::string base = "rtsp://" + host_;
    if (port_ != 554) base += ":" + std::to_string(port_);
    base += path_;
    std::string hdrs, body;
    int sc = rtsp_req("PLAY", base,
                      "Session: " + session_ + "\r\nRange: npt=0.000-\r\n",
                      hdrs, body);
    if (sc != 200) printf("[RTSP] re-PLAY status=%d\n", sc);
    return sc == 200;
}

int H264Source::read_audio(uint8_t* buf, int max_len) {
    return drain_audio_outbuf(buf, max_len);
}

bool H264Source::reopen() {
    bool want_audio = want_audio_;
    close();
    return open("rtsp://" + (user_.empty() ? "" : user_ + ":" + pass_ + "@") +
                host_ + (port_ == 554 ? "" : ":" + std::to_string(port_)) + path_,
                want_audio);
}

void H264Source::close() {
    // 发送 TEARDOWN (best-effort)
    if (sock_ >= 0 && !session_.empty()) {
        std::string base = "rtsp://" + host_;
        if (port_ != 554) base += ":" + std::to_string(port_);
        base += path_;
        std::string hdrs, body;
        rtsp_req("TEARDOWN", base, "Session: " + session_ + "\r\n", hdrs, body);
        session_.clear();
    }
    tcp_close();
    if (pipe_fd_ >= 0) { ::close(pipe_fd_); pipe_fd_ = -1; }
    stdin_mode_ = false;
    out_buf_.clear();
    out_off_ = 0;
    audio_out_buf_.clear();
    audio_out_off_ = 0;
    audio_ = SdpAudio();
    frag_ = RtpFrag{};
    got_rtp_ = false;
    ssrc_peer_ = 0;
    seq_last_ = 0;
    seq_cycles_ = 0;
    ext_seq_ = 0;
    last_rtcp_send_ = -1e9;
    pkt_dbg_n_ = 0;
}

H264Source::~H264Source() { close(); }

} // namespace dw
