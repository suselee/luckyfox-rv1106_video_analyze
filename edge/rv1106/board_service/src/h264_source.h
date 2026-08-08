#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace dw {

// 原生 RTSP/RTP 客户端, 零外部依赖, 仅用 POSIX socket。
// TCP 直连摄像头 → RTSP 握手 (OPTIONS/DESCRIBE/SETUP/PLAY)
// → 接收 RTP interleaved H.264 → 解封装为 Annex-B → 输出给 MPP 硬解。
class H264Source {
public:
    ~H264Source();

    // rtsp_url 格式: rtsp://[user:pass@]host[:port]/path
    bool open(const std::string& rtsp_url);

    // 非阻塞读取 Annex-B H.264 数据。返回 >0=字节数, 0=暂无数据, <0=错误/断线。
    int read_chunk(uint8_t* buf, int max_len);

    void close();
    bool reopen();
    bool is_open() const { return sock_ >= 0; }

private:
    static const int RTP_CHANNEL  = 0;
    static const int RTCP_CHANNEL = 1;

    struct RtpFrag {
        std::vector<uint8_t> buf;
        bool    active = false;
        uint8_t nal_type = 0;
    };

    // ---------- URL 解析 ----------
    void parse_url(const std::string& url);

    // ---------- TCP ----------
    bool tcp_connect();
    bool tcp_send(const void* buf, int len);
    int  tcp_recv(uint8_t* buf, int len, int timeout_ms);
    void tcp_close();

    // ---------- RTSP ----------
    // 通用 RTSP 请求, 返回 status_code (如 200), 失败返回 -1。
    // out_headers 和 out_body 填充响应内容。
    int rtsp_req(const std::string& method, const std::string& url,
                 const std::string& extra_hdr,
                 std::string& out_headers, std::string& out_body);
    bool rtsp_handshake();

    // ---------- RTP/H.264 ----------
    int  read_rtp_packet(uint8_t* buf, int max_len);
    void depacketize_h264(const uint8_t* rtp_payload, int len);

    // ---------- Annex-B 输出缓冲 ----------
    int  drain_outbuf(uint8_t* buf, int max_len);
    void append_nal(const uint8_t* data, int len);

    // ---------- Auth ----------
    std::string basic_auth() const;
    static std::string base64_encode(const uint8_t* data, int len);

    // ---------- 字段 ----------
    int         sock_ = -1;
    std::string host_, path_, user_, pass_;
    int         port_ = 554;
    std::string session_;
    int         cseq_    = 0;
    int         timeout_ = 3000;  // 读超时(ms)

    RtpFrag     frag_;
    std::vector<uint8_t> out_buf_;
    int         out_off_ = 0;
};

} // namespace dw
