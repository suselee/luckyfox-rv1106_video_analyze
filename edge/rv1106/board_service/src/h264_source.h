#pragma once
#include <string>
#include <vector>
#include <cstdint>

#include "audio_util.h"
#include "time_util.h"

namespace dw {

// 原生 RTSP/RTP 客户端, 零外部依赖, 仅用 POSIX socket。
// TCP 直连摄像头 → RTSP 握手 (OPTIONS/DESCRIBE/SETUP/PLAY)
// → 接收 RTP interleaved H.264/H.265 → 解封装为 Annex-B → 输出。
// 可选音频: with_audio=true 时额外 SETUP 音频轨 (interleaved=2-3),
// 把 PCMU/PCMA 透传为原始 PCM, 或把 MPEG4-GENERIC (AAC) 封装为 ADTS。
// 类名保持 H264Source 兼容旧调用方; 实际码流由 SDP 决定 (codec() 可查)。
class H264Source {
public:
    ~H264Source();

    // rtsp_url 格式: rtsp://[user:pass@]host[:port]/path
    // with_audio=true 时尝试协商音频轨; 摄像头无音频时静默降级 (has_audio()==false)。
    bool open(const std::string& rtsp_url, bool with_audio = false);

    // SDP 协商出的码流: "H264" 或 "H265"
    const std::string& codec() const { return codec_; }

    // 音频轨协商结果。
    bool has_audio() const { return audio_.ok; }
    const SdpAudio& audio() const { return audio_; }

    // 非阻塞读取 Annex-B H.264 数据。返回 >0=字节数, 0=暂无数据, <0=错误/断线。
    int read_chunk(uint8_t* buf, int max_len);

    // 非阻塞读取音频输出 (PCMU/PCMA 为原始 PCM, AAC 为 ADTS)。
    // 返回 >0=字节数, 0=暂无数据, <0=错误/断线。
    int read_audio(uint8_t* buf, int max_len);

    // RTSP GET_PARAMETER 保活: 会话静默时探测摄像头是否还活着。
    bool keepalive();
    // 重新 PLAY: 摄像头静默后重发 PLAY 请求, 尝试重新激活推流。
    bool replay();

    void close();
    bool reopen();
    bool is_open() const { return sock_ >= 0; }

    // 绑定源 IP (可选): 摄像头固件对"历史高频重连的客户端"有持久配额
    // (每会话只推几 KB 数据); 换一个源 IP 让它认为是新客户端。
    void set_bind_ip(const std::string& ip) { bind_ip_ = ip; }

    // stdin 模式: 不直连摄像头, 从 fd 0 (管道) 读 Annex-B 码流。
    // 由外部 ffprobe 拉流输出 (摄像头对 ffprobe 无配额限制)。
    // 传入 codec 名 ("H264"/"H265") 供 depacketize 判断 (stdin 下不需要)。
    bool open_stdin();
    // FIFO 模式: 从命名管道读 Annex-B (供 high_stream 用 ffmpeg 拉 4K 流)。
    bool open_pipe(const std::string& path);

private:
    static const int RTP_CHANNEL  = 0;
    static const int RTCP_CHANNEL = 1;
    static const int RTP_AUDIO_CHANNEL  = 2;
    static const int RTCP_AUDIO_CHANNEL = 3;

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
    // 读一个 interleaved RTP 包, 返回 payload 长度, channel 经 ch 输出。
    // 0=超时, -1=错误, -2=EOF, -3=RTCP 包 (继续读)。
    int  read_rtp_packet(uint8_t* buf, int max_len, int& ch);
    // 剥 RTP 头, 返回 payload 长度 (失败返回 -1)。
    int  strip_rtp_header(const uint8_t* buf, int pl) const;
    void depacketize_h264(const uint8_t* rtp_payload, int len);
    void depacketize_h265(const uint8_t* rtp_payload, int len);
    void depacketize_audio(const uint8_t* rtp_payload, int len);
    // RTCP RR 保活: 摄像头固件要求客户端 ~5s 内发接收者报告,
    // 否则主动 FIN 断开 (ffprobe 发 RR 不被踢, 实测本客户端不发被 7.5s 踢)。
    void maybe_send_rtcp();

    // ---------- Annex-B 输出缓冲 ----------
    int  drain_outbuf(uint8_t* buf, int max_len);
    void append_nal(const uint8_t* data, int len);

    // ---------- 音频输出缓冲 ----------
    int  drain_audio_outbuf(uint8_t* buf, int max_len);

    // ---------- Auth ----------
    std::string basic_auth() const;
    static std::string base64_encode(const uint8_t* data, int len);

    // ---------- 字段 ----------
    int         sock_ = -1;
    bool        stdin_mode_ = false;   // 从 fd 0 读 Annex-B (ffprobe 管道)
    int         pipe_fd_ = -1;         // FIFO 模式读 fd
    std::string host_, path_, user_, pass_;
    std::string bind_ip_;              // 可选: 绑定的源地址
    std::string codec_;                // "H264" / "H265" (SDP 协商结果)
    int         port_ = 554;
    std::string session_;
    int         cseq_    = 0;
    int         timeout_ = 3000;  // 读超时(ms)

    // 音频
    bool    want_audio_ = false;       // open(with_audio=true) 时置位
    SdpAudio audio_;                   // 协商出的音频参数
    int      audio_object_type_ = 2;   // AAC: ASC object type (默认 AAC LC)
    int      audio_sf_index_ = 0;      // AAC: ASC sampling_frequency_index

    // RTCP RR 保活状态
    bool     got_rtp_ = false;
    uint32_t ssrc_peer_ = 0;
    uint32_t ssrc_local_ = 0x7A41AA11;
    uint16_t seq_last_ = 0;
    uint32_t seq_cycles_ = 0;
    uint32_t ext_seq_ = 0;
    double   last_rtcp_send_ = -1e9;
    int      pkt_dbg_n_ = 0;
    int      audio_chan_cfg_ = 0;      // AAC: ASC channel_configuration

    RtpFrag     frag_;
    std::vector<uint8_t> out_buf_;
    int         out_off_ = 0;
    std::vector<uint8_t> audio_out_buf_;
    int         audio_out_off_ = 0;
};

} // namespace dw
