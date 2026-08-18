// 主机侧单元测试: audio_util (SDP 音频解析 + ADTS) + AudioRing (纯逻辑, 无板端依赖)
// 编译运行:
//   g++ -std=c++11 -Wall -Isrc tests/audio_test.cpp src/audio_util.cpp src/audio_ring.cpp -o /tmp/audio_test && /tmp/audio_test
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include "audio_ring.h"
#include "audio_util.h"

using namespace dw;

// ---- SDP 解析 -------------------------------------------------------------
static void test_parse_pcma() {
    const char* sdp =
        "v=0\r\n"
        "o=- 1 1 IN IP4 192.168.1.64\r\n"
        "s=Streaming\r\n"
        "m=video 0 RTP/AVP 96\r\n"
        "a=rtpmap:96 H264/90000\r\n"
        "a=control:track1\r\n"
        "m=audio 0 RTP/AVP 8\r\n"
        "a=rtpmap:8 PCMA/8000/1\r\n"
        "a=control:track2\r\n";
    SdpAudio a = parse_audio_track(sdp);
    assert(a.ok);
    assert(a.codec == "PCMA");
    assert(a.rate == 8000);
    assert(a.channels == 1);
    assert(a.control == "track2");
}

static void test_parse_pcmu() {
    const char* sdp =
        "m=video 0 RTP/AVP 96\r\n"
        "a=rtpmap:96 H264/90000\r\n"
        "a=control:track1\r\n"
        "m=audio 0 RTP/AVP 0\r\n"
        "a=rtpmap:0 PCMU/8000\r\n"
        "a=control:track2\r\n";
    SdpAudio a = parse_audio_track(sdp);
    assert(a.ok);
    assert(a.codec == "PCMU");
    assert(a.rate == 8000);
    assert(a.channels == 1);  // 缺声道默认 1
}

static void test_parse_aac() {
    const char* sdp =
        "m=video 0 RTP/AVP 96\r\n"
        "a=rtpmap:96 H265/90000\r\n"
        "a=control:track1\r\n"
        "m=audio 0 RTP/AVP 97\r\n"
        "a=rtpmap:97 MPEG4-GENERIC/16000/2\r\n"
        "a=fmtp:97 streamtype=5; profile-level-id=1; mode=AAC-hbr; "
        "sizelength=13; indexlength=3; indexdeltalength=3; config=1210\r\n"
        "a=control:track2\r\n";
    SdpAudio a = parse_audio_track(sdp);
    assert(a.ok);
    assert(a.codec == "AAC");
    assert(a.rate == 16000);
    assert(a.channels == 2);
    assert(a.fmtp.find("config=1210") != std::string::npos);
    int ot, sf, ch;
    assert(parse_audio_specific_config(a.fmtp, ot, sf, ch));
    assert(ot == 2);   // AAC LC
    assert(sf == 4);   // 44100
    assert(ch == 2);   // stereo
}

static void test_parse_static_pt_fallback() {
    // 摄像头不给 rtpmap 时按 RFC 3551 静态 PT 推断 G711
    const char* sdp =
        "m=video 0 RTP/AVP 96\r\n"
        "a=rtpmap:96 H264/90000\r\n"
        "a=control:track1\r\n"
        "m=audio 0 RTP/AVP 8\r\n"
        "a=control:track2\r\n";
    SdpAudio a = parse_audio_track(sdp);
    assert(a.ok);
    assert(a.codec == "PCMA");
    assert(a.rate == 8000);
}

static void test_parse_no_audio() {
    const char* sdp =
        "m=video 0 RTP/AVP 96\r\n"
        "a=rtpmap:96 H264/90000\r\n"
        "a=control:track1\r\n";
    SdpAudio a = parse_audio_track(sdp);
    assert(!a.ok);
}

static void test_parse_unsupported_codec() {
    const char* sdp =
        "m=video 0 RTP/AVP 96\r\n"
        "a=rtpmap:96 H264/90000\r\n"
        "a=control:track1\r\n"
        "m=audio 0 RTP/AVP 12\r\n"
        "a=rtpmap:12 QCELP/8000\r\n"
        "a=control:track2\r\n";
    SdpAudio a = parse_audio_track(sdp);
    assert(!a.ok);  // 不支持的编码 → 不启用音频
}

// ---- ADTS -----------------------------------------------------------------
static void test_adts_header() {
    uint8_t h[7];
    build_adts_header(h, 1, 4, 2, 100);
    assert(h[0] == 0xFF);
    assert((h[1] & 0xF6) == 0xF0);  // syncword 低 4=1111, ID=0, layer=00
    assert(h[1] & 0x01);            // protection_absent=1
    // profile (h[2]>>6) = 1, sf_idx ((h[2]>>2)&0xF) = 4, chan cfg 高 1 = 0
    assert((h[2] >> 6) == 1);
    assert(((h[2] >> 2) & 0x0F) == 4);
    assert(((h[3] >> 6) & 0x03) == 2);  // chan cfg 低 2 bit
    // frame_length 13 bit = 100
    int fl = ((h[3] & 0x03) << 11) | (h[4] << 3) | (h[5] >> 5);
    assert(fl == 100);
}

// ---- AAC 解包 ---------------------------------------------------------------
static void test_append_pcma_passthrough() {
    SdpAudio a;
    a.ok = true;
    a.codec = "PCMA";
    std::vector<uint8_t> out;
    uint8_t pcm[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    size_t n = append_audio_payload(out, a, 2, 4, 2, pcm, sizeof(pcm));
    assert(n == 10);
    assert(out.size() == 10);
    assert(memcmp(out.data(), pcm, 10) == 0);
}

static void test_append_aac_single_au() {
    SdpAudio a;
    a.ok = true;
    a.codec = "AAC";
    // 1 AU: AU-headers-length=16, AU-size=200bit(25B), data 25 字节
    std::vector<uint8_t> pkt;
    pkt.push_back(0x00);
    pkt.push_back(0x10);
    pkt.push_back((200 >> 5) & 0xFF);
    pkt.push_back((200 & 0x1F) << 3);
    for (int i = 0; i < 25; ++i) pkt.push_back((uint8_t)(0x40 + i));

    std::vector<uint8_t> out;
    size_t n = append_audio_payload(out, a, 2, 4, 2, pkt.data(), pkt.size());
    assert(n == 7 + 25);
    assert(out.size() == 32);
    // ADTS syncword + frame_length 校验
    assert(out[0] == 0xFF && (out[1] & 0xF0) == 0xF0);
    int fl = ((out[3] & 0x03) << 11) | (out[4] << 3) | (out[5] >> 5);
    assert(fl == 32);
    for (int i = 0; i < 25; ++i) assert(out[7 + i] == (uint8_t)(0x40 + i));
}

static void test_append_aac_no_au_header() {
    // AU-headers-length=0: 整包视为单个 AU
    SdpAudio a;
    a.ok = true;
    a.codec = "AAC";
    std::vector<uint8_t> pkt;
    pkt.push_back(0x00);
    pkt.push_back(0x00);
    for (int i = 0; i < 20; ++i) pkt.push_back((uint8_t)(0x60 + i));

    std::vector<uint8_t> out;
    size_t n = append_audio_payload(out, a, 2, 8, 1, pkt.data(), pkt.size());
    assert(n == 7 + 20);
    assert(out[0] == 0xFF && (out[1] & 0xF0) == 0xF0);
    int fl = ((out[3] & 0x03) << 11) | (out[4] << 3) | (out[5] >> 5);
    assert(fl == 27);
}

static void test_append_aac_size_in_bytes() {
    // TP-LINK 摄像头实测: AU-size 字段直接填字节数 (非 RFC 的 bit)。
    // 构造: AU-headers-length=16(bit), AU-size=207(字节), 数据 207 字节。
    SdpAudio a;
    a.ok = true;
    a.codec = "AAC";
    std::vector<uint8_t> pkt;
    pkt.push_back(0x00);
    pkt.push_back(0x10);
    pkt.push_back((207 >> 5) & 0xFF);
    pkt.push_back((207 & 0x1F) << 3);
    for (int i = 0; i < 207; ++i) pkt.push_back((uint8_t)(i & 0xFF));

    std::vector<uint8_t> out;
    size_t n = append_audio_payload(out, a, 2, 8, 1, pkt.data(), pkt.size());
    assert(n == 7 + 207);
    assert(out[0] == 0xFF && (out[1] & 0xF0) == 0xF0);
    int fl = ((out[3] & 0x03) << 11) | (out[4] << 3) | (out[5] >> 5);
    assert(fl == 7 + 207);
    for (int i = 0; i < 207; ++i)
        assert(out[7 + i] == (uint8_t)(i & 0xFF));
}

static void test_append_aac_invalid() {
    SdpAudio a;
    a.ok = true;
    a.codec = "AAC";
    uint8_t pkt[3] = {0x00, 0x10, 0x00};
    std::vector<uint8_t> out;
    assert(append_audio_payload(out, a, 2, 4, 2, pkt, 3) == 0);
}

// ---- AudioRing --------------------------------------------------------------
static void test_ring_cut() {
    AudioRing ring(64 * 1024);
    std::vector<uint8_t> chunk(160, 0x55);
    // 10ms 间隔喂 160B (G711 8kHz 20ms 包近似)
    for (int i = 0; i < 500; ++i)
        ring.push(chunk.data(), chunk.size(), 1000.0 + i * 0.01);

    std::vector<uint8_t> out;
    int rc = ring.cut(1001.0, 1003.0, 2.5, 64 * 1024, out);
    assert(rc == 0);
    assert(!out.empty());
    assert(out.size() == 201 * 160);  // [1001.00, 1003.00] 含两端

    // 窗口超出 → 1
    assert(ring.cut(900.0, 901.0, 2.5, 64 * 1024, out) == 1);
    assert(ring.cut(2000.0, 2001.0, 2.5, 64 * 1024, out) == 1);

    // 断流: 中间空了 1 秒 → 2
    AudioRing ring2(64 * 1024);
    for (int i = 0; i < 100; ++i)
        ring2.push(chunk.data(), chunk.size(), 1000.0 + i * 0.01);
    for (int i = 0; i < 100; ++i)
        ring2.push(chunk.data(), chunk.size(), 1002.0 + i * 0.01);
    assert(ring2.cut(1000.5, 1002.5, 0.5, 64 * 1024, out) == 2);
}

static void test_ring_wrap() {
    AudioRing ring(1024);  // 小环, 强制环绕
    std::vector<uint8_t> chunk(100, 0x77);
    for (int i = 0; i < 50; ++i)
        ring.push(chunk.data(), chunk.size(), 1000.0 + i * 0.1);
    // 环绕后仍能切出最新窗口
    std::vector<uint8_t> out;
    int rc = ring.cut(1004.0, 1004.9, 2.5, 4096, out);
    assert(rc == 0);
    assert(!out.empty());
    for (size_t i = 0; i < out.size(); ++i) assert(out[i] == 0x77);
}

int main() {
    test_parse_pcma();
    test_parse_pcmu();
    test_parse_aac();
    test_parse_static_pt_fallback();
    test_parse_no_audio();
    test_parse_unsupported_codec();
    test_adts_header();
    test_append_pcma_passthrough();
    test_append_aac_single_au();
    test_append_aac_no_au_header();
    test_append_aac_size_in_bytes();
    test_append_aac_invalid();
    test_ring_cut();
    test_ring_wrap();
    printf("ALL AUDIO TESTS PASSED\n");
    return 0;
}
