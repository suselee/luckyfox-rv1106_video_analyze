// 主机侧单元测试: NalScanner / NalStats (纯逻辑, 无板端依赖)
// 编译运行: g++ -std=c++11 -Wall -Isrc tests/nal_test.cpp src/nal_stats.cpp -o /tmp/nal_test && /tmp/nal_test
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include "nal_stats.h"

using namespace dw;

static const uint8_t SC4[4] = {0, 0, 0, 1};
static const uint8_t SC3[3] = {0, 0, 1};

// 用例 1+2 通用: 一条含 SPS/PPS/IDR/若干片段的 H264 流
// 注意: NalStats::on_nal 收到的是 scanner 剥掉起始码后的 NAL(头部在 nal[0]),
//       因此这里构造的是不含起始码的 NAL 体。
std::vector<uint8_t> build_h264_nal(int type, int size) {
    std::vector<uint8_t> nal;
    nal.push_back((uint8_t)type);  // H264: type == (b0 & 0x1F)
    nal.insert(nal.end(), size, 0x55);
    return nal;
}

std::vector<uint8_t> build_h265_nal(int type, int size) {
    std::vector<uint8_t> nal;
    uint8_t header = (uint8_t)((type << 1) | 1);  // H265: type == ((b0>>1) & 0x3F)
    nal.push_back(header);
    nal.push_back(0x01);  // layer id + tid
    nal.insert(nal.end(), size, 0x55);
    return nal;
}

void test_nal_scanner_cross_chunk() {
    std::vector<int> types;
    std::vector<int> sizes;
    NalScanner scanner;
    scanner.set_callback([&](const uint8_t* nal, size_t len, double) {
        types.push_back(nal[0] & 0x1F);
        sizes.push_back((int)len);
    });
    std::vector<uint8_t> stream;
    stream.insert(stream.end(), SC4, SC4 + 4);  // SPS (type 7, 1 data byte)
    stream.push_back(0x67);
    stream.insert(stream.end(), SC4, SC4 + 4);
    stream.push_back(0x68);                     // PPS (type 8)
    for (int i = 0; i < 30; ++i) {              // 30 slices
        std::vector<uint8_t> nal = build_h264_nal(i % 15 == 0 ? 5 : 1, 100);
        stream.insert(stream.end(), SC4, SC4 + 4);
        stream.insert(stream.end(), nal.begin(), nal.end());
    }
    // 每 199 字节切一刀, 制造跨块 NAL
    for (size_t pos = 0; pos < stream.size(); pos += 199)
        scanner.feed(stream.data() + pos, stream.size() - pos < 199 ? stream.size() - pos : 199, 1.0);
    scanner.flush(1.0);
    assert(types.size() == 32);
    assert(types[0] == 7 && types[1] == 8);
    for (int i = 2; i < 32; ++i) {
        assert(sizes[i] == 100 + 1);  // 起始码 4 + 头 1 + 载荷 100
    }
    // 覆盖 H265 头
}

void test_nal_stats_gop() {
    NalStats stats;
    std::string codec = "H264";
    stats.codec = &codec;
    std::vector<uint8_t> key_nal = build_h264_nal(5, 200);
    std::vector<uint8_t> slice_nal = build_h264_nal(1, 200);
    double ts = 0;
    for (int frame = 0; frame < 60; ++frame) {
        bool key = (frame % 30) == 0;
        stats.on_nal(key ? key_nal.data() : slice_nal.data(),
                     key ? key_nal.size() : slice_nal.size(), ts);
        ts += 1.0 / 15.0;
    }
    assert(stats.nals == 60);
    assert(stats.key_count == 2);
    assert(stats.gop_secs.size() == 1);
    assert(abs(stats.gop_secs[0] - 2.0) < 1e-6);  // 30f/15fps = 2s
}

void test_gaps() {
    NalStats stats;
    std::string codec = "H264";
    stats.codec = &codec;
    uint8_t slice = 0x41;
    stats.on_nal(&slice, 1, 0.0);
    stats.on_nal(&slice, 1, 9.0);
    assert(stats.gaps == 1);
}

void test_3byte_start_code() {
    NalScanner scanner;
    std::vector<int> types;
    scanner.set_callback([&](const uint8_t* nal, size_t, double) { types.push_back(nal[0] & 0x1F); });
    std::vector<uint8_t> stream;
    stream.insert(stream.end(), SC3, SC3 + 3);
    stream.push_back(0x67);  // SPS
    stream.insert(stream.end(), 10, 0x11);
    scanner.feed(stream.data(), stream.size(), 1.0);
    scanner.flush(1.0);
    assert(types.size() == 1 && types[0] == 7);
}

void test_no_keyframe() {
    NalStats stats;
    std::string codec = "H264";
    stats.codec = &codec;
    uint8_t slice = 0x41;
    stats.on_nal(&slice, 1, 0.0);
    assert(stats.key_count == 0);
}

int main() {
    test_nal_scanner_cross_chunk();
    test_nal_stats_gop();
    test_gaps();
    test_3byte_start_code();
    test_no_keyframe();
    printf("ALL NAL TESTS PASSED\n");
    return 0;
}