// 主机侧单元测试: VideoRing (纯逻辑环形缓冲 + cut 切片, 无板端依赖)
// 编译运行: g++ -std=c++11 -Wall -Isrc tests/ring_test.cpp src/video_ring.cpp -o /tmp/ring_test && /tmp/ring_test
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include "video_ring.h"

using namespace dw;

// 合成 15fps H264 流: 每帧 1 条 NAL (关键帧=IDR, 前带 SPS/PPS)。
// 与 high_stream 入 ring 的数据一致: NAL 体 (不含起始码), 4 字节起始码在切片时补。
static void push_nal(VideoRing& ring, uint8_t type, size_t payload, double ts) {
    std::vector<uint8_t> nal(1 + payload, 0x55);
    nal[0] = type;
    NalKind k = classify_nal("H264", type);
    ring.push(nal.data(), nal.size(), type, k, ts);
}

// 关键帧: SPS(7) + PPS(8) + IDR(5); 普通帧: slice(1)
static void feed_frame(VideoRing& ring, double ts, bool key) {
    if (key) {
        push_nal(ring, 7, 3, ts);
        push_nal(ring, 8, 3, ts);
        push_nal(ring, 5, 64, ts);
    } else {
        push_nal(ring, 1, 64, ts);
    }
}

// 从 T0 到 T1 按 15fps 喂帧, 每 2s 一个关键帧 (每 30 帧)。
// skip_a/skip_b: 断流区间 (秒), 模拟中间无数据的 GAP。
static void feed_ring(VideoRing& ring, double T0, double T1,
                      double skip_a = -1.0, double skip_b = -1.0) {
    const double fps = 15.0;
    for (int f = 0;; ++f) {
        double t = T0 + f / fps;
        if (t > T1 + 1e-9) break;
        if (skip_a > 0 && t >= skip_a && t < skip_b) continue;
        feed_frame(ring, t, (f % 30) == 0);
    }
}

// 校验整个片段的每一段都以 4 字节起始码开头。
static bool all_start_codes(const std::vector<uint8_t>& b) {
    size_t i = 0;
    while (i < b.size()) {
        if (i + 4 > b.size()) return false;  // 只有尾部残余, 缺起始码
        if (b[i] != 0 || b[i + 1] != 0 || b[i + 2] != 0 || b[i + 3] != 1)
            return false;
        i += 4;
        // 找下一个起始码; 若没有则这条是最后一条 NAL, 到缓冲区结尾即合法
        bool found_next = false;
        while (i + 4 <= b.size()) {
            if (b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 0 && b[i + 3] == 1) {
                found_next = true;
                break;
            }
            ++i;
        }
        if (!found_next) break;  // 结尾
    }
    return true;
}

// 片段首 NAL 的 type (按 H264: b0 & 0x1F)
static int first_nal_type(const std::vector<uint8_t>& out) {
    if (out.size() < 5 || out[0] != 0 || out[1] != 0 || out[2] != 0 || out[3] != 1)
        return -1;
    return out[4] & 0x1F;
}

// 数片段里 type==n 的 NAL 个数
static int count_nal_type(const std::vector<uint8_t>& out, uint8_t type) {
    int n = 0;
    size_t i = 0;
    while (i + 5 <= out.size()) {
        if (out[i] == 0 && out[i + 1] == 0 && out[i + 2] == 0 && out[i + 3] == 1 &&
            (out[i + 4] & 0x1F) == type)
            n++;
        ++i;
    }
    return n;
}

// 基本切片: 起点回退到关键帧并带出 SPS/PPS, 终点在下一关键帧前收尾。
static void test_basic_cut() {
    VideoRing ring(4 * 1024 * 1024);
    feed_ring(ring, 1000.0, 1200.0);
    assert(ring.nal_count() > 0);

    std::vector<uint8_t> out;
    CutResult rc = ring.cut(1105.0, 1205.0, 2.0, 3.0, 2.5,
                            2 * 1024 * 1024, out);
    assert(rc == CutResult::OK);
    assert(!out.empty());
    assert(all_start_codes(out));
    // 起点 = 1103 前最近关键帧 (1102), 其 SPS/PPS 必须被带回开头
    assert(first_nal_type(out) == 7);
    assert(count_nal_type(out, 7) >= 1);
    assert(count_nal_type(out, 8) >= 1);
    assert(count_nal_type(out, 5) >= 1);
}

// 断流: 窗口横跨 8 秒无数据区间 → GAP 拒绝
void test_gap() {
    VideoRing ring(4 * 1024 * 1024);
    feed_ring(ring, 1000.0, 1030.0, 1010.0, 1018.0);

    std::vector<uint8_t> out;
    CutResult cut = ring.cut(1009.0, 1019.0, 2.0, 2.0, 2.5,
                             1 * 1024 * 1024, out);
    assert(cut == CutResult::GAP);
    assert(out.empty());
}

// 超过 max_bytes → TOO_BIG
void test_too_big() {
    VideoRing ring(4 * 1024 * 1024);
    feed_ring(ring, 1000.0, 1006.0);

    std::vector<uint8_t> out;
    CutResult cut = ring.cut(1002.0, 1004.0, 1.0, 1.0, 2.5, 64, out);
    assert(cut == CutResult::TOO_BIG);
    assert(out.empty());
}

// 空缓冲 / 窗口完全超出 → EMPTY_WINDOW
void test_empty_window() {
    VideoRing empty(4 * 1024 * 1024);
    std::vector<uint8_t> out;
    assert(empty.cut(0.0, 10.0, 2.0, 2.0, 2.5, 1024 * 1024, out) ==
           CutResult::EMPTY_WINDOW);

    VideoRing ring(4 * 1024 * 1024);
    feed_ring(ring, 1000.0, 1004.0);
    assert(ring.cut(500.0, 501.0, 2.0, 2.0, 2.5, 1024 * 1024, out) ==
           CutResult::EMPTY_WINDOW);  // hi_ts 在数据之前
    assert(ring.cut(1100.0, 1101.0, 2.0, 2.0, 2.5, 1024 * 1024, out) ==
           CutResult::EMPTY_WINDOW);  // lo_ts 在数据之后
}

// 窗口跨越/包含关键帧: 中间 IDR(5) 应原样保留, 起点仍是窗口前最近关键帧
void test_cross_keyframe() {
    VideoRing ring(4 * 1024 * 1024);
    feed_ring(ring, 1000.0, 1008.0);

    std::vector<uint8_t> out;
    CutResult cut = ring.cut(1003.5, 1006.5, 1.0, 1.0, 2.5, 1024 * 1024, out);
    assert(cut == CutResult::OK);
    assert(all_start_codes(out));
    assert(first_nal_type(out) == 7);  // 回退到 1002 的 SPS
    assert(count_nal_type(out, 5) == 3);  // 窗口内 IDR: 1002 + 1004 + 1006
}

int main() {
    test_basic_cut();
    test_gap();
    test_too_big();
    test_empty_window();
    test_cross_keyframe();
    printf("ALL RING TESTS PASSED\n");
    return 0;
}