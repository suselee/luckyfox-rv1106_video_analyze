// high_stream_probe: 板端 4K 主码流参数实测工具 (P0)。
// 用法: ./high_stream_probe <rtsp_url> [duration_seconds=60] [window_seconds=5]
// 作用: 连接摄像头主码流 (H264/H265), 逐 NAL 解析, 统计码流编码、
//   平均/窗口码率、GOP 分布、参数集频率、估计帧率、断流停顿。
// 结果决定: 板端环形缓冲秒数、切片对齐粒度、上传耗时预估。
// 仅依赖 POSIX socket, 不依赖 MPP/RKNN。

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <string>
#include <vector>

#include "h264_source.h"
#include "nal_stats.h"

using namespace dw;

// (本地 now_seconds 已删除: 与 time_util.h 的 dw::now_seconds() 同源同义,
// 两者并存 + using namespace 会导致 overloaded ambiguous 编译失败。)

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <rtsp_url> [duration_seconds=60] [window_seconds=5]\n",
               argv[0]);
        return 1;
    }
    const char* url = argv[1];
    double duration = argc >= 3 ? atof(argv[2]) : 60.0;
    double window_secs = argc >= 4 ? atof(argv[3]) : 5.0;
    if (duration <= 0 || window_secs <= 0) {
        printf("[ERR] duration/window must be > 0\n");
        return 1;
    }

    H264Source src;
    if (!src.open(url, true)) {
        printf("[ERR] RTSP open failed\n");
        return 1;
    }
    const std::string& codec = src.codec();
    printf("[PROBE] codec=%s duration=%.0fs window=%.0fs\n",
           codec.c_str(), duration, window_secs);
    if (src.has_audio()) {
        printf("[PROBE] audio=%s rate=%d channels=%d (fmtp=%s)\n",
               src.audio().codec.c_str(), src.audio().rate,
               src.audio().channels, src.audio().fmtp.c_str());
    } else {
        printf("[PROBE] audio=none\n");
    }

    NalStats stats;
    stats.codec = &codec;
    stats.window_secs = window_secs;

    NalScanner scanner;
    scanner.set_callback([&](const uint8_t* nal, size_t len, double ts) {
        stats.on_nal(nal, len, ts);
    });

    std::vector<uint8_t> chunk(512 * 1024);
    std::vector<uint8_t> audio_chunk(64 * 1024);
    unsigned long long audio_bytes = 0;
    double begin = now_seconds();
    stats.window_start = begin;
    int reconnect_wait = 2;

    while (now_seconds() - begin < duration) {
        int n = src.read_chunk(chunk.data(), (int)chunk.size());
        double now = now_seconds();

        if (n < 0) {
            printf("[PROBE] read error; reconnect in %ds\n", reconnect_wait);
            sleep(reconnect_wait);
            reconnect_wait = std::min(30, reconnect_wait * 2);
            scanner.flush(now);
            if (!src.reopen()) continue;
            printf("[PROBE] reconnected\n");
            reconnect_wait = 2;
            continue;
        }
        if (n > 0) {
            reconnect_wait = 2;
            stats.window_bytes += (unsigned long long)n;
            scanner.feed(chunk.data(), (size_t)n, now);
        }
        // 音频排空 (防止源内缓冲无限增长), 统计码率。
        while (true) {
            int an = src.read_audio(audio_chunk.data(), (int)audio_chunk.size());
            if (an <= 0) break;
            audio_bytes += (unsigned long long)an;
        }
        if (now - stats.window_start >= window_secs) {
            double el = now - stats.window_start;
            double kbps = stats.window_bytes * 8.0 / el / 1e3;
            stats.window_kbps.push_back(kbps);
            printf("[WIN] %7.1fs  stream %9.1f kbps  avg %8.1f kbps  "
                   "key=%llu vcl=%llu\n",
                   now - begin, kbps,
                   stats.bytes * 8.0 / (now - begin) / 1e3,
                   stats.key_count, stats.vcl_count);
            stats.window_start = now;
            stats.window_bytes = 0;
        }
    }

    double end = now_seconds();
    scanner.flush(end);
    stats.print_summary(end - begin);
    if (audio_bytes > 0)
        printf("[PROBE] audio stream: %llu bytes ≈ %.1f kbps\n",
               audio_bytes, audio_bytes * 8.0 / (end - begin) / 1e3);
    if (stats.nals == 0) {
        printf("[PROBE] no NAL data received; stream may be unauthenticated or wrong channel\n");
        return 2;
    }
    return 0;
}