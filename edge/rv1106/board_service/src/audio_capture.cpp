// audio_capture: 抓取 RTSP 音频输出 (G711 原始 PCM / AAC ADTS) 到文件,
// 可选同时保存视频基本流 (Annex-B), 供 NAS 端到端验证。
// 用法: ./audio_capture <rtsp_url> <out_file> [duration_seconds=30] [video_out]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

#include "h264_source.h"

using namespace dw;

static double now_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <rtsp_url> <out_file> [duration=30]\n", argv[0]);
        return 1;
    }
    double duration = argc >= 4 ? atof(argv[3]) : 30.0;

    H264Source src;
    if (!src.open(argv[1], true)) {
        printf("[ERR] RTSP open failed\n");
        return 1;
    }
    printf("[CAP] video=%s audio=%s/%dHz/%dch duration=%.0fs\n",
           src.codec().c_str(), src.audio().codec.c_str(), src.audio().rate,
           src.audio().channels, duration);

    FILE* f = fopen(argv[2], "wb");
    if (!f) { printf("[ERR] cannot open %s\n", argv[2]); return 1; }
    FILE* fv = NULL;
    if (argc >= 5 && argv[4][0]) fv = fopen(argv[4], "wb");

    std::vector<uint8_t> chunk(512 * 1024);
    std::vector<uint8_t> audio(64 * 1024);
    unsigned long long audio_bytes = 0;
    unsigned long long video_bytes = 0;
    double begin = now_seconds();

    while (now_seconds() - begin < duration) {
        int n = src.read_chunk(chunk.data(), (int)chunk.size());
        if (n < 0) {
            printf("[CAP] stream error; reopen\n");
            src.reopen();
            continue;
        }
        if (n > 0) {
            video_bytes += (unsigned long long)n;
            if (fv && fwrite(chunk.data(), 1, (size_t)n, fv) != (size_t)n) {
                printf("[CAP] video write error\n");
                break;
            }
        }
        while (true) {
            int an = src.read_audio(audio.data(), (int)audio.size());
            if (an <= 0) break;
            if (fwrite(audio.data(), 1, (size_t)an, f) != (size_t)an) {
                printf("[CAP] write error\n");
                break;
            }
            audio_bytes += (unsigned long long)an;
        }
    }
    fclose(f);
    if (fv) fclose(fv);
    printf("[CAP] done: audio=%llu bytes video=%llu bytes\n", audio_bytes, video_bytes);
    return 0;
}
