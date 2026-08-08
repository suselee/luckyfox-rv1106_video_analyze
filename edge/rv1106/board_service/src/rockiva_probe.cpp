#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>

#include "config.h"
#include "h264_source.h"
#include "mpp_decoder.h"
#include "rockiva_detector.h"
#include "system_monitor.h"

using namespace dw;

static double now_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

static int h264_nal_type(const uint8_t* data, int len) {
    int off = 0;
    if (len >= 4 && data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1) off = 4;
    else if (len >= 3 && data[0] == 0 && data[1] == 0 && data[2] == 1) off = 3;
    return off < len ? data[off] & 0x1f : -1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <config.ini> [frames]\n", argv[0]);
        return 1;
    }
    Config config;
    if (!config.load(argv[1])) return 2;
    int width = config.get_int("rtsp.width", 640);
    int height = config.get_int("rtsp.height", 360);
    int wanted_frames = argc > 2 ? atoi(argv[2]) : 30;
    if (wanted_frames < 1) wanted_frames = 1;

    RockIvaDetector detector;
    if (!detector.init(config.get("model.rockiva_dir", "/root/daughter_watch/models/rockiva"),
                       width, height,
                       config.get_int("rockiva.person_score", 45),
                       config.get_int("rockiva.face_score", 45))) {
        printf("PROBE_FAIL init_error=%d\n", detector.last_error());
        return 3;
    }
    H264Source source;
    MppDecoder decoder(width, height);
    if (!source.open(config.get("rtsp.url")) || !decoder.init()) {
        printf("PROBE_FAIL rtsp_or_decoder\n");
        return 4;
    }

    std::vector<uint8_t> chunk(256 * 1024);
    std::vector<double> timings;
    uint64_t pts = 0;
    uint32_t frame_id = 0;
    int analyzed = 0;
    int objects = 0;
    double deadline = now_seconds() + 30.0;
    bool keyframes_only = config.get_bool("rtsp.keyframes_only", false);
    while (analyzed < wanted_frames && now_seconds() < deadline) {
        int n = source.read_chunk(chunk.data(), (int)chunk.size());
        if (n < 0) break;
        if (n > 0) {
            int type = h264_nal_type(chunk.data(), n);
            bool feed = !keyframes_only || type == 5 || type == 6 || type == 7 || type == 8 || type == 9;
            if (feed) decoder.send(chunk.data(), n, pts++, true);
        }
        while (analyzed < wanted_frames) {
            Nv12Frame frame;
            if (!decoder.get_frame(frame, 0)) break;
            if (!frame.yuv420_layout() || (frame.data_fd() < 0 && !frame.physical_addr())) {
                printf("PROBE_FAIL no_zero_copy_yuv420 layout=%d fd=%d phys=%lu\n",
                       frame.yuv420_layout(), frame.data_fd(),
                       (unsigned long)frame.physical_addr());
                return 5;
            }
            IvaResult result;
            double start = now_seconds();
            bool ok = frame.data_fd() >= 0
                ? detector.detect_fd(++frame_id, frame.data_fd(), frame.yuv420_layout(),
                                     frame.width(), frame.height(), result)
                : detector.detect_physical(++frame_id, frame.physical_addr(),
                                           frame.yuv420_layout(),
                                           frame.width(), frame.height(), result);
            if (!ok) {
                printf("PROBE_FAIL frame=%d error=%d\n", analyzed, detector.last_error());
                return 6;
            }
            timings.push_back((now_seconds() - start) * 1000.0);
            objects += (int)result.people.size();
            analyzed++;
        }
    }
    source.close();
    decoder.deinit();
    if (analyzed < wanted_frames) {
        printf("PROBE_FAIL only_frames=%d\n", analyzed);
        return 7;
    }
    SystemMonitor monitor;
    SystemStats stats = monitor.sample();
    double p95 = SystemMonitor::percentile95(timings);
    printf("PROBE_OK frames=%d person_results=%d p95_ms=%.1f memory_mb=%.1f temperature_c=%.1f\n",
           analyzed, objects, p95, stats.available_memory_kb / 1024.0, stats.temperature_c);
    return p95 <= config.get_double("guard.max_detector_p95_ms", 150.0) ? 0 : 8;
}
