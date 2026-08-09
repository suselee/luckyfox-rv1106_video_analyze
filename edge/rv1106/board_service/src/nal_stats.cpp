#include "nal_stats.h"

#include <algorithm>
#include <stdio.h>

namespace dw {

// ---- NalScanner -------------------------------------------------------------
void NalScanner::feed(const uint8_t* data, size_t len, double ts) {
    size_t i = 0;
    while (i < len) {
        if (i + 3 <= len && data[i] == 0 && data[i + 1] == 0 &&
            (data[i + 2] == 1 ||
             (i + 3 < len && data[i + 2] == 0 && data[i + 3] == 1))) {
            emit(ts);
            size_t code_len =
                (i + 3 < len && data[i + 2] == 0 && data[i + 3] == 1) ? 4 : 3;
            i += code_len;
            continue;
        }
        cur_.push_back(data[i++]);
        if (cur_.size() > MAX_NAL_BYTES) {
            printf("[NAL] WARN oversized NAL (%.2f MB), dropped\n",
                   cur_.size() / 1048576.0);
            cur_.clear();
            cur_.shrink_to_fit();
        }
    }
}

void NalScanner::flush(double ts) { emit(ts); }

void NalScanner::emit(double ts) {
    if (cur_.empty()) return;
    if (cb_) cb_(cur_.data(), cur_.size(), ts);
    cur_.clear();
    cur_.shrink_to_fit();
}

// ---- NalStats ----------------------------------------------------------------
void NalStats::on_nal(const uint8_t* nal, size_t len, double ts) {
    nals++;
    bytes += (unsigned long long)len;

    if (last_data_ts >= 0 && ts - last_data_ts > 3.0) gaps++;
    last_data_ts = ts;

    uint8_t type = 0;
    if (*codec == "H265") {
        if (len >= 2) type = (uint8_t)((nal[0] >> 1) & 0x3F);
    } else if (len >= 1) {
        type = (uint8_t)(nal[0] & 0x1F);
    }
    NalKind kind = classify_nal(*codec, type);

    if (kind.key) {
        if (last_key_ts >= 0) gop_secs.push_back(ts - last_key_ts);
        last_key_ts = ts;
        key_count++;
    }
    if (kind.vcl) vcl_count++;

    if (*codec == "H264") {
        if (type == 7) { sps_count++; sps_bytes += len; }
        else if (type == 8) { pps_count++; pps_bytes += len; }
        else if (type == 6) { sei_count++; }
    } else if (*codec == "H265") {
        if (type == 32) { vps_count++; vps_bytes += len; }
        else if (type == 33) { sps_count++; sps_bytes += len; }
        else if (type == 34) { pps_count++; pps_bytes += len; }
        else if (type == 39 || type == 40) { sei_count++; }
    }
}

void NalStats::print_summary(double total_secs) const {
    double safe = total_secs > 0.001 ? total_secs : 1.0;
    printf("===== PROBE SUMMARY =====\n");
    printf("codec          = %s\n", codec ? codec->c_str() : "(null)");
    printf("duration       = %.1fs\n", total_secs);
    printf("bytes_total    = %llu (%.1f MB)\n", bytes, bytes / 1048576.0);
    printf("avg_bitrate    = %.2f Mbps\n", bytes * 8.0 / safe / 1e6);
    if (!window_kbps.empty()) {
        double wmin = *std::min_element(window_kbps.begin(), window_kbps.end());
        double wmax = *std::max_element(window_kbps.begin(), window_kbps.end());
        double wsum = 0;
        for (size_t i = 0; i < window_kbps.size(); ++i) wsum += window_kbps[i];
        printf("window         = min %.0f / avg %.0f / max %.0f kbps\n",
               wmin, wsum / window_kbps.size(), wmax);
    }
    printf("nals           = %llu (key %llu, vcl %llu)\n", nals, key_count, vcl_count);
    printf("fps_estimate   = %.2f\n", vcl_count / safe);
    if (!gop_secs.empty()) {
        double gmin = *std::min_element(gop_secs.begin(), gop_secs.end());
        double gmax = *std::max_element(gop_secs.begin(), gop_secs.end());
        double gsum = 0;
        for (size_t i = 0; i < gop_secs.size(); ++i) gsum += gop_secs[i];
        printf("gop_secs       = count %zu | min %.2f / avg %.2f / max %.2f\n",
               gop_secs.size(), gmin, gsum / gop_secs.size(), gmax);
    } else {
        printf("gop_secs       = (no keyframe pair observed)\n");
    }
    if (codec && *codec == "H264") {
        printf("parameter sets = SPS %llu (%llu B), PPS %llu (%llu B), SEI %llu\n",
               sps_count, sps_bytes, pps_count, pps_bytes, sei_count);
    } else {
        printf("parameter sets = VPS %llu (%llu B), SPS %llu (%llu B), "
               "PPS %llu (%llu B), SEI %llu\n",
               vps_count, vps_bytes, sps_count, sps_bytes, pps_count, pps_bytes, sei_count);
    }
    printf("stream_gaps    = %llu (>3s no data)\n", gaps);
    if (!window_kbps.empty() || total_secs > 0) {
        double mbps = bytes * 8.0 / safe / 1e6;
        double bps = bytes / safe;
        printf("estimate       = 45s clip ~= %.1f MB | 64MB ring ~= %.0fs of stream\n",
               bps * 45.0 / 1048576.0, 64.0 * 1048576.0 / (bps > 1.0 ? bps : 1.0));
        printf("(%s stream)\n", mbps < 4.5 ? "fits well within budget" :
               (mbps < 9.0 ? "moderate; consider ring <= 48MB" : "bitrate high; verify with camera OSD/config"));
    }
}

} // namespace dw