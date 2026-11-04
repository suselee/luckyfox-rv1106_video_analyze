// 人体检测候选模型探针 (RV1106, 零拷贝 IO)
// 用途: 不改生产管线, 用真实片段帧离线评估候选检测模型 (yolov5n/yolov8n rknn)
//       的检出率与单帧耗时。
//
// 用法:
//   ./person_probe <model.rknn> <img1.jpg> [img2.jpg ...]
//
// 说明: RV1106 的 rknn 运行时要求 rknn_set_io_mem (零拷贝), 不支持
//       rknn_inputs_set 路径; 输出可能是 NC1HWC2 (量化), 这里统一转 NCHW float。

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <algorithm>

#include "rknn_api.h"
#include "stb/stb_image.h"

struct Box { float x1, y1, x2, y2, score; };

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static float iou(const Box& a, const Box& b) {
    float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, x2 - x1), h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    float aa = (a.x2 - a.x1) * (a.y2 - a.y1);
    float bb = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / std::max(1e-6f, aa + bb - inter);
}

static void nms(std::vector<Box>& boxes, float thresh) {
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) { return a.score > b.score; });
    std::vector<Box> keep;
    std::vector<bool> dead(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); i++) {
        if (dead[i]) continue;
        keep.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); j++)
            if (!dead[j] && iou(boxes[i], boxes[j]) > thresh) dead[j] = true;
    }
    boxes.swap(keep);
}

// NC1HWC2 int8 -> NCHW float
static void nc1hwc2_to_nchw(const int8_t* src, float* dst, int* dims,
                            int channel, int h, int w, int zp, float scale) {
    int batch = dims[0], C1 = dims[1], C2 = dims[4];
    int hw_src = dims[2] * dims[3], hw_dst = h * w;
    for (int i = 0; i < batch; i++) {
        const int8_t* src_b = src + i * C1 * hw_src * C2;
        float* dst_b = dst + i * channel * hw_dst;
        for (int c = 0; c < channel; ++c) {
            int plane = c / C2;
            const int8_t* src_c = plane * hw_src * C2 + src_b;
            int offset = c % C2;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    dst_b[c * hw_dst + y * w + x] =
                        (src_c[C2 * (y * w + x) + offset] - zp) * scale;
        }
    }
}

static float* dequant(rknn_tensor_attr& native_attr, rknn_tensor_attr& orig_attr,
                      void* mem) {
    int n = orig_attr.n_elems;
    float* out = (float*)malloc(n * sizeof(float));
    if (!out) return NULL;
    int zp = native_attr.zp;
    float scale = native_attr.scale;
    if (native_attr.fmt == RKNN_TENSOR_UNDEFINED) { zp = orig_attr.zp; scale = orig_attr.scale; }
    if (native_attr.fmt == RKNN_TENSOR_NC1HWC2 && native_attr.type == RKNN_TENSOR_INT8) {
        int channel = orig_attr.dims[1];
        int h = orig_attr.n_dims > 2 ? orig_attr.dims[2] : 1;
        int w = orig_attr.n_dims > 3 ? orig_attr.dims[3] : 1;
        nc1hwc2_to_nchw((int8_t*)mem, out, (int*)native_attr.dims, channel, h, w,
                        native_attr.zp, native_attr.scale);
    } else if (native_attr.type == RKNN_TENSOR_INT8) {
        int8_t* s = (int8_t*)mem;
        for (int i = 0; i < n; i++) out[i] = (s[i] - zp) * scale;
    } else if (native_attr.type == RKNN_TENSOR_UINT8) {
        uint8_t* s = (uint8_t*)mem;
        for (int i = 0; i < n; i++) out[i] = (s[i] - zp) * scale;
    } else if (native_attr.type == RKNN_TENSOR_FLOAT32) {
        memcpy(out, mem, n * sizeof(float));
    } else {
        memset(out, 0, n * sizeof(float));
    }
    return out;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <model.rknn> <img1.jpg> [img2 ...]\n", argv[0]);
        return 1;
    }
    FILE* fp = fopen(argv[1], "rb");
    if (!fp) { printf("PROBE_FAIL open model\n"); return 2; }
    fseek(fp, 0, SEEK_END); long msize = ftell(fp); fseek(fp, 0, SEEK_SET);
    std::vector<unsigned char> model(msize);
    if (fread(model.data(), 1, msize, fp) != (size_t)msize) { fclose(fp); return 2; }
    fclose(fp);

    rknn_context ctx = 0;
    if (rknn_init(&ctx, model.data(), msize, 0, NULL) < 0) { printf("PROBE_FAIL init\n"); return 2; }

    rknn_input_output_num io;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));

    std::vector<rknn_tensor_attr> in_nat(1), out_nat(io.n_output), out_orig(io.n_output);
    memset(in_nat.data(), 0, sizeof(rknn_tensor_attr));
    in_nat[0].index = 0;
    if (rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &in_nat[0], sizeof(rknn_tensor_attr)) != RKNN_SUCC)
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_nat[0], sizeof(rknn_tensor_attr));
    printf("[MODEL] queried input: dims=[%d,%d,%d,%d] fmt=%d type=%d zp=%d scale=%f stride=%d size=%d\n",
           in_nat[0].dims[0], in_nat[0].dims[1], in_nat[0].dims[2], in_nat[0].dims[3],
           in_nat[0].fmt, in_nat[0].type, in_nat[0].zp, in_nat[0].scale,
           in_nat[0].w_stride, in_nat[0].size);
    in_nat[0].type = RKNN_TENSOR_INT8;
    in_nat[0].fmt = RKNN_TENSOR_NHWC;

    int iw = in_nat[0].dims[2], ih = in_nat[0].dims[1];
    printf("[MODEL] input %dx%d fmt=%d zp=%d scale=%f, outputs=%d\n",
           iw, ih, in_nat[0].fmt, in_nat[0].zp, in_nat[0].scale, io.n_output);

    size_t in_sz = in_nat[0].size_with_stride ? in_nat[0].size_with_stride : in_nat[0].size;
    if (in_sz == 0) in_sz = (size_t)iw * ih * 3;
    rknn_tensor_mem* in_mem = rknn_create_mem(ctx, in_sz);
    if (!in_mem || rknn_set_io_mem(ctx, in_mem, &in_nat[0]) < 0) {
        printf("PROBE_FAIL input io_mem\n");
        return 3;
    }

    std::vector<rknn_tensor_mem*> out_mem(io.n_output, NULL);
    for (uint32_t i = 0; i < io.n_output; i++) {
        memset(&out_nat[i], 0, sizeof(rknn_tensor_attr));
        memset(&out_orig[i], 0, sizeof(rknn_tensor_attr));
        out_nat[i].index = i;
        out_orig[i].index = i;
        if (rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &out_nat[i], sizeof(rknn_tensor_attr)) != RKNN_SUCC)
            rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_nat[i], sizeof(rknn_tensor_attr));
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_orig[i], sizeof(rknn_tensor_attr));
        size_t sz = out_nat[i].size_with_stride ? out_nat[i].size_with_stride : out_nat[i].size;
        if (sz == 0) sz = out_nat[i].n_elems;
        out_mem[i] = rknn_create_mem(ctx, sz);
        if (!out_mem[i] || rknn_set_io_mem(ctx, out_mem[i], &out_nat[i]) < 0) {
            printf("PROBE_FAIL output[%u] io_mem (sz=%zu)\n", i, sz);
            return 3;
        }
        printf("  out[%u] nat_dims=[", i);
        for (uint32_t d = 0; d < out_nat[i].n_dims; d++)
            printf("%d%s", out_nat[i].dims[d], d + 1 < out_nat[i].n_dims ? "," : "");
        printf("] fmt=%d type=%d | orig_dims=[", out_nat[i].fmt, out_nat[i].type);
        for (uint32_t d = 0; d < out_orig[i].n_dims; d++)
            printf("%d%s", out_orig[i].dims[d], d + 1 < out_orig[i].n_dims ? "," : "");
        printf("]\n");
    }

    std::vector<float> times;
    for (int k = 2; k < argc; k++) {
        int w = 0, h = 0, c = 0;
        unsigned char* rgb = stbi_load(argv[k], &w, &h, &c, 3);
        if (!rgb) { printf("SKIP %s\n", argv[k]); continue; }

        float r = std::min((float)iw / w, (float)ih / h);
        int nw = (int)(w * r), nh = (int)(h * r);
        int ox = (iw - nw) / 2, oy = (ih - nh) / 2;
        int stride = in_nat[0].w_stride ? in_nat[0].w_stride : iw;
        int8_t* dst = (int8_t*)in_mem->virt_addr;
        memset(dst, 114 - 128, (size_t)stride * ih * 3);
        for (int y = 0; y < nh; y++) {
            int sy = std::min(h - 1, (int)(y / r));
            for (int x = 0; x < nw; x++) {
                int sx = std::min(w - 1, (int)(x / r));
                const unsigned char* s = rgb + ((size_t)sy * w + sx) * 3;
                int8_t* d = dst + ((size_t)(oy + y) * stride + (ox + x)) * 3;
                d[0] = (int8_t)((int)s[0] - 128);
                d[1] = (int8_t)((int)s[1] - 128);
                d[2] = (int8_t)((int)s[2] - 128);
            }
        }

        // 零拷贝内存必须显式同步到设备, 否则 NPU 读到的是旧数据
        rknn_mem_sync(ctx, in_mem, RKNN_MEMORY_SYNC_TO_DEVICE);
        double t0 = now_ms();
        if (rknn_run(ctx, NULL) < 0) { printf("PROBE_FAIL run\n"); stbi_image_free(rgb); continue; }
        double dt = now_ms() - t0;
        times.push_back((float)dt);
        for (uint32_t oi = 0; oi < io.n_output; oi++)
            rknn_mem_sync(ctx, out_mem[oi], RKNN_MEMORY_SYNC_FROM_DEVICE);

        std::vector<Box> persons;
        if (io.n_output == 1) {
            float* p = dequant(out_nat[0], out_orig[0], out_mem[0]->virt_addr);
            if (p) {
                int nd = out_orig[0].n_dims;
                int n = 0, st = 0;
                if (nd == 3) {
                    if (out_orig[0].dims[2] >= 6 && out_orig[0].dims[2] <= 200) { n = out_orig[0].dims[1]; st = out_orig[0].dims[2]; }
                    else { n = out_orig[0].dims[2]; st = out_orig[0].dims[1]; }
                }
                // 自动判断布局: (1,N,85) 还是 (1,85,N)
                bool channel_major = false;
                {
                    int nn = out_orig[0].dims[1], cc = out_orig[0].dims[2];
                    if (cc < 200) {
                        float maxA = 0, maxB = 0;
                        for (int i = 0; i < nn; i += 7) maxA = std::max(maxA, p[(size_t)i * cc + 4]);
                        for (int i = 0; i < nn; i += 7) maxB = std::max(maxB, p[(size_t)4 * nn + i]);
                        channel_major = maxB > maxA;
                        n = nn; st = cc;
                    }
                }
                for (int i = 0; i < n; i++) {
                    float obj, cls;
                    if (channel_major) {
                        obj = p[(size_t)4 * n + i];
                        cls = p[(size_t)5 * n + i];
                        if (obj < 0.15f || cls < 0.15f) continue;
                        float cx = p[(size_t)0 * n + i], cy = p[(size_t)1 * n + i];
                        float bw = p[(size_t)2 * n + i], bh = p[(size_t)3 * n + i];
                        persons.push_back({cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, obj * cls});
                        continue;
                    }
                    if (st == 84) {
                        cls = 0.0f;
                        for (int cc = 0; cc < 80; cc++) cls = std::max(cls, p[(size_t)i * st + 4 + cc]);
                        obj = cls;
                    } else {
                        obj = p[(size_t)i * st + 4];
                        cls = p[(size_t)i * st + 5];
                    }
                    if (obj < 0.15f || cls < 0.15f) continue;
                    float cx = p[(size_t)i * st + 0], cy = p[(size_t)i * st + 1];
                    float bw = p[(size_t)i * st + 2], bh = p[(size_t)i * st + 3];
                    persons.push_back({cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, obj * cls});
                }
                free(p);
            }
        } else {
            // 原始三尺度输出 (1,3,H,W,85) 或 (1,255,H,W): 需 sigmoid + anchor 解码
            const int anchors[3][6] = {{10,13,16,30,33,23},{30,61,62,45,59,119},{116,90,156,198,373,326}};
            for (uint32_t o = 0; o < io.n_output && o < 3; o++) {
                float* p = dequant(out_nat[o], out_orig[o], out_mem[o]->virt_addr);
                if (!p) continue;
                int nd = out_orig[o].n_dims;
                int H = 0, W = 0, na = 3, C = 85;
                bool layout5d = (nd == 5);
                if (layout5d) { na = out_orig[o].dims[1]; H = out_orig[o].dims[2]; W = out_orig[o].dims[3]; C = out_orig[o].dims[4]; }
                else if (nd == 4) { C = out_orig[o].dims[1]; H = out_orig[o].dims[2]; W = out_orig[o].dims[3]; na = C / 85; }
                else { free(p); continue; }
                int stride = iw / W;
                int ai = (stride == 8) ? 0 : (stride == 16 ? 1 : 2);
                auto sig = [](float v) { return 1.0f / (1.0f + expf(-v)); };
                for (int a = 0; a < na && a < 3; a++) {
                    for (int y = 0; y < H; y++) {
                        for (int x = 0; x < W; x++) {
                            size_t base = layout5d
                                ? (((size_t)a * H + y) * W + x) * C
                                : ((size_t)a * 85 * H * W + (size_t)y * W + x);
                            float obj, cls;
                            float bx, by, bw, bh;
                            if (layout5d) {
                                obj = sig(p[base + 4]); cls = sig(p[base + 5]);
                                if (obj < 0.15f || cls < 0.15f) continue;
                                bx = p[base + 0]; by = p[base + 1]; bw = p[base + 2]; bh = p[base + 3];
                            } else {
                                obj = sig(p[base + (size_t)4 * H * W]);
                                cls = sig(p[base + (size_t)5 * H * W]);
                                if (obj < 0.15f || cls < 0.15f) continue;
                                bx = p[base + 0]; by = p[base + 1];
                                bw = p[base + (size_t)2 * H * W]; bh = p[base + (size_t)3 * H * W];
                            }
                            float cx = (sig(bx) * 2.0f - 0.5f + x) * stride;
                            float cy = (sig(by) * 2.0f - 0.5f + y) * stride;
                            float ww = powf(sig(bw) * 2.0f, 2) * anchors[ai][a * 2];
                            float hh = powf(sig(bh) * 2.0f, 2) * anchors[ai][a * 2 + 1];
                            persons.push_back({cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2, obj * cls});
                        }
                    }
                }
                free(p);
            }
        }
        if (k == 2) {
            float mx = -1e9f, mn = 1e9f;
            float* q0 = dequant(out_nat[0], out_orig[0], out_mem[0]->virt_addr);
            int nn = out_orig[0].n_elems;
            for (int i = 0; i < nn; i++) { if (q0[i] > mx) mx = q0[i]; if (q0[i] < mn) mn = q0[i]; }
            printf("[DBG] out0 nat(zp=%d,sc=%.5f,fmt=%d) orig(zp=%d,sc=%.5f,fmt=%d) deq[%.2f,%.2f]\n",
                   out_nat[0].zp, out_nat[0].scale, out_nat[0].fmt,
                   out_orig[0].zp, out_orig[0].scale, out_orig[0].fmt, mn, mx);
            free(q0);
        }
        nms(persons, 0.45f);
        printf("%s person=%zu %.1fms\n", argv[k], persons.size(), dt);
        stbi_image_free(rgb);
    }
    if (!times.empty()) {
        std::sort(times.begin(), times.end());
        printf("[SUMMARY] n=%zu p50=%.1fms p95=%.1fms\n",
               times.size(), times[times.size() / 2], times[(size_t)(times.size() * 0.95)]);
    }
    for (uint32_t i = 0; i < io.n_output; i++) if (out_mem[i]) rknn_destroy_mem(ctx, out_mem[i]);
    rknn_destroy_mem(ctx, in_mem);
    rknn_destroy(ctx);
    return 0;
}
