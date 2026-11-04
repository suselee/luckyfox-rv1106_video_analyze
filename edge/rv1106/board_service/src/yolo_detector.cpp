#include "yolo_detector.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <unistd.h>

namespace dw {

namespace {

struct Box { float x1, y1, x2, y2, score; };

const int kAnchors[3][6] = {
    {10, 13, 16, 30, 33, 23},
    {30, 61, 62, 45, 59, 119},
    {116, 90, 156, 198, 373, 326},
};

inline float sigmoid(float v) { return 1.0f / (1.0f + expf(-v)); }

inline float iou(const Box& a, const Box& b) {
    float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, x2 - x1), h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    float aa = (a.x2 - a.x1) * (a.y2 - a.y1);
    float bb = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / std::max(1e-6f, aa + bb - inter);
}

void nms(std::vector<Box>& boxes, float thresh) {
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        return a.score > b.score;
    });
    std::vector<Box> keep;
    std::vector<bool> dead(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (dead[i]) continue;
        keep.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j)
            if (!dead[j] && iou(boxes[i], boxes[j]) > thresh) dead[j] = true;
    }
    boxes.swap(keep);
}

// NC1HWC2 int8 -> NCHW float
void nc1hwc2_to_nchw(const int8_t* src, float* dst, const uint32_t* dims, int channel,
                     int h, int w, int zp, float scale) {
    int batch = (int)dims[0], C1 = (int)dims[1], C2 = (int)dims[4];
    int hw_src = (int)dims[2] * (int)dims[3], hw_dst = h * w;
    for (int i = 0; i < batch; ++i) {
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

float* dequantize(const rknn_tensor_attr& nat, const rknn_tensor_attr& orig, void* mem) {
    int n = orig.n_elems;
    float* out = static_cast<float*>(malloc(n * sizeof(float)));
    if (!out) return nullptr;
    int zp = nat.zp;
    float scale = nat.scale;
    if (nat.fmt == RKNN_TENSOR_UNDEFINED) { zp = orig.zp; scale = orig.scale; }
    if (nat.fmt == RKNN_TENSOR_NC1HWC2 && nat.type == RKNN_TENSOR_INT8) {
        int channel = orig.dims[1];
        int h = orig.n_dims > 2 ? orig.dims[2] : 1;
        int w = orig.n_dims > 3 ? orig.dims[3] : 1;
        nc1hwc2_to_nchw(static_cast<int8_t*>(mem), out, nat.dims,
                        channel, h, w, nat.zp, nat.scale);
    } else if (nat.type == RKNN_TENSOR_INT8) {
        int8_t* s = static_cast<int8_t*>(mem);
        for (int i = 0; i < n; ++i) out[i] = (s[i] - zp) * scale;
    } else if (nat.type == RKNN_TENSOR_UINT8) {
        uint8_t* s = static_cast<uint8_t*>(mem);
        for (int i = 0; i < n; ++i) out[i] = (s[i] - zp) * scale;
    } else if (nat.type == RKNN_TENSOR_FLOAT32) {
        memcpy(out, mem, n * sizeof(float));
    } else {
        memset(out, 0, n * sizeof(float));
    }
    return out;
}

} // namespace

bool YoloPersonDetector::init(const char* model_path, int input_size, float conf) {
    if (inited_) destroy();
    input_size_ = input_size > 0 ? input_size : 416;
    conf_ = conf;

    FILE* fp = fopen(model_path, "rb");
    if (!fp) { printf("[YOLO] model open failed: %s\n", model_path); return false; }
    fseek(fp, 0, SEEK_END);
    long msize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<unsigned char> model(msize);
    if (fread(model.data(), 1, msize, fp) != static_cast<size_t>(msize)) {
        fclose(fp);
        return false;
    }
    fclose(fp);

    if (rknn_init(&ctx_, model.data(), msize, 0, NULL) < 0) {
        printf("[YOLO] rknn_init failed\n");
        ctx_ = 0;
        return false;
    }
    rknn_input_output_num io;
    rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    n_output_ = io.n_output;

    memset(&in_attr_, 0, sizeof(in_attr_));
    in_attr_.index = 0;
    if (rknn_query(ctx_, RKNN_QUERY_NATIVE_INPUT_ATTR, &in_attr_, sizeof(in_attr_)) != RKNN_SUCC)
        rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &in_attr_, sizeof(in_attr_));
    in_attr_.type = RKNN_TENSOR_INT8;
    in_attr_.fmt = RKNN_TENSOR_NHWC;

    size_t in_sz = in_attr_.size_with_stride ? in_attr_.size_with_stride : in_attr_.size;
    if (in_sz == 0) in_sz = static_cast<size_t>(input_size_) * input_size_ * 3;
    in_mem_ = rknn_create_mem(ctx_, in_sz);
    if (!in_mem_ || rknn_set_io_mem(ctx_, in_mem_, &in_attr_) < 0) {
        printf("[YOLO] input io_mem failed\n");
        destroy();
        return false;
    }
    input_buf_.assign(in_sz, -128);

    out_nat_.resize(n_output_);
    out_orig_.resize(n_output_);
    out_mem_.assign(n_output_, nullptr);
    for (uint32_t i = 0; i < n_output_; ++i) {
        memset(&out_nat_[i], 0, sizeof(rknn_tensor_attr));
        memset(&out_orig_[i], 0, sizeof(rknn_tensor_attr));
        out_nat_[i].index = i;
        out_orig_[i].index = i;
        if (rknn_query(ctx_, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &out_nat_[i], sizeof(rknn_tensor_attr)) != RKNN_SUCC)
            rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &out_nat_[i], sizeof(rknn_tensor_attr));
        rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &out_orig_[i], sizeof(rknn_tensor_attr));
        size_t sz = out_nat_[i].size_with_stride ? out_nat_[i].size_with_stride
                                                 : out_nat_[i].size;
        if (sz == 0) sz = out_nat_[i].n_elems;
        out_mem_[i] = rknn_create_mem(ctx_, sz);
        if (!out_mem_[i] || rknn_set_io_mem(ctx_, out_mem_[i], &out_nat_[i]) < 0) {
            printf("[YOLO] output[%u] io_mem failed\n", i);
            destroy();
            return false;
        }
    }
    inited_ = true;
    printf("[YOLO] ready: %s input=%d conf=%.2f outputs=%u\n",
           model_path, input_size_, conf_, n_output_);
    return true;
}

void YoloPersonDetector::destroy() {
    if (ctx_) {
        for (size_t i = 0; i < out_mem_.size(); ++i)
            if (out_mem_[i]) rknn_destroy_mem(ctx_, out_mem_[i]);
        if (in_mem_) rknn_destroy_mem(ctx_, in_mem_);
        rknn_destroy(ctx_);
    }
    out_mem_.clear();
    out_nat_.clear();
    out_orig_.clear();
    in_mem_ = nullptr;
    ctx_ = 0;
    inited_ = false;
}

bool YoloPersonDetector::detect(const unsigned char* rgb, int width, int height,
                                IvaResult& result) {
    if (!inited_ || !rgb || width <= 0 || height <= 0) return false;

    const int S = input_size_;
    const int stride = in_attr_.w_stride ? in_attr_.w_stride : S;
    const int8_t pad = static_cast<int8_t>(114 - 128);

    float r = std::min(static_cast<float>(S) / width, static_cast<float>(S) / height);
    int nw = static_cast<int>(width * r);
    int nh = static_cast<int>(height * r);
    int ox = (S - nw) / 2;
    int oy = (S - nh) / 2;

    int8_t* dst = static_cast<int8_t*>(in_mem_->virt_addr);
    memset(dst, pad, static_cast<size_t>(stride) * S * 3);
    for (int y = 0; y < nh; ++y) {
        int sy = std::min(height - 1, static_cast<int>(y / r));
        const unsigned char* src_row = rgb + static_cast<size_t>(sy) * width * 3;
        int8_t* dst_row = dst + static_cast<size_t>(oy + y) * stride * 3 + ox * 3;
        for (int x = 0; x < nw; ++x) {
            dst_row[x * 3 + 0] = static_cast<int8_t>(static_cast<int>(src_row[x * 3 + 0]) - 128);
            dst_row[x * 3 + 1] = static_cast<int8_t>(static_cast<int>(src_row[x * 3 + 1]) - 128);
            dst_row[x * 3 + 2] = static_cast<int8_t>(static_cast<int>(src_row[x * 3 + 2]) - 128);
        }
    }

    rknn_mem_sync(ctx_, in_mem_, RKNN_MEMORY_SYNC_TO_DEVICE);
    if (rknn_run(ctx_, NULL) < 0) return false;
    for (uint32_t i = 0; i < n_output_; ++i)
        rknn_mem_sync(ctx_, out_mem_[i], RKNN_MEMORY_SYNC_FROM_DEVICE);

    std::vector<Box> persons;
    for (uint32_t o = 0; o < n_output_; ++o) {
        float* p = dequantize(out_nat_[o], out_orig_[o], out_mem_[o]->virt_addr);
        if (!p) continue;
        int nd = out_orig_[o].n_dims;
        int H = 0, W = 0, na = 3, C = 85;
        bool five_d = (nd == 5);
        if (five_d) {
            na = out_orig_[o].dims[1];
            H = out_orig_[o].dims[2];
            W = out_orig_[o].dims[3];
            C = out_orig_[o].dims[4];
        } else if (nd == 4) {
            C = out_orig_[o].dims[1];
            H = out_orig_[o].dims[2];
            W = out_orig_[o].dims[3];
            na = C / 85;
        } else {
            free(p);
            continue;
        }
        int stp = S / W;
        int ai = (stp == 8) ? 0 : (stp == 16 ? 1 : 2);
        for (int a = 0; a < na && a < 3; ++a) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    size_t base = five_d
                        ? ((static_cast<size_t>(a) * H + y) * W + x) * C
                        : (static_cast<size_t>(a) * 85 * H * W + static_cast<size_t>(y) * W + x);
                    float obj, cls, bx, by, bw, bh;
                    if (five_d) {
                        obj = sigmoid(p[base + 4]);
                        cls = sigmoid(p[base + 5]);
                        if (obj * cls < conf_) continue;   // 按乘积过滤: 各自 0.15 会放过 0.0225 的噪声
                        bx = p[base + 0]; by = p[base + 1];
                        bw = p[base + 2]; bh = p[base + 3];
                    } else {
                        obj = sigmoid(p[base + static_cast<size_t>(4) * H * W]);
                        cls = sigmoid(p[base + static_cast<size_t>(5) * H * W]);
                        if (obj * cls < conf_) continue;   // 同上
                        bx = p[base + 0];
                        by = p[base + 1];
                        bw = p[base + static_cast<size_t>(2) * H * W];
                        bh = p[base + static_cast<size_t>(3) * H * W];
                    }
                    float cx = (sigmoid(bx) * 2.0f - 0.5f + x) * stp;
                    float cy = (sigmoid(by) * 2.0f - 0.5f + y) * stp;
                    float ww = powf(sigmoid(bw) * 2.0f, 2) * kAnchors[ai][a * 2];
                    float hh = powf(sigmoid(bh) * 2.0f, 2) * kAnchors[ai][a * 2 + 1];
                    persons.push_back({cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2,
                                       obj * cls});
                }
            }
        }
        free(p);
    }
    // 诊断: 若存在 /root/yolo_dump_flag, 前 3 帧把输入 RGB 存 raw, 便于离线比对
    {
        static int dump_count = 0;
        if (dump_count < 3 && access("/root/yolo_dump_flag", F_OK) == 0) {
            char path[64];
            snprintf(path, sizeof(path), "/root/yolo_dump_%d.raw", dump_count);
            FILE* df = fopen(path, "wb");
            if (df) {
                fwrite(rgb, 1, (size_t)width * height * 3, df);
                fclose(df);
                char meta[96];
                snprintf(meta, sizeof(meta), "/root/yolo_dump_%d.txt", dump_count);
                FILE* mf = fopen(meta, "w");
                if (mf) { fprintf(mf, "%d %d\n", width, height); fclose(mf); }
                printf("[YOLO] dumped frame %d (%dx%d)\n", dump_count, width, height);
                ++dump_count;
            }
        }
    }
    nms(persons, 0.45f);

    // 反 letterbox 回原图并归一化
    for (size_t i = 0; i < persons.size(); ++i) {
        Box& b = persons[i];
        IvaObject person = {};
        person.id = 0;
        person.type = 0;
        person.score = b.score;
        person.x1 = std::max(0.0f, (b.x1 - ox) / r / width);
        person.y1 = std::max(0.0f, (b.y1 - oy) / r / height);
        person.x2 = std::min(1.0f, (b.x2 - ox) / r / width);
        person.y2 = std::min(1.0f, (b.y2 - oy) / r / height);
        float bw = person.x2 - person.x1, bh = person.y2 - person.y1;
        // 最小尺寸
        if (bw < 0.02f || bh < 0.05f) continue;
        // 过宽: 人体在正常机位下不会横向铺开 (椅子/沙发/窗帘常被误检)
        if (bw > 0.60f) continue;
        // 宽高比异常: 站/坐/躺的人体都远窄于 1.2 (躺姿约 1.0-1.4, 但只在贴边场景才误检)
        if (bw / bh > 1.20f) continue;
        // 贴边窄条: 画面边缘的碎片 (门框/柜边) 常见误检
        if (person.x1 <= 0.005f && bw < 0.06f) continue;
        if (person.x2 >= 0.995f && bw < 0.06f) continue;
        // 高度不设上限: 人走近镜头时框高可达 1.0, 限高会误杀
        result.people.push_back(person);
    }
    result.frame_id = 0;
    return true;
}

} // namespace dw
