#include "face_detect.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

#include "stb/stb_image_resize.h"

namespace dw {

// RetinaFace (320x320) priorbox 配置
static const int   RF_STEPS[3]        = {8, 16, 32};
static const float RF_MIN_SIZES[3][2] = {{16, 32}, {64, 128}, {256, 512}};
static const float RF_VAR[2]          = {0.1f, 0.2f};

namespace {

struct Prior { float cx, cy, sx, sy; }; // 均为归一化值

const std::vector<Prior>& priors() {
    static std::vector<Prior> p;
    if (!p.empty()) return p;
    for (int f = 0; f < 3; f++) {
        int step   = RF_STEPS[f];
        int feat_h = (FaceDetector::INPUT_H + step - 1) / step;
        int feat_w = (FaceDetector::INPUT_W + step - 1) / step;
        for (int i = 0; i < feat_h; i++)
            for (int j = 0; j < feat_w; j++)
                for (int k = 0; k < 2; k++) {
                    float ms = RF_MIN_SIZES[f][k];
                    Prior pr;
                    pr.cx = (j + 0.5f) * step / FaceDetector::INPUT_W;
                    pr.cy = (i + 0.5f) * step / FaceDetector::INPUT_H;
                    pr.sx = ms / FaceDetector::INPUT_W;
                    pr.sy = ms / FaceDetector::INPUT_H;
                    p.push_back(pr);
                }
    }
    return p; // 4200
}

float iou(const FaceBox& a, const FaceBox& b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float iw = ix2 - ix1 > 0 ? ix2 - ix1 : 0;
    float ih = iy2 - iy1 > 0 ? iy2 - iy1 : 0;
    float inter  = iw * ih;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

std::vector<FaceBox> nms(std::vector<FaceBox>& boxes, float iou_thresh) {
    std::sort(boxes.begin(), boxes.end(),
              [](const FaceBox& a, const FaceBox& b) { return a.score > b.score; });
    std::vector<FaceBox> keep;
    std::vector<char> removed(boxes.size(), 0);
    for (size_t i = 0; i < boxes.size(); i++) {
        if (removed[i]) continue;
        keep.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); j++)
            if (!removed[j] && iou(boxes[i], boxes[j]) > iou_thresh)
                removed[j] = 1;
    }
    return keep;
}

// 按 n_elems 辨认 loc/conf/landms 输出索引 (与输出顺序无关)。
void identify_outputs(RknnModel* m, int num_priors, int* loc_i, int* conf_i, int* lmk_i) {
    *loc_i = *conf_i = *lmk_i = -1;
    for (uint32_t i = 0; i < m->io_num.n_output; i++) {
        int n = m->orig_out_attrs[i].n_elems;
        if (n == num_priors * 4)  *loc_i  = i;
        else if (n == num_priors * 2)  *conf_i = i;
        else if (n == num_priors * 10) *lmk_i  = i;
    }
}

} // namespace

bool FaceDetector::init(const char* model_path) {
    if (rknn_model_init(&m_, model_path) < 0) return false;
    inited_ = true;
    return true;
}

void FaceDetector::destroy() {
    if (inited_) { rknn_model_destroy(&m_); inited_ = false; }
}

std::vector<FaceBox> FaceDetector::detect(const unsigned char* rgb, int img_w, int img_h,
                                          float score_thresh, float iou_thresh) {
    std::vector<FaceBox> result;
    if (!inited_) return result;

    // Letterbox: 等比缩放到 320x320 画布, 保持宽高比, 四周补均值色(避免变形)。
    // scale/pad 用于把检测结果坐标换算回原帧。
    float scale = std::min((float)INPUT_W / img_w, (float)INPUT_H / img_h);
    int   rw    = (int)(img_w * scale + 0.5f);
    int   rh    = (int)(img_h * scale + 0.5f);
    if (rw > INPUT_W) rw = INPUT_W;
    if (rh > INPUT_H) rh = INPUT_H;
    int   pad_x = (INPUT_W - rw) / 2;
    int   pad_y = (INPUT_H - rh) / 2;

    // 先等比缩放到 rw x rh
    std::vector<unsigned char> resized((size_t)rw * rh * 3);
    stbir_resize_uint8(rgb, img_w, img_h, 0, resized.data(), rw, rh, 0, 3);

    // 画布填 RetinaFace 均值色 (RGB=123,117,104 -> 转 BGR 后即 [104,117,123],
    // 模型内部减均值后补边区域≈0, 不产生伪边缘), 再把缩放图贴到中间, 顺带转 BGR。
    std::vector<unsigned char> bgr((size_t)INPUT_W * INPUT_H * 3);
    for (size_t i = 0; i < (size_t)INPUT_W * INPUT_H; i++) {
        bgr[i * 3 + 0] = 104; // B
        bgr[i * 3 + 1] = 117; // G
        bgr[i * 3 + 2] = 123; // R
    }
    for (int y = 0; y < rh; y++) {
        for (int x = 0; x < rw; x++) {
            const unsigned char* s = resized.data() + ((size_t)y * rw + x) * 3;
            unsigned char* d = bgr.data() + ((size_t)(y + pad_y) * INPUT_W + (x + pad_x)) * 3;
            d[0] = s[2]; // R->B
            d[1] = s[1];
            d[2] = s[0]; // B->R
        }
    }

    rknn_model_set_input(&m_, bgr.data(), INPUT_W, INPUT_H, 3);
    if (rknn_run(m_.ctx, NULL) < 0) return result;

    const std::vector<Prior>& pr = priors();
    int num = (int)pr.size();

    int loc_i, conf_i, lmk_i;
    identify_outputs(&m_, num, &loc_i, &conf_i, &lmk_i);
    if (loc_i < 0 || conf_i < 0 || lmk_i < 0) {
        printf("[DET] 输出辨认失败 (期望 loc/conf/landms, N=%d)\n", num);
        return result;
    }

    float* loc  = rknn_model_get_output_float(&m_, loc_i);   // [N,4]
    float* conf = rknn_model_get_output_float(&m_, conf_i);  // [N,2] (softmax 概率)
    float* lmk  = rknn_model_get_output_float(&m_, lmk_i);   // [N,10]

    std::vector<FaceBox> cand;
    for (int i = 0; i < num; i++) {
        float score = conf[i * 2 + 1]; // face 概率
        if (score < score_thresh) continue;

        const Prior& p = pr[i];
        // 解码框
        float cx = p.cx + loc[i * 4 + 0] * RF_VAR[0] * p.sx;
        float cy = p.cy + loc[i * 4 + 1] * RF_VAR[0] * p.sy;
        float bw = p.sx * expf(loc[i * 4 + 2] * RF_VAR[1]);
        float bh = p.sy * expf(loc[i * 4 + 3] * RF_VAR[1]);

        FaceBox fb;
        fb.x1 = cx - bw * 0.5f;
        fb.y1 = cy - bh * 0.5f;
        fb.x2 = cx + bw * 0.5f;
        fb.y2 = cy + bh * 0.5f;
        fb.score = score;

        // 解码 5 关键点
        for (int k = 0; k < 5; k++) {
            fb.lmk[k * 2 + 0] = p.cx + lmk[i * 10 + k * 2 + 0] * RF_VAR[0] * p.sx;
            fb.lmk[k * 2 + 1] = p.cy + lmk[i * 10 + k * 2 + 1] * RF_VAR[0] * p.sy;
        }
        cand.push_back(fb);
    }

    free(loc);
    free(conf);
    free(lmk);

    result = nms(cand, iou_thresh);

    // 坐标从 320x320 画布(letterbox)换算回原帧的归一化值:
    //   frame_px = (canvas_norm * INPUT - pad) / scale ; frame_norm = frame_px / img
    for (FaceBox& b : result) {
        b.x1 = ((b.x1 * INPUT_W - pad_x) / scale) / img_w;
        b.x2 = ((b.x2 * INPUT_W - pad_x) / scale) / img_w;
        b.y1 = ((b.y1 * INPUT_H - pad_y) / scale) / img_h;
        b.y2 = ((b.y2 * INPUT_H - pad_y) / scale) / img_h;
        for (int k = 0; k < 5; k++) {
            b.lmk[k * 2 + 0] = ((b.lmk[k * 2 + 0] * INPUT_W - pad_x) / scale) / img_w;
            b.lmk[k * 2 + 1] = ((b.lmk[k * 2 + 1] * INPUT_H - pad_y) / scale) / img_h;
        }
        // 框裁剪到 [0,1] (关键点保持原值, 对齐时会自行处理)
        if (b.x1 < 0) b.x1 = 0;
        if (b.y1 < 0) b.y1 = 0;
        if (b.x2 > 1) b.x2 = 1;
        if (b.y2 > 1) b.y2 = 1;
    }
    return result;
}

} // namespace dw
