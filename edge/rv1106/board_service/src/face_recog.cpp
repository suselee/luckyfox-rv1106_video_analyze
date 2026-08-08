#include "face_recog.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#include "stb/stb_image_resize.h"

namespace dw {

// ArcFace 112x112 标准 5 点模板 (左眼,右眼,鼻,左嘴角,右嘴角)
static const float ARC_TEMPLATE[5][2] = {
    {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f},
    {41.5493f, 92.3655f}, {70.7299f, 92.2041f}};

namespace {

// 解 4x4 线性方程组 A u = b (高斯消元, 带部分主元)。成功返回 true。
bool solve4(double A[4][4], double b[4], double u[4]) {
    for (int col = 0; col < 4; col++) {
        int piv = col;
        for (int r = col + 1; r < 4; r++)
            if (fabs(A[r][col]) > fabs(A[piv][col])) piv = r;
        if (fabs(A[piv][col]) < 1e-12) return false;
        if (piv != col) {
            for (int c = 0; c < 4; c++) { double t = A[col][c]; A[col][c] = A[piv][c]; A[piv][c] = t; }
            double t = b[col]; b[col] = b[piv]; b[piv] = t;
        }
        for (int r = 0; r < 4; r++) {
            if (r == col) continue;
            double f = A[r][col] / A[col][col];
            for (int c = 0; c < 4; c++) A[r][c] -= f * A[col][c];
            b[r] -= f * b[col];
        }
    }
    for (int i = 0; i < 4; i++) u[i] = b[i] / A[i][i];
    return true;
}

// 用 5 点最小二乘拟合相似变换: (X,Y) = (a*x - b*y + tx, b*x + a*y + ty)
// 映射 检测点(src, 像素) -> 模板点(dst)。返回 u=[a,b,tx,ty]。
bool fit_similarity(const float src[5][2], const float dst[5][2], double u[4]) {
    double A[4][4] = {{0}};
    double rhs[4]  = {0};
    // 每点两条方程:
    //   [x, -y, 1, 0]·u = X
    //   [y,  x, 0, 1]·u = Y
    for (int i = 0; i < 5; i++) {
        double x = src[i][0], y = src[i][1], X = dst[i][0], Y = dst[i][1];
        double r1[4] = { x, -y, 1, 0 };
        double r2[4] = { y,  x, 0, 1 };
        for (int a = 0; a < 4; a++) {
            for (int c = 0; c < 4; c++) A[a][c] += r1[a] * r1[c] + r2[a] * r2[c];
            rhs[a] += r1[a] * X + r2[a] * Y;
        }
    }
    return solve4(A, rhs, u);
}

inline unsigned char bilinear(const unsigned char* img, int w, int h, float x, float y, int ch, int c) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > w - 1) x = w - 1;
    if (y > h - 1) y = h - 1;
    int x0 = (int)x, y0 = (int)y;
    int x1 = x0 + 1 < w ? x0 + 1 : x0;
    int y1 = y0 + 1 < h ? y0 + 1 : y0;
    float dx = x - x0, dy = y - y0;
    const unsigned char* p00 = img + ((size_t)y0 * w + x0) * ch;
    const unsigned char* p01 = img + ((size_t)y0 * w + x1) * ch;
    const unsigned char* p10 = img + ((size_t)y1 * w + x0) * ch;
    const unsigned char* p11 = img + ((size_t)y1 * w + x1) * ch;
    float top = p00[c] * (1 - dx) + p01[c] * dx;
    float bot = p10[c] * (1 - dx) + p11[c] * dx;
    return (unsigned char)(top * (1 - dy) + bot * dy + 0.5f);
}

// 对齐: 用关键点把人脸 warp 到 in_w x in_h。成功返回 true。
bool align_face(const unsigned char* rgb, int img_w, int img_h,
                const FaceBox& box, int in_w, int in_h, unsigned char* out) {
    // 关键点: 归一化 -> 像素
    float src[5][2], dst[5][2];
    for (int k = 0; k < 5; k++) {
        src[k][0] = box.lmk[k * 2 + 0] * img_w;
        src[k][1] = box.lmk[k * 2 + 1] * img_h;
        // 模板按输入尺寸缩放 (标准模板基于 112)
        dst[k][0] = ARC_TEMPLATE[k][0] * in_w / 112.0f;
        dst[k][1] = ARC_TEMPLATE[k][1] * in_h / 112.0f;
    }

    double u[4];
    if (!fit_similarity(src, dst, u)) return false;
    double a = u[0], b = u[1], tx = u[2], ty = u[3];
    double s2 = a * a + b * b;
    if (s2 < 1e-9) return false;

    // 对每个输出像素 (X,Y), 逆映射回源图 (x,y) 并双线性采样
    for (int Y = 0; Y < in_h; Y++) {
        for (int X = 0; X < in_w; X++) {
            double xt = X - tx, yt = Y - ty;
            float sx = (float)((a * xt + b * yt) / s2);
            float sy = (float)((-b * xt + a * yt) / s2);
            unsigned char* o = out + ((size_t)Y * in_w + X) * 3;
            for (int c = 0; c < 3; c++)
                o[c] = bilinear(rgb, img_w, img_h, sx, sy, 3, c);
        }
    }
    return true;
}

// 回退: 无对齐时 box 裁剪 + 缩放
bool crop_resize(const unsigned char* rgb, int img_w, int img_h,
                 const FaceBox& box, int in_w, int in_h, unsigned char* out) {
    int fx1 = (int)(box.x1 * img_w), fy1 = (int)(box.y1 * img_h);
    int fx2 = (int)(box.x2 * img_w), fy2 = (int)(box.y2 * img_h);
    if (fx1 < 0) fx1 = 0;
    if (fy1 < 0) fy1 = 0;
    if (fx2 > img_w) fx2 = img_w;
    if (fy2 > img_h) fy2 = img_h;
    int fw = fx2 - fx1, fh = fy2 - fy1;
    if (fw <= 0 || fh <= 0) return false;
    std::vector<unsigned char> crop((size_t)fw * fh * 3);
    for (int y = 0; y < fh; y++)
        memcpy(crop.data() + (size_t)y * fw * 3,
               rgb + ((size_t)(fy1 + y) * img_w + fx1) * 3, (size_t)fw * 3);
    stbir_resize_uint8(crop.data(), fw, fh, 0, out, in_w, in_h, 0, 3);
    return true;
}

} // namespace

bool FaceRecognizer::init(const char* model_path) {
    if (rknn_model_init(&m_, model_path) < 0) return false;
    rknn_model_input_hw(&m_, &in_w_, &in_h_);
    feat_dim_ = m_.orig_out_attrs[0].n_elems;
    inited_   = true;
    printf("[RECOG] input=%dx%d, feature dim=%d\n", in_w_, in_h_, feat_dim_);
    return feat_dim_ > 0;
}

void FaceRecognizer::destroy() {
    if (inited_) { rknn_model_destroy(&m_); inited_ = false; }
}

bool FaceRecognizer::extract(const unsigned char* rgb, int img_w, int img_h,
                             const FaceBox& box, std::vector<float>& emb) {
    if (!inited_) return false;

    std::vector<unsigned char> in((size_t)in_w_ * in_h_ * 3);
    if (!align_face(rgb, img_w, img_h, box, in_w_, in_h_, in.data())) {
        if (!crop_resize(rgb, img_w, img_h, box, in_w_, in_h_, in.data())) return false;
    }

    rknn_model_set_input(&m_, in.data(), in_w_, in_h_, 3);
    if (rknn_run(m_.ctx, NULL) < 0) return false;

    float* feat = rknn_model_get_output_float(&m_, 0);
    emb.assign(feat, feat + feat_dim_);
    free(feat);
    return true;
}

} // namespace dw
