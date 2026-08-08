#pragma once
#include "rknn_engine.h"
#include "face_detect.h"
#include <vector>

namespace dw {

// MobileFaceNet 人脸特征提取器 (输入 112x112 RGB, 输出 512 维)。
// 用 RetinaFace 的 5 关键点做仿射对齐到 ArcFace 标准模板, 再送模型 —— 对
// 婴儿侧脸/大角度更稳。对齐失败(如关键点异常)时回退为 box 裁剪+缩放。
class FaceRecognizer {
public:
    bool init(const char* model_path);
    void destroy();

    int dim() const { return feat_dim_; }

    // 提取特征。emb 填入 feat_dim_ 个 float(未归一化, 由 FaceDB 归一化)。
    bool extract(const unsigned char* rgb, int img_w, int img_h,
                 const FaceBox& box, std::vector<float>& emb);

private:
    RknnModel m_{};
    bool      inited_   = false;
    int       in_w_     = 112;
    int       in_h_     = 112;
    int       feat_dim_ = 0;
};

} // namespace dw
