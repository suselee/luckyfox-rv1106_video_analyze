#pragma once
#include "rknn_engine.h"
#include <vector>

namespace dw {

// 人脸框 + 5 关键点, 坐标均为相对分析帧的归一化值 [0,1]。
// 关键点顺序 (RetinaFace/ArcFace 约定): 左眼, 右眼, 鼻尖, 左嘴角, 右嘴角。
struct FaceBox {
    float x1, y1, x2, y2, score;
    float lmk[10]; // [x0,y0, x1,y1, ... x4,y4]
};

// RetinaFace (mobilenet0.25) 人脸检测器, 输入 320x320。
// 复用 Rockchip rknn_model_zoo 的 RetinaFace_mobile320:
//   3 个输出: loc[1,N,4], conf[1,N,2], landms[1,N,10]  (N=4200 priors)
//   本实现按 n_elems 自动辨认三者顺序, 不依赖输出索引。
// 注意: RetinaFace 权重按 BGR + 均值[104,117,123] 训练, 故喂入前需转 BGR
//       (转 rknn 时 mean_values=[[104,117,123]], std=[[1,1,1]])。
class FaceDetector {
public:
    static const int INPUT_W = 320;
    static const int INPUT_H = 320;

    bool init(const char* model_path);
    void destroy();

    // 输入整帧 RGB (img_w x img_h), 返回归一化坐标的人脸框(含关键点, 已 NMS, 分数降序)。
    std::vector<FaceBox> detect(const unsigned char* rgb, int img_w, int img_h,
                                float score_thresh = 0.5f, float iou_thresh = 0.4f);

private:
    RknnModel m_{};
    bool      inited_ = false;
};

} // namespace dw
