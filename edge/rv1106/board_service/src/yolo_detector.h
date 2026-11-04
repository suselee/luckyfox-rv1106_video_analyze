#pragma once
#include <stdint.h>
#include <vector>

#include "rknn_engine.h"
#include "rockiva_detector.h"   // IvaObject / IvaResult

namespace dw {

// YOLOv5n (RKNN, RV1106) 人体检测器。
//
// 为什么不用 RockIVA: 实测这个机位下 RockIVA 的 person 召回只有 8% (逆光+远景),
// 下游的体型/相对身高判据全部缺输入。YOLOv5 离线基准 65-96%, 单帧 ~33ms@416。
//
// 模型要求: 三尺度原始输出头 (1,3,H,W,85), 解码在 C++ 侧做 sigmoid + anchor。
// 不能用默认导出的单输出 (1,N,85): 框坐标与置信度共用一个量化 scale,
// INT8 下置信度会被压成 2 档, 板上直接检不出人。
class YoloPersonDetector {
public:
    // input_size: 模型输入边长 (320/416/512)
    // conf: 置信度阈值 (INT8 量化后建议 0.15-0.25)
    bool init(const char* model_path, int input_size, float conf);
    void destroy();
    bool ready() const { return inited_ && ctx_ != 0; }

    // rgb: width*height*3, 行优先
    // 输出填 result.people (归一化坐标 0..1); result.faces 不动 (人脸走 RetinaFace/RockIVA)
    bool detect(const unsigned char* rgb, int width, int height, IvaResult& result);

private:
    rknn_context ctx_ = 0;
    rknn_tensor_attr in_attr_{};
    std::vector<rknn_tensor_attr> out_nat_;
    std::vector<rknn_tensor_attr> out_orig_;
    rknn_tensor_mem* in_mem_ = nullptr;
    std::vector<rknn_tensor_mem*> out_mem_;
    uint32_t n_output_ = 0;
    int input_size_ = 416;
    float conf_ = 0.20f;
    bool inited_ = false;

    std::vector<int8_t> input_buf_;   // letterbox 后的 INT8 输入
    std::vector<float>  deq_buf_;     // 反量化缓冲区
};

} // namespace dw
