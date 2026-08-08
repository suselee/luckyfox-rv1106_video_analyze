#pragma once
#include "rknn_api.h"

// 注: rknn 的类型 (rknn_context / rknn_tensor_attr ...) 在全局命名空间。
// 命名空间 rknpu2 来自 fp16/Float16.h, 仅 rknn_engine.cpp 里(引入该头后)用到。

namespace dw {

// RV1106 RKNN 模型封装 (零拷贝 IO)。
// 直接沿用 rknn_face_emotion / rknn_live_monitor 中验证过的实现。
struct RknnModel {
    rknn_context           ctx            = 0;
    rknn_input_output_num  io_num         = {};
    rknn_tensor_attr*      input_attrs    = nullptr;
    rknn_tensor_attr*      output_attrs   = nullptr;  // native (可能 NC1HWC2)
    rknn_tensor_attr*      orig_out_attrs = nullptr;  // 原始 NCHW 形状
    rknn_tensor_mem**      input_mems     = nullptr;
    rknn_tensor_mem**      output_mems    = nullptr;
};

// 加载模型并绑定 IO。成功返回 0。
int rknn_model_init(RknnModel* m, const char* path);

// 把 HxWxC 的 uint8 数据拷入输入 (处理 w_stride)。
void rknn_model_set_input(RknnModel* m, const unsigned char* data,
                          int width, int height, int channels);

// 取第 idx 个输出并反量化为 float (调用者负责 free)。
float* rknn_model_get_output_float(RknnModel* m, int idx);

// 查询输入的 H/W (NHWC 或 NCHW 均适配)。
void rknn_model_input_hw(RknnModel* m, int* w, int* h);

void rknn_model_destroy(RknnModel* m);

} // namespace dw
