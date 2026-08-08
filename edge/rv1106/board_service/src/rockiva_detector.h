#pragma once

#include <stdint.h>
#include <mutex>
#include <string>
#include <vector>

extern "C" {
#include "rockiva_common.h"
}

namespace dw {

struct IvaObject {
    uint32_t id;
    int type;
    float score;
    float x1;
    float y1;
    float x2;
    float y2;
};

struct IvaResult {
    uint32_t frame_id;
    std::vector<IvaObject> people;
    std::vector<IvaObject> faces;
};

class RockIvaDetector {
public:
    RockIvaDetector();
    ~RockIvaDetector();

    bool init(const std::string& model_dir, int width, int height,
              int person_score, int face_score);
    void destroy();
    bool detect_fd(uint32_t frame_id, int data_fd, int yuv420_layout,
                   int width, int height,
                   IvaResult& result, int timeout_ms = 1000);
    bool detect_physical(uint32_t frame_id, uintptr_t physical_addr,
                         int yuv420_layout, int width, int height,
                         IvaResult& result,
                         int timeout_ms = 1000);
    bool detect_nv12(uint32_t frame_id, uint8_t* data, int width, int height,
                     IvaResult& result, int timeout_ms = 1000);
    int last_error() const { return last_error_; }

private:
    static void result_callback(const RockIvaDetectResult* result,
                                RockIvaExecuteStatus status, void* userdata);
    bool detect(uint32_t frame_id, int data_fd, uintptr_t physical_addr,
                int yuv420_layout, uint8_t* data,
                int width, int height, IvaResult& result, int timeout_ms);

    void* handle_;
    bool detector_inited_;
    int last_error_;
    std::mutex mutex_;
    IvaResult latest_;
    bool callback_received_;
    int callback_status_;
};

} // namespace dw
