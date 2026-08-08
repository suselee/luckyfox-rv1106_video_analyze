#include "rockiva_detector.h"

#include <stdio.h>
#include <string.h>

extern "C" {
#include "rockiva_common.h"
#include "rockiva_det_api.h"
}

namespace dw {

RockIvaDetector::RockIvaDetector()
    : handle_(NULL), detector_inited_(false), last_error_(0),
      callback_received_(false), callback_status_(ROCKIVA_UNKNOWN) {}

RockIvaDetector::~RockIvaDetector() { destroy(); }

bool RockIvaDetector::init(const std::string& model_dir, int width, int height,
                           int person_score, int face_score) {
    destroy();
    RockIvaInitParam global;
    memset(&global, 0, sizeof(global));
    global.logLevel = ROCKIVA_LOG_ERROR;
    snprintf(global.modelPath, sizeof(global.modelPath), "%s", model_dir.c_str());
    global.imageInfo.width = (uint16_t)width;
    global.imageInfo.height = (uint16_t)height;
    global.imageInfo.format = ROCKIVA_IMAGE_FORMAT_YUV420SP_NV12;
    global.imageInfo.transformMode = ROCKIVA_IMAGE_TRANSFORM_NONE;
    global.cameraType = ROCKIVA_CAMERA_TYPE_ONE;
    global.detModel = ROCKIVA_DET_MODEL_PFP;
    global.trackerVersion = 2;

    RockIvaHandle handle = NULL;
    RockIvaRetCode rc = ROCKIVA_Init(&handle, ROCKIVA_MODE_VIDEO, &global, this);
    if (rc != ROCKIVA_RET_SUCCESS) {
        last_error_ = (int)rc;
        printf("[ROCKIVA] init failed: %d\n", (int)rc);
        return false;
    }
    handle_ = handle;

    RockIvaDetTaskParams task;
    memset(&task, 0, sizeof(task));
    task.detObjectType = ROCKIVA_OBJECT_TYPE_BITMASK(ROCKIVA_OBJECT_TYPE_PERSON) |
                         ROCKIVA_OBJECT_TYPE_BITMASK(ROCKIVA_OBJECT_TYPE_FACE);
    task.scores[ROCKIVA_OBJECT_TYPE_PERSON] =
        (uint8_t)(person_score < 1 ? 1 : (person_score > 100 ? 100 : person_score));
    task.scores[ROCKIVA_OBJECT_TYPE_FACE] =
        (uint8_t)(face_score < 1 ? 1 : (face_score > 100 ? 100 : face_score));
    task.min_det_count = 1;
    rc = ROCKIVA_DETECT_Init((RockIvaHandle)handle_, &task,
                             &RockIvaDetector::result_callback);
    if (rc != ROCKIVA_RET_SUCCESS) {
        last_error_ = (int)rc;
        printf("[ROCKIVA] detector init failed: %d\n", (int)rc);
        ROCKIVA_Release((RockIvaHandle)handle_);
        handle_ = NULL;
        return false;
    }
    detector_inited_ = true;
    last_error_ = 0;
    printf("[ROCKIVA] PFP detector ready, video tracking enabled\n");
    return true;
}

void RockIvaDetector::destroy() {
    if (handle_) {
        if (detector_inited_) ROCKIVA_DETECT_Release((RockIvaHandle)handle_);
        ROCKIVA_Release((RockIvaHandle)handle_);
    }
    handle_ = NULL;
    detector_inited_ = false;
}

void RockIvaDetector::result_callback(const RockIvaDetectResult* result,
                                      RockIvaExecuteStatus status, void* userdata) {
    RockIvaDetector* self = reinterpret_cast<RockIvaDetector*>(userdata);
    if (!self) return;
    if (!result) {
        std::lock_guard<std::mutex> lock(self->mutex_);
        self->callback_received_ = true;
        self->callback_status_ = (int)status;
        return;
    }
    if (status != ROCKIVA_SUCCESS) {
        std::lock_guard<std::mutex> lock(self->mutex_);
        self->callback_received_ = true;
        self->callback_status_ = (int)status;
        return;
    }
    IvaResult converted;
    converted.frame_id = result->frameId;
    for (uint32_t i = 0; i < result->objNum; ++i) {
        const RockIvaObjectInfo& obj = result->objInfo[i];
        IvaObject item;
        item.id = obj.objId;
        item.type = (int)obj.type;
        item.score = obj.score / 100.0f;
        item.x1 = obj.rect.topLeft.x / 10000.0f;
        item.y1 = obj.rect.topLeft.y / 10000.0f;
        item.x2 = obj.rect.bottomRight.x / 10000.0f;
        item.y2 = obj.rect.bottomRight.y / 10000.0f;
        if (obj.type == ROCKIVA_OBJECT_TYPE_PERSON) converted.people.push_back(item);
        else if (obj.type == ROCKIVA_OBJECT_TYPE_FACE) converted.faces.push_back(item);
    }
    std::lock_guard<std::mutex> lock(self->mutex_);
    self->callback_received_ = true;
    self->callback_status_ = (int)status;
    self->latest_ = converted;
}

bool RockIvaDetector::detect_fd(uint32_t frame_id, int data_fd, int yuv420_layout,
                                int width, int height,
                                IvaResult& result, int timeout_ms) {
    return detect(frame_id, data_fd, 0, yuv420_layout, NULL,
                  width, height, result, timeout_ms);
}

bool RockIvaDetector::detect_physical(uint32_t frame_id, uintptr_t physical_addr,
                                      int yuv420_layout, int width, int height,
                                      IvaResult& result,
                                      int timeout_ms) {
    return detect(frame_id, -1, physical_addr, yuv420_layout, NULL,
                  width, height, result, timeout_ms);
}

bool RockIvaDetector::detect_nv12(uint32_t frame_id, uint8_t* data, int width, int height,
                                  IvaResult& result, int timeout_ms) {
    return detect(frame_id, -1, 0, 1, data, width, height, result, timeout_ms);
}

bool RockIvaDetector::detect(uint32_t frame_id, int data_fd, uintptr_t physical_addr,
                             int yuv420_layout, uint8_t* data,
                             int width, int height, IvaResult& result, int timeout_ms) {
    if (!handle_ || (!data && data_fd < 0 && physical_addr == 0)) return false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_ = IvaResult();
        latest_.frame_id = frame_id;
        callback_received_ = false;
        callback_status_ = ROCKIVA_UNKNOWN;
    }
    RockIvaImage image;
    memset(&image, 0, sizeof(image));
    image.frameId = frame_id;
    image.info.width = (uint16_t)width;
    image.info.height = (uint16_t)height;
    image.info.format = yuv420_layout == 2
        ? ROCKIVA_IMAGE_FORMAT_YUV420P_YU12
        : (yuv420_layout == 3
            ? ROCKIVA_IMAGE_FORMAT_YUV420P_YV12
            : ROCKIVA_IMAGE_FORMAT_YUV420SP_NV12);
    image.info.transformMode = ROCKIVA_IMAGE_TRANSFORM_NONE;
    image.dataFd = data_fd;
    image.dataAddr = data;
    image.dataPhyAddr = reinterpret_cast<uint8_t*>(physical_addr);
    image.size = (uint32_t)(width * height * 3 / 2);
    RockIvaRetCode rc = ROCKIVA_PushFrame((RockIvaHandle)handle_, &image, NULL);
    if (rc != ROCKIVA_RET_SUCCESS) {
        last_error_ = (int)rc;
        return false;
    }
    rc = ROCKIVA_WaitFinish((RockIvaHandle)handle_, frame_id, timeout_ms);
    if (rc != ROCKIVA_RET_SUCCESS) {
        last_error_ = (int)rc;
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    result = latest_;
    if (!callback_received_ || callback_status_ != ROCKIVA_SUCCESS) {
        last_error_ = callback_status_;
        return false;
    }
    return result.frame_id == frame_id;
}

} // namespace dw
