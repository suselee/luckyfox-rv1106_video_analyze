#pragma once
#include <sys/time.h>

namespace dw {

// 统一墙钟 (秒): 与 main.cpp / track_fusion 的 now_seconds() 同源 (gettimeofday)。
// 环形缓冲时间戳、切片窗口、上传元数据都基于它, 保证跨线程一致。
inline double now_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

} // namespace dw