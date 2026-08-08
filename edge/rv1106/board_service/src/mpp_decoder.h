#pragma once
#include <vector>
#include <cstdint>

namespace dw {

class Nv12Frame {
public:
    Nv12Frame();
    ~Nv12Frame();
    Nv12Frame(Nv12Frame&& other);
    Nv12Frame& operator=(Nv12Frame&& other);

    bool valid() const { return frame_info_ != nullptr; }
    bool is_nv12() const;
    // 1=NV12, 2=YU12/I420, 3=YV12; 0=unsupported.
    int yuv420_layout() const;
    bool zero_copy_compatible() const;
    int width() const;
    int height() const;
    int stride() const;
    int data_fd() const;
    uintptr_t physical_addr() const;
    uint8_t* data_addr() const;
    bool copy_nv12(std::vector<uint8_t>& nv12) const;
    bool to_rgb(std::vector<uint8_t>& rgb) const;
    void release();

private:
    friend class MppDecoder;
    Nv12Frame(const Nv12Frame&);
    Nv12Frame& operator=(const Nv12Frame&);
    void* frame_info_;
    int channel_;
};

// RockIt/MPP H.264 解码器封装。是否由硬件执行取决于 SoC；RV1106
// 没有 VDEC 硬件块，因此应配合 keyframes_only 避免软件解码全部 P 帧。
// 用法: init() -> 反复 send()喂H.264访问单元(Annex-B, 含起始码) + get_rgb()取解码帧。
// 输出为 RGB888, 尺寸 = 码流原生分辨率(由解码器给出)。
class MppDecoder {
public:
    explicit MppDecoder(int max_w = 1920, int max_h = 1080);
    ~MppDecoder();

    bool init();
    void deinit();

    // 送一个 H.264 数据块(Annex-B, 需自带 00 00 00 01 起始码)。
    // end_of_frame=true 表示这是当前帧的最后一包(VIDEO_MODE_FRAME 下按帧送)。
    bool send(const uint8_t* data, int len, uint64_t pts, bool end_of_frame);

    // 取一帧解码结果。convert=false 时只释放该帧，不做昂贵的 RGB 转换，
    // 用于在目标分析 FPS 之外快速丢帧。timeout_ms: -1阻塞,0非阻塞,>0超时。
    // convert=true 成功时 rgb 填 out_w*out_h*3 字节。返回是否取到帧。
    bool get_rgb(std::vector<uint8_t>& rgb, int& out_w, int& out_h,
                 int timeout_ms, bool convert = true);

    // Acquire one decoded frame and keep the VDEC buffer alive until the
    // Nv12Frame is destroyed. This lets RockIVA consume the DMA fd directly.
    bool get_frame(Nv12Frame& frame, int timeout_ms);

private:
    int         chn_    = 0;
    uint32_t    pool_   = 0;   // MB_POOL
    bool        inited_ = false;
    bool        chn_ok_ = false;
    int         max_w_  = 1920;
    int         max_h_  = 1080;
};

} // namespace dw
