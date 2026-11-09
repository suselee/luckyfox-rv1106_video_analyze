#include "mpp_decoder.h"

#include <unistd.h>

#include <stdio.h>
#include <string.h>

extern "C" {
#include "rk_mpi_sys.h"
#include "rk_mpi_vdec.h"
#include "rk_mpi_mb.h"
#include "rk_comm_vdec.h"
#include "rk_comm_video.h"
}

namespace dw {

// 输入码流 MB 池参数
#define STREAM_MB_SIZE   (1024 * 1024)  // 单块 1MB, 足够 640x360 的 I 帧
#define STREAM_MB_CNT    8

static inline uint8_t clamp8(int v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }

// 带 stride 的 NV12(planar, Y + 交织UV) -> RGB888, BT.601
static void nv12_planar_to_rgb(const uint8_t* y, const uint8_t* uv, int stride,
                               int w, int h, uint8_t* rgb) {
    for (int row = 0; row < h; row++) {
        const uint8_t* yr  = y  + (size_t)row * stride;
        const uint8_t* uvr = uv + (size_t)(row / 2) * stride;
        uint8_t*       o   = rgb + (size_t)row * w * 3;
        for (int col = 0; col < w; col++) {
            int Y = yr[col];
            int U = uvr[(col & ~1)] - 128;
            int V = uvr[(col & ~1) + 1] - 128;
            o[col * 3 + 0] = clamp8(Y + ((359 * V) >> 8));            // R
            o[col * 3 + 1] = clamp8(Y - ((88 * U + 183 * V) >> 8));   // G
            o[col * 3 + 2] = clamp8(Y + ((454 * U) >> 8));            // B
        }
    }
}

static void yuv420p_to_rgb(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                           int y_stride, int uv_stride,
                           int w, int h, uint8_t* rgb) {
    for (int row = 0; row < h; row++) {
        const uint8_t* yr = y + (size_t)row * y_stride;
        const uint8_t* ur = u + (size_t)(row / 2) * uv_stride;
        const uint8_t* vr = v + (size_t)(row / 2) * uv_stride;
        uint8_t* o = rgb + (size_t)row * w * 3;
        for (int col = 0; col < w; col++) {
            int Y = yr[col];
            int U = ur[col / 2] - 128;
            int V = vr[col / 2] - 128;
            o[col * 3 + 0] = clamp8(Y + ((359 * V) >> 8));
            o[col * 3 + 1] = clamp8(Y - ((88 * U + 183 * V) >> 8));
            o[col * 3 + 2] = clamp8(Y + ((454 * U) >> 8));
        }
    }
}

static bool g_sys_inited = false;

Nv12Frame::Nv12Frame() : frame_info_(NULL), channel_(-1) {}

Nv12Frame::~Nv12Frame() { release(); }

Nv12Frame::Nv12Frame(Nv12Frame&& other)
    : frame_info_(other.frame_info_), channel_(other.channel_) {
    other.frame_info_ = NULL;
    other.channel_ = -1;
}

Nv12Frame& Nv12Frame::operator=(Nv12Frame&& other) {
    if (this != &other) {
        release();
        frame_info_ = other.frame_info_;
        channel_ = other.channel_;
        other.frame_info_ = NULL;
        other.channel_ = -1;
    }
    return *this;
}

static VIDEO_FRAME_INFO_S* frame_info(void* value) {
    return reinterpret_cast<VIDEO_FRAME_INFO_S*>(value);
}

bool Nv12Frame::is_nv12() const {
    return valid() && frame_info(frame_info_)->stVFrame.enPixelFormat == RK_FMT_YUV420SP;
}

int Nv12Frame::yuv420_layout() const {
    if (!valid()) return 0;
    PIXEL_FORMAT_E format = frame_info(frame_info_)->stVFrame.enPixelFormat;
    if (format == RK_FMT_YUV420SP) return 1;
    if (format == RK_FMT_YUV420P) return 2;
    if (format == RK_FMT_YUV420P_VU) return 3;
    return 0;
}

bool Nv12Frame::zero_copy_compatible() const {
    if (!yuv420_layout()) return false;
    const VIDEO_FRAME_S& f = frame_info(frame_info_)->stVFrame;
    // RockIVA's RV1106 path is designed for RockIt DMA buffers and can obtain
    // the aligned stride from the fd/MB metadata. CPU-address NV12 instead
    // enters a libyuv path that is not present in the minimal Luckfox image.
    return f.pMbBlk != NULL &&
           (RK_MPI_MB_Handle2Fd(f.pMbBlk) >= 0 ||
            RK_MPI_MB_Handle2PhysAddr(f.pMbBlk) != 0);
}

int Nv12Frame::width() const {
    return valid() ? (int)frame_info(frame_info_)->stVFrame.u32Width : 0;
}

int Nv12Frame::height() const {
    return valid() ? (int)frame_info(frame_info_)->stVFrame.u32Height : 0;
}

int Nv12Frame::stride() const {
    if (!valid()) return 0;
    const VIDEO_FRAME_S& f = frame_info(frame_info_)->stVFrame;
    return f.u32VirWidth ? (int)f.u32VirWidth : (int)f.u32Width;
}

int Nv12Frame::data_fd() const {
    if (!valid() || !frame_info(frame_info_)->stVFrame.pMbBlk) return -1;
    return RK_MPI_MB_Handle2Fd(frame_info(frame_info_)->stVFrame.pMbBlk);
}

uintptr_t Nv12Frame::physical_addr() const {
    if (!valid() || !frame_info(frame_info_)->stVFrame.pMbBlk) return 0;
    return (uintptr_t)RK_MPI_MB_Handle2PhysAddr(frame_info(frame_info_)->stVFrame.pMbBlk);
}

uint8_t* Nv12Frame::data_addr() const {
    if (!valid()) return NULL;
    return reinterpret_cast<uint8_t*>(frame_info(frame_info_)->stVFrame.pVirAddr[0]);
}

bool Nv12Frame::copy_nv12(std::vector<uint8_t>& nv12) const {
    if (!is_nv12()) return false;
    const VIDEO_FRAME_S& f = frame_info(frame_info_)->stVFrame;
    int w = (int)f.u32Width;
    int h = (int)f.u32Height;
    int y_stride = f.u32VirWidth ? (int)f.u32VirWidth : w;
    int vir_h = f.u32VirHeight ? (int)f.u32VirHeight : h;
    const uint8_t* y = reinterpret_cast<const uint8_t*>(f.pVirAddr[0]);
    if (!y) return false;
    const uint8_t* uv = f.pVirAddr[1]
        ? reinterpret_cast<const uint8_t*>(f.pVirAddr[1])
        : y + (size_t)y_stride * vir_h;
    nv12.resize((size_t)w * h * 3 / 2);
    for (int row = 0; row < h; ++row)
        memcpy(nv12.data() + (size_t)row * w, y + (size_t)row * y_stride, w);
    uint8_t* dst_uv = nv12.data() + (size_t)w * h;
    for (int row = 0; row < h / 2; ++row)
        memcpy(dst_uv + (size_t)row * w, uv + (size_t)row * y_stride, w);
    return true;
}

// 诊断: 存在 /root/yolo_dump_flag 时, 首帧把帧参数与原始平面落盘, 用于离线定位 stride/格式问题
static void dump_frame_planes(const VIDEO_FRAME_S& f) {
    static int dbg_count = 0;
    if (dbg_count >= 1) return;
    if (access("/root/yolo_dump_flag", F_OK) != 0) return;
    FILE* mf = fopen("/root/frame_meta.txt", "w");
    if (mf) {
        fprintf(mf, "enPixelFormat=%d w=%u h=%u virW=%u virH=%u p0=%p p1=%p\n",
                (int)f.enPixelFormat, f.u32Width, f.u32Height,
                f.u32VirWidth, f.u32VirHeight,
                (void*)f.pVirAddr[0], (void*)f.pVirAddr[1]);
        fclose(mf);
    }
    int w = (int)f.u32Width, h = (int)f.u32Height;
    int ys = f.u32VirWidth ? (int)f.u32VirWidth : w;
    int vh = f.u32VirHeight ? (int)f.u32VirHeight : h;
    if (f.pVirAddr[0]) {
        FILE* yf = fopen("/root/plane_Y.bin", "wb");
        if (yf) { fwrite(f.pVirAddr[0], 1, (size_t)ys * vh, yf); fclose(yf); }
    }
    {
        const uint8_t* y0 = reinterpret_cast<const uint8_t*>(f.pVirAddr[0]);
        const uint8_t* guess = y0 + (size_t)ys * vh;
        const uint8_t* p1 = reinterpret_cast<const uint8_t*>(f.pVirAddr[1]);
        const uint8_t* uvp = (p1 && p1 >= guess) ? p1 : guess;
        FILE* uf = fopen("/root/plane_UV.bin", "wb");
        if (uf) { fwrite(uvp, 1, (size_t)ys * vh / 2, uf); fclose(uf); }
    }
    printf("[MPP] dumped frame planes fmt=%d %ux%u vir=%ux%u\n",
           (int)f.enPixelFormat, w, h, f.u32VirWidth, f.u32VirHeight);
    dbg_count++;
}

bool Nv12Frame::to_rgb(std::vector<uint8_t>& rgb) const {
    if (!valid()) return false;
    const VIDEO_FRAME_S& f = frame_info(frame_info_)->stVFrame;
    dump_frame_planes(f);
    int w = (int)f.u32Width;
    int h = (int)f.u32Height;
    int y_stride = f.u32VirWidth ? (int)f.u32VirWidth : w;
    int vir_h = f.u32VirHeight ? (int)f.u32VirHeight : h;
    if (f.enPixelFormat == RK_FMT_YUV420SP && f.pVirAddr[0]) {
        const uint8_t* y = reinterpret_cast<const uint8_t*>(f.pVirAddr[0]);
        // RockIt 的 VIDEO_FRAME_S 在 NV12 下常把 pVirAddr[1] 也填成 Y 平面首地址
        // (实测 p0 == p1), 直接当 UV 用会把亮度当色度, 导致画面严重偏色/错位。
        // 只有当 p1 确实落在 Y 平面之后时才采信, 否则按 Y 平面大小推算 UV 偏移。
        const uint8_t* uv_guess = y + (size_t)y_stride * vir_h;
        const uint8_t* p1 = reinterpret_cast<const uint8_t*>(f.pVirAddr[1]);
        const uint8_t* uv = (p1 && p1 >= uv_guess) ? p1 : uv_guess;
        rgb.resize((size_t)w * h * 3);
        nv12_planar_to_rgb(y, uv, y_stride, w, h, rgb.data());
        return true;
    }
    if ((f.enPixelFormat == RK_FMT_YUV420P || f.enPixelFormat == RK_FMT_YUV420P_VU) &&
        f.pVirAddr[0]) {
        const uint8_t* y = reinterpret_cast<const uint8_t*>(f.pVirAddr[0]);
        int uv_stride = (y_stride + 1) / 2;
        int uv_h = (vir_h + 1) / 2;
        // 同上: pVirAddr[1] 可能是 Y 平面首地址, 需校验后再采信
        const uint8_t* uv_guess = y + (size_t)y_stride * vir_h;
        const uint8_t* p1 = reinterpret_cast<const uint8_t*>(f.pVirAddr[1]);
        const uint8_t* first = (p1 && p1 >= uv_guess) ? p1 : uv_guess;
        const uint8_t* second = first + (size_t)uv_stride * uv_h;
        const uint8_t* u = f.enPixelFormat == RK_FMT_YUV420P ? first : second;
        const uint8_t* v = f.enPixelFormat == RK_FMT_YUV420P ? second : first;
        rgb.resize((size_t)w * h * 3);
        yuv420p_to_rgb(y, u, v, y_stride, uv_stride, w, h, rgb.data());
        return true;
    }
    return false;
}

void Nv12Frame::release() {
    if (frame_info_) {
        RK_MPI_VDEC_ReleaseFrame(channel_, frame_info(frame_info_));
        delete frame_info(frame_info_);
        frame_info_ = NULL;
        channel_ = -1;
    }
}

MppDecoder::MppDecoder(int max_w, int max_h) : max_w_(max_w), max_h_(max_h) {}

MppDecoder::~MppDecoder() { deinit(); }

bool MppDecoder::init() {
    if (inited_) return true;

    if (!g_sys_inited) {
        if (RK_MPI_SYS_Init() != RK_SUCCESS) {
            printf("[MPP] RK_MPI_SYS_Init failed\n");
            return false;
        }
        g_sys_inited = true;
    }

    // 输入码流 MB 池
    MB_POOL_CONFIG_S pc;
    memset(&pc, 0, sizeof(pc));
    pc.u64MBSize    = STREAM_MB_SIZE;
    pc.u32MBCnt     = STREAM_MB_CNT;
    pc.enRemapMode  = MB_REMAP_MODE_NOCACHE;
    pc.enAllocType  = MB_ALLOC_TYPE_DMA;
    pc.enDmaType    = MB_DMA_TYPE_CMA;
    pc.bPreAlloc    = RK_TRUE;
    pool_ = RK_MPI_MB_CreatePool(&pc);
    if (pool_ == MB_INVALID_POOLID) {
        printf("[MPP] MB_CreatePool failed\n");
        return false;
    }

    // 创建 VDEC 通道 (H.264, 流模式: 送任意字节块, 硬件自行找帧边界)
    VDEC_CHN_ATTR_S attr;
    memset(&attr, 0, sizeof(attr));
    attr.enMode          = VIDEO_MODE_STREAM;     // 按流送, 无需精确切帧
    attr.enType          = RK_VIDEO_ID_AVC;       // H.264
    attr.u32PicWidth     = max_w_;                // 上限提示(实际以码流为准)
    attr.u32PicHeight    = max_h_;
    attr.u32StreamBufSize = STREAM_MB_SIZE;
    attr.u32StreamBufCnt = STREAM_MB_CNT;
    attr.u32FrameBufCnt  = 6;
    attr.stVdecVideoAttr.u32RefFrameNum = 4;

    if (RK_MPI_VDEC_CreateChn(chn_, &attr) != RK_SUCCESS) {
        printf("[MPP] VDEC_CreateChn failed\n");
        // 必须清零: 调用方随后会调用 deinit(), 否则会对同一 pool 二次
        // DestroyPool, 破坏 MB 子系统 (RV1106 上表现为后续重建失败)。
        RK_MPI_MB_DestroyPool(pool_);
        pool_ = 0;
        return false;
    }
    chn_ok_ = true;

    // 输出像素格式 NV12
    VDEC_CHN_PARAM_S param;
    memset(&param, 0, sizeof(param));
    if (RK_MPI_VDEC_GetChnParam(chn_, &param) == RK_SUCCESS) {
        param.enType = RK_VIDEO_ID_AVC;
        param.stVdecVideoParam.enDecMode = VIDEO_DEC_MODE_IPB;
        param.stVdecVideoParam.enCompressMode = COMPRESS_MODE_NONE;
        param.stVdecVideoParam.enOutputOrder  = VIDEO_OUTPUT_ORDER_DISP;
        RK_MPI_VDEC_SetChnParam(chn_, &param);
    }

    if (RK_MPI_VDEC_StartRecvStream(chn_) != RK_SUCCESS) {
        printf("[MPP] VDEC_StartRecvStream failed\n");
        deinit();
        return false;
    }

    inited_ = true;
    printf("[MPP] RockIt H.264 decoder ready\n");
    return true;
}

bool MppDecoder::send(const uint8_t* data, int len, uint64_t pts, bool end_of_frame) {
    if (!inited_ || len <= 0 || len > STREAM_MB_SIZE) return false;

    MB_BLK blk = RK_MPI_MB_GetMB(pool_, STREAM_MB_SIZE, RK_TRUE);
    if (!blk) return false;

    void* vaddr = RK_MPI_MB_Handle2VirAddr(blk);
    memcpy(vaddr, data, len);

    VDEC_STREAM_S st;
    memset(&st, 0, sizeof(st));
    st.pMbBlk       = blk;
    st.u32Len       = (RK_U32)len;
    st.u64PTS       = pts;
    st.bEndOfStream = RK_FALSE;
    st.bEndOfFrame  = end_of_frame ? RK_TRUE : RK_FALSE;
    st.bBypassMbBlk = RK_FALSE;   // 让 VDEC 内部拷贝, 送完即可释放

    // 非阻塞投递: 阻塞 (-1) 会拖慢主循环读包速率 (曾实测主循环 0.4s/轮,
    // 摄像头 35 包/s 推流时 socket 缓冲涨满, 摄像头静默踢断会话)。
    // 失败时丢该块 (中间帧), 关键帧稍后会再到达。
    RK_S32 ret = RK_MPI_VDEC_SendStream(chn_, &st, 0);
    RK_MPI_MB_ReleaseMB(blk);
    if (ret != RK_SUCCESS) ++send_fail_;
    return ret == RK_SUCCESS;
}

bool MppDecoder::get_rgb(std::vector<uint8_t>& rgb, int& out_w, int& out_h,
                         int timeout_ms, bool convert) {
    Nv12Frame frame;
    if (!get_frame(frame, timeout_ms)) return false;
    out_w = frame.width();
    out_h = frame.height();

    // Decoding continues at camera FPS, but recognition normally runs at
    // only 2-5 FPS. Release skipped frames before doing a CPU YUV->RGB pass.
    if (!convert) {
        return true;
    }
    return frame.to_rgb(rgb);
}

void MppDecoder::dump_status(const char* tag) const {
    VDEC_CHN_STATUS_S st;
    memset(&st, 0, sizeof(st));
    RK_S32 ret = chn_ok_ ? RK_MPI_VDEC_QueryStatus(chn_, &st) : -1;
    if (ret != RK_SUCCESS) {
        printf("[MPP] %s status query failed ret=%d (chn_ok=%d)\n", tag, ret, (int)chn_ok_);
        return;
    }
    printf("[MPP] %s leftBytes=%u leftFrames=%u leftPics=%u recvFrames=%u "
           "decodedFrames=%u startRecv=%d\n",
           tag, st.u32LeftStreamBytes, st.u32LeftStreamFrames, st.u32LeftPics,
           st.u32RecvStreamFrames, st.u32DecodeStreamFrames,
           (int)st.bStartRecvStream);
    // 解码错误分类计数: 哪个非零就指向哪类输入问题
    printf("[MPP] %s decErr fmt=%d size=%d unsprt=%d pack=%d prtcl=%d ref=%d bufsz=%d oversz=%d notrelease=%d\n",
           tag, st.stVdecDecErr.s32FormatErr, st.stVdecDecErr.s32PicSizeErrSet,
           st.stVdecDecErr.s32StreamUnsprt, st.stVdecDecErr.s32PackErr,
           st.stVdecDecErr.s32PrtclNumErrSet, st.stVdecDecErr.s32RefErrSet,
           st.stVdecDecErr.s32PicBufSizeErrSet, st.stVdecDecErr.s32StreamSizeOver,
           st.stVdecDecErr.s32VdecStreamNotRelease);
    printf("[MPP] %s counters: send_fail=%u get_empty=%u\n",
           tag, send_fail_, get_empty_);
}

bool MppDecoder::get_frame(Nv12Frame& frame, int timeout_ms) {
    if (!inited_) return false;
    frame.release();
    VIDEO_FRAME_INFO_S* fi = new VIDEO_FRAME_INFO_S;
    memset(fi, 0, sizeof(*fi));
    if (RK_MPI_VDEC_GetFrame(chn_, fi, timeout_ms) != RK_SUCCESS) {
        delete fi;
        ++get_empty_;
        return false;
    }
    frame.frame_info_ = fi;
    frame.channel_ = chn_;
    return true;
}

void MppDecoder::deinit() {
    if (chn_ok_) {
        RK_MPI_VDEC_StopRecvStream(chn_);
        RK_MPI_VDEC_DestroyChn(chn_);
        chn_ok_ = false;
    }
    if (pool_ && pool_ != MB_INVALID_POOLID) {
        RK_MPI_MB_DestroyPool(pool_);
        pool_ = 0;
    }
    inited_ = false;
}

} // namespace dw
