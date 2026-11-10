#pragma once
#include <stdint.h>
#include <vector>

namespace dw {

// H.264 访问单元(AU)切分器 —— 只输出"完整的 IDR AU"。
//
// 为什么需要它: 摄像头子码流只按 read() 块到达, 块边界与帧边界无关。
// 旧的 keyframes_only 逻辑按"块首 NAL 类型"决定喂/丢, 会把半帧、孤儿 P 帧
// 送进软解 —— 参考链断裂后 OpenH264 实例会楔死 (实测每 2-3 小时一次)。
//
// 这里按 NAL 边界累积, 只在 AU 闭合时判断: 含 IDR(5) 的 AU 整段输出,
// 其余丢弃。这样送进解码器的永远是自包含的合法单元; 若 AU 内缺 SPS/PPS,
// 自动补上最近一次见到的参数集。
//
// 纯逻辑, 无第三方依赖, 可在宿主机单测。
class AuAssembler {
public:
    // 喂入任意字节块; 每切出一个完整 IDR AU 就 push 到 out (Annex-B, 含起始码)。
    void feed(const uint8_t* data, int len, std::vector<std::vector<uint8_t>>& out);

    // 把当前攒着的 AU 强制闭合输出 (流结束/重连前调用)。
    // 正常运行时不需要: 下一个 NAL 到达时会自动闭合。
    void flush(std::vector<std::vector<uint8_t>>& out);

    // 复位缓存 (重连/重建解码器后调用: 旧参数集可能已失效)。
    void reset();

    // 统计
    uint64_t aus_seen() const { return aus_seen_; }
    uint64_t idr_aus() const { return idr_aus_; }

private:
    void close_au(std::vector<std::vector<uint8_t>>& out);
    void append_nals(std::vector<uint8_t>& dst, const std::vector<std::vector<uint8_t>>& nals) const;

    std::vector<uint8_t> pending_;                 // 尚未切出完整 NAL 的尾部字节
    std::vector<std::vector<uint8_t>> au_nals_;    // 当前 AU 的 NAL 列表
    std::vector<uint8_t> last_sps_;
    std::vector<uint8_t> last_pps_;
    bool au_has_vcl_ = false;
    bool au_has_idr_ = false;
    uint64_t aus_seen_ = 0;
    uint64_t idr_aus_ = 0;
};

} // namespace dw
