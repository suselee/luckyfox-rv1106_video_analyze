// 硬解通路验证工具: 读一个 H.264 裸流(Annex-B)文件 -> RockIt VDEC 硬解 ->
// 取首帧 -> 存成 JPG。用来在板子上确认 MPP 硬件解码 + NV12->RGB 通路是否 OK,
// 之后再在其上搭 RTSP 客户端。
//
// 制作测试裸流(在 PC 上, 从已有视频或摄像头):
//   ffmpeg -i input.mp4 -c:v copy -bsf:v h264_mp4toannexb -an -t 3 test.h264
//   或从摄像头子码流抓几秒: ffmpeg -rtsp_transport tcp -i rtsp://... -c:v copy -bsf:v h264_mp4toannexb -t 3 test.h264
//
// 用法: ./decode_test test.h264 out.jpg

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "stb/stb_image_write.h"
#include "mpp_decoder.h"

using namespace dw;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <in.h264 (annex-b)> <out.jpg>\n", argv[0]);
        return 1;
    }

    // 读整个裸流文件
    FILE* fp = fopen(argv[1], "rb");
    if (!fp) { printf("cannot open %s\n", argv[1]); return 1; }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if (fread(buf.data(), 1, sz, fp) != (size_t)sz) { fclose(fp); printf("read fail\n"); return 1; }
    fclose(fp);
    printf("loaded %ld bytes\n", sz);

    MppDecoder dec;
    if (!dec.init()) { printf("decoder init failed\n"); return 1; }

    std::vector<uint8_t> rgb;
    int w = 0, h = 0;
    bool got = false;

    // 按 256KB 分块喂 (流模式下可任意切分)
    const int CHUNK = 256 * 1024;
    for (long off = 0; off < sz && !got; off += CHUNK) {
        int len = (int)((sz - off) < CHUNK ? (sz - off) : CHUNK);
        dec.send(buf.data() + off, len, (uint64_t)off, true);
        if (dec.get_rgb(rgb, w, h, 30)) got = true;   // 有帧就停
    }
    // 喂完后再多取几次(解码器有缓冲延迟)
    for (int i = 0; i < 30 && !got; i++)
        if (dec.get_rgb(rgb, w, h, 100)) got = true;

    if (!got) { printf("no frame decoded\n"); dec.deinit(); return 1; }

    printf("decoded frame %dx%d, saving %s\n", w, h, argv[2]);
    int ok = stbi_write_jpg(argv[2], w, h, 3, rgb.data(), 90);
    dec.deinit();
    printf(ok ? "OK\n" : "save failed\n");
    return ok ? 0 : 1;
}
