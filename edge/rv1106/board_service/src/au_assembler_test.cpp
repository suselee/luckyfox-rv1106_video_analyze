// AuAssembler 宿主机回归测试 (无板端依赖)。
//
// 编译: g++ -std=c++11 -Wall -Isrc src/au_assembler_test.cpp src/au_assembler.cpp -o /tmp/au_test

#include <assert.h>
#include <stdio.h>
#include <vector>

#include "au_assembler.h"

using namespace dw;

static std::vector<uint8_t> nal(int type, int payload)
{
    // 4 字节起始码 + header + payload
    std::vector<uint8_t> v = {0, 0, 0, 1, (uint8_t)(0x60 | type)};
    for (int i = 0; i < payload; ++i) v.push_back((uint8_t)(i & 0xff));
    return v;
}

static void append(std::vector<uint8_t>& dst, const std::vector<uint8_t>& src)
{
    dst.insert(dst.end(), src.begin(), src.end());
}

int main()
{
    // 1) SPS+PPS+IDR -> 一个 AU
    {
        AuAssembler a;
        std::vector<std::vector<uint8_t>> out;
        std::vector<uint8_t> stream;
        append(stream, nal(7, 8));
        append(stream, nal(8, 4));
        append(stream, nal(5, 32));
        a.feed(stream.data(), (int)stream.size(), out);
        a.flush(out);
        assert(out.size() == 1);
        assert(a.idr_aus() == 1);
        printf("case1 ok: %zu AU\n", out.size());
    }

    // 2) SPS+PPS+IDR 之后跟若干 P 帧 -> 只有第一个 IDR AU
    {
        AuAssembler a;
        std::vector<std::vector<uint8_t>> out;
        std::vector<uint8_t> stream;
        append(stream, nal(7, 8));
        append(stream, nal(8, 4));
        append(stream, nal(5, 32));
        for (int i = 0; i < 5; ++i) append(stream, nal(1, 16));
        a.feed(stream.data(), (int)stream.size(), out);
        a.flush(out);
        assert(out.size() == 1);
        printf("case2 ok: %zu AU (P 帧被丢弃)\n", out.size());
    }

    // 3) 同一码流按任意字节边界切块喂入 -> 结果一致
    {
        AuAssembler a;
        std::vector<std::vector<uint8_t>> out;
        std::vector<uint8_t> stream;
        append(stream, nal(7, 8));
        append(stream, nal(8, 4));
        append(stream, nal(5, 32));
        append(stream, nal(1, 16));
        append(stream, nal(7, 8));
        append(stream, nal(8, 4));
        append(stream, nal(5, 32));
        size_t pos = 0;
        const size_t chunk = 7;   // 故意不对齐
        while (pos < stream.size()) {
            size_t n = std::min(chunk, stream.size() - pos);
            a.feed(stream.data() + pos, (int)n, out);
            pos += n;
        }
        a.flush(out);
        assert(out.size() == 2);
        printf("case3 ok: %zu AU (跨块切分正确)\n", out.size());
    }

    // 4) 第二个 AU 缺少 SPS/PPS -> 自动补上
    {
        AuAssembler a;
        std::vector<std::vector<uint8_t>> out;
        std::vector<uint8_t> first;
        append(first, nal(7, 8));
        append(first, nal(8, 4));
        append(first, nal(5, 32));
        a.feed(first.data(), (int)first.size(), out);
        std::vector<uint8_t> second;
        append(second, nal(1, 16));
        append(second, nal(5, 32));   // 只有 IDR, 没有参数集
        a.feed(second.data(), (int)second.size(), out);
        a.flush(out);
        assert(out.size() == 2);
        // 第二个 AU 应以 SPS(7) 开头 -> 检查 header 类型 (4 字节起始码 -> 索引 4)
        assert((out[1][4] & 0x1f) == 7);
        printf("case4 ok: 参数集自动补齐\n");
    }

    printf("AU_ASSEMBLER_TEST_OK\n");
    return 0;
}
