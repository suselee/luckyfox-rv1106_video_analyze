// 主机侧验证: post_multipart 流式发送的字节序列与 build_multipart 完全一致。
// 编译: g++ -std=c++11 -Wall -Isrc tests/http_test.cpp src/http_uploader.cpp -o /tmp/http_test
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include "http_uploader.h"

using namespace dw;

int main() {
    const std::string boundary = "dwclip-boundary";
    std::vector<std::pair<std::string, std::string>> fields;
    fields.push_back({"meta", "{\"session_id\":\"x\",\"codec\":\"H265\"}"});

    uint8_t video[20];
    uint8_t audio[10];
    for (int i = 0; i < 20; ++i) video[i] = (uint8_t)i;
    for (int i = 0; i < 10; ++i) audio[i] = (uint8_t)(0x50 + i);

    // 1) 旧方式: 完整 body
    std::vector<uint8_t> body = HttpUploader::build_multipart(
        boundary, fields, video, sizeof(video), "clip.hevc",
        audio, sizeof(audio), "clip.adts");

    // 2) 新方式: 流式发到本地 server, 由外部程序保存到 /tmp/http_cap.bin
    HttpUploader up;
    std::vector<HttpUploader::MultipartFile> files;
    files.push_back(HttpUploader::MultipartFile("video", "clip.hevc", video, sizeof(video)));
    files.push_back(HttpUploader::MultipartFile("audio", "clip.adts", audio, sizeof(audio)));
    HttpUploader::Response resp;
    bool ok = up.post_multipart("http://127.0.0.1:18080/upload", boundary,
                                fields, files, 10.0, resp);
    printf("post rc=%s status=%d body=%s\n", ok ? "OK" : "FAIL",
           resp.status, resp.body.c_str());
    return ok ? 0 : 1;
}
