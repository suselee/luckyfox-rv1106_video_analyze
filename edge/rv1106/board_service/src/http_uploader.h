#pragma once
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace dw {

// 手写 HTTP/1.1 客户端 (仅 POSIX socket), 用于把板端切片推送到 NAS。
// 与 mqtt_publisher 同风格: 无外部库, 连接即用。
class HttpUploader {
public:
    struct Response {
        int status = 0;         // 0 = 未收到响应 (网络/超时错误)
        std::string body;
    };

    // 解析 http://host[:port]/path
    static bool parse_url(const std::string& url, std::string& host,
                          int& port, std::string& path);

    // 组装 multipart/form-data 请求体:
    //   parts: 文本字段 (如 meta=json)
    //   file_bytes/file_name: 一个文件字段 (video)
    static std::vector<uint8_t> build_multipart(
        const std::string& boundary,
        const std::vector<std::pair<std::string, std::string>>& parts,
        const uint8_t* file_bytes, size_t file_len, const std::string& file_name);

    // 阻塞式 POST; timeout_sec 为总超时 (连接+发送+接收)。
    bool post(const std::string& url, const std::vector<uint8_t>& body,
              double timeout_sec, Response& resp);
};

} // namespace dw