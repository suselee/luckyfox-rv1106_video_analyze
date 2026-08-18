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

    // multipart 文件字段 (流式发送用, 数据零拷贝引用)。
    struct MultipartFile {
        std::string name;
        std::string filename;
        const uint8_t* data;
        size_t len;
        MultipartFile(const std::string& n, const std::string& fn,
                      const uint8_t* d, size_t l)
            : name(n), filename(fn), data(d), len(l) {}
    };

    // 解析 http://host[:port]/path
    static bool parse_url(const std::string& url, std::string& host,
                          int& port, std::string& path);

    // 组装 multipart/form-data 请求体:
    //   parts: 文本字段 (如 meta=json)
    //   file_bytes/file_name: 一个文件字段 (video)
    //   audio_bytes/audio_len/audio_name: 可选的第二个文件字段 (audio);
    //     为 NULL/0 时只发视频。
    static std::vector<uint8_t> build_multipart(
        const std::string& boundary,
        const std::vector<std::pair<std::string, std::string>>& parts,
        const uint8_t* file_bytes, size_t file_len, const std::string& file_name,
        const uint8_t* audio_bytes = NULL, size_t audio_len = 0,
        const std::string& audio_name = "");

    // 流式 multipart POST: 文本字段 + 文件字段, 数据零拷贝直接分片发送。
    // 避免为超大 body 再分配一份完整拷贝 (省峰值内存, 4K 剪辑可达 20MB+)。
    // 内部先发送 HTTP 头 (含 Content-Length), 再逐片发送数据。
    bool post_multipart(const std::string& url, const std::string& boundary,
                        const std::vector<std::pair<std::string, std::string>>& fields,
                        const std::vector<MultipartFile>& files,
                        double timeout_sec, Response& resp);

    // 阻塞式 POST; timeout_sec 为总超时 (连接+发送+接收)。
    bool post(const std::string& url, const std::vector<uint8_t>& body,
              double timeout_sec, Response& resp);
};

} // namespace dw