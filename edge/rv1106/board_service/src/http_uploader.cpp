#include "http_uploader.h"

#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace dw {

static const char* BOUNDARY_DEFAULT = "dwclip-boundary";

static bool http_connect(const std::string& url, double timeout_sec, int& fd) {
    std::string host, path;
    int port = 80;
    if (!HttpUploader::parse_url(url, host, port, path)) return false;

    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);
    if (getaddrinfo(host.c_str(), port_str, &hints, &res) != 0 || !res)
        return false;

    fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) {
        freeaddrinfo(res);
        return false;
    }
    struct timeval tv;
    tv.tv_sec = (time_t)timeout_sec;
    tv.tv_usec = (suseconds_t)((timeout_sec - tv.tv_sec) * 1e6);
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    bool ok = connect(fd, res->ai_addr, res->ai_addrlen) == 0;
    freeaddrinfo(res);
    if (!ok) {
        close(fd);
        fd = -1;
    }
    return ok;
}

static bool http_send_all(int fd, const void* buf, size_t len) {
    const uint8_t* p = (const uint8_t*)buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) return false;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return true;
}

static bool http_read_response(int fd, double timeout_sec, HttpUploader::Response& resp) {
    resp = HttpUploader::Response();
    std::string header;
    char tmp[1024];
    bool got_headers = false;
    while (header.size() < 64 * 1024) {
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) break;
        header.append(tmp, (size_t)n);
        if (header.find("\r\n\r\n") != std::string::npos) {
            got_headers = true;
            break;
        }
    }
    if (!got_headers) return false;
    resp.status = atoi(header.c_str() + 9);  // "HTTP/1.1 200 ..."
    size_t body_pos = header.find("\r\n\r\n") + 4;
    resp.body = header.substr(body_pos);
    return resp.status >= 200 && resp.status < 300;
}

bool HttpUploader::post_multipart(
    const std::string& url, const std::string& boundary,
    const std::vector<std::pair<std::string, std::string>>& fields,
    const std::vector<MultipartFile>& files,
    double timeout_sec, Response& resp) {
    std::string host, path;
    int port = 80;
    if (!parse_url(url, host, port, path)) {
        printf("[UPLOAD] bad url: %s\n", url.c_str());
        return false;
    }

    // 预计算 Content-Length: 所有字段头 + 字段体 + 文件头 + 文件体 + 尾边界
    size_t total = 0;
    std::vector<std::string> field_heads;
    field_heads.reserve(fields.size());
    for (size_t i = 0; i < fields.size(); ++i) {
        std::string h;
        h.reserve(fields[i].second.size() + 128);
        h += "--" + boundary + "\r\n";
        h += "Content-Disposition: form-data; name=\"" + fields[i].first + "\"\r\n";
        h += "Content-Type: application/json\r\n\r\n";
        h += fields[i].second;
        h += "\r\n";
        field_heads.push_back(h);
        total += h.size();
    }
    std::vector<std::string> file_heads;
    file_heads.reserve(files.size());
    for (size_t i = 0; i < files.size(); ++i) {
        std::string h;
        h += "--" + boundary + "\r\n";
        h += "Content-Disposition: form-data; name=\"" + files[i].name +
             "\"; filename=\"" + files[i].filename + "\"\r\n";
        h += "Content-Type: application/octet-stream\r\n\r\n";
        file_heads.push_back(h);
        total += h.size() + files[i].len + 2;  // + \r\n
    }
    std::string tail = "--" + boundary + "--\r\n";  // 文件数据后已发 \r\n
    total += tail.size();

    int fd = -1;
    if (!http_connect(url, timeout_sec, fd)) {
        printf("[UPLOAD] cannot connect %s\n", url.c_str());
        return false;
    }

    std::string req;
    req += "POST " + path + " HTTP/1.1\r\n";
    req += "Host: " + host + ":" + std::to_string(port) + "\r\n";
    req += "Content-Type: multipart/form-data; boundary=" + boundary + "\r\n";
    req += "Content-Length: " + std::to_string(total) + "\r\n";
    req += "Connection: close\r\n\r\n";

    bool ok = http_send_all(fd, req.data(), req.size());
    for (size_t i = 0; ok && i < field_heads.size(); ++i)
        ok = http_send_all(fd, field_heads[i].data(), field_heads[i].size());
    for (size_t i = 0; ok && i < files.size(); ++i) {
        ok = http_send_all(fd, file_heads[i].data(), file_heads[i].size());
        if (ok && files[i].data && files[i].len > 0)
            ok = http_send_all(fd, files[i].data, files[i].len);
        if (ok) ok = http_send_all(fd, "\r\n", 2);
    }
    if (ok) ok = http_send_all(fd, tail.data(), tail.size());

    if (ok) ok = http_read_response(fd, timeout_sec, resp);
    close(fd);
    return ok;
}

bool HttpUploader::parse_url(const std::string& url, std::string& host,
                             int& port, std::string& path) {
    host.clear();
    port = 80;
    path = "/";
    if (url.compare(0, 7, "http://") != 0) return false;
    const char* p = url.c_str() + 7;
    const char* slash = strchr(p, '/');
    std::string authority = slash ? std::string(p, slash - p) : std::string(p);
    size_t colon = authority.find(':');
    if (colon != std::string::npos) {
        host = authority.substr(0, colon);
        port = atoi(authority.c_str() + (long)colon + 1);
        if (port <= 0 || port > 65535) return false;
    } else {
        host = authority;
    }
    if (slash) path = slash;
    if (host.empty()) return false;
    return true;
}

std::vector<uint8_t> HttpUploader::build_multipart(
    const std::string& boundary,
    const std::vector<std::pair<std::string, std::string>>& parts,
    const uint8_t* file_bytes, size_t file_len, const std::string& file_name,
    const uint8_t* audio_bytes, size_t audio_len, const std::string& audio_name) {
    std::vector<uint8_t> body;
    std::string head;
    for (size_t i = 0; i < parts.size(); ++i) {
        head.clear();
        head += "--" + boundary + "\r\n";
        head += "Content-Disposition: form-data; name=\"" + parts[i].first + "\"\r\n";
        head += "Content-Type: application/json\r\n\r\n";
        head += parts[i].second;
        head += "\r\n";
        body.insert(body.end(), head.begin(), head.end());
    }
    head.clear();
    head += "--" + boundary + "\r\n";
    head += "Content-Disposition: form-data; name=\"video\"; filename=\"";
    head += file_name + "\"\r\n";
    head += "Content-Type: application/octet-stream\r\n\r\n";
    body.insert(body.end(), head.begin(), head.end());
    body.insert(body.end(), file_bytes, file_bytes + file_len);
    head.clear();
    head += "\r\n";
    if (audio_bytes && audio_len > 0 && !audio_name.empty()) {
        head += "--" + boundary + "\r\n";
        head += "Content-Disposition: form-data; name=\"audio\"; filename=\"";
        head += audio_name + "\"\r\n";
        head += "Content-Type: application/octet-stream\r\n\r\n";
        body.insert(body.end(), head.begin(), head.end());
        body.insert(body.end(), audio_bytes, audio_bytes + audio_len);
        head.clear();
        head += "\r\n";
    }
    head += "--" + boundary + "--\r\n";
    body.insert(body.end(), head.begin(), head.end());
    return body;
}

bool HttpUploader::post(const std::string& url, const std::vector<uint8_t>& body,
                        double timeout_sec, Response& resp) {
    std::string host, path;
    int port = 80;
    if (!parse_url(url, host, port, path)) {
        printf("[UPLOAD] bad url: %s\n", url.c_str());
        return false;
    }

    int fd = -1;
    if (!http_connect(url, timeout_sec, fd)) {
        printf("[UPLOAD] cannot connect %s\n", url.c_str());
        return false;
    }

    std::string req;
    req += "POST " + path + " HTTP/1.1\r\n";
    req += "Host: " + host + ":" + std::to_string(port) + "\r\n";
    req += "Content-Type: multipart/form-data; boundary=";
    req += BOUNDARY_DEFAULT + std::string("\r\n");
    req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    req += "Connection: close\r\n\r\n";

    bool ok = http_send_all(fd, req.data(), req.size()) &&
              http_send_all(fd, body.data(), body.size());
    if (ok) ok = http_read_response(fd, timeout_sec, resp);
    close(fd);
    return ok;
}

} // namespace dw