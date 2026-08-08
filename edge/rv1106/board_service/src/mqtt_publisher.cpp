#include "mqtt_publisher.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <unistd.h>
#include <cstring>
#include <string>
#include <vector>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace dw {
namespace {

void put_u16(std::vector<uint8_t>& b, uint16_t v) {
    b.push_back(static_cast<uint8_t>(v >> 8));
    b.push_back(static_cast<uint8_t>(v & 0xFF));
}

void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put_u16(b, static_cast<uint16_t>(s.size()));
    b.insert(b.end(), s.begin(), s.end());
}

// MQTT "remaining length" 变长编码 (1~4 字节)
void put_remlen(std::vector<uint8_t>& b, size_t len) {
    do {
        uint8_t d = static_cast<uint8_t>(len % 128);
        len /= 128;
        if (len > 0) d |= 0x80;
        b.push_back(d);
    } while (len > 0);
}

} // namespace

MqttPublisher::~MqttPublisher() { disconnect(); }

bool MqttPublisher::send_all(const uint8_t* buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = ::send(fd_, buf + off, len - off, MSG_NOSIGNAL);
        if (n <= 0) return false;
        off += static_cast<size_t>(n);
    }
    return true;
}

bool MqttPublisher::recv_all(uint8_t* buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = ::recv(fd_, buf + off, len - off, 0);
        if (n <= 0) return false;
        off += static_cast<size_t>(n);
    }
    return true;
}

bool MqttPublisher::do_connect() {
    // 1) 解析域名 + 建立 TCP 连接
    struct addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo* res = nullptr;
    std::string portstr = std::to_string(port_);
    if (::getaddrinfo(host_.c_str(), portstr.c_str(), &hints, &res) != 0 || !res)
        return false;

    int fd = -1;
    for (struct addrinfo* p = res; p; p = p->ai_next) {
        fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(res);
    if (fd < 0) return false;

    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    fd_ = fd;

    // 2) 组装并发送 CONNECT
    std::vector<uint8_t> vh; // 可变头 + payload
    put_str(vh, "MQTT");
    vh.push_back(0x04); // protocol level 4 == MQTT 3.1.1
    uint8_t flags = 0x02; // clean session
    if (!user_.empty()) flags |= 0x80;
    if (!pass_.empty()) flags |= 0x40;
    vh.push_back(flags);
    put_u16(vh, static_cast<uint16_t>(keepalive_));
    put_str(vh, client_id_);
    if (!user_.empty()) put_str(vh, user_);
    if (!pass_.empty()) put_str(vh, pass_);

    std::vector<uint8_t> pkt;
    pkt.push_back(0x10); // CONNECT
    put_remlen(pkt, vh.size());
    pkt.insert(pkt.end(), vh.begin(), vh.end());
    if (!send_all(pkt.data(), pkt.size())) { disconnect(); return false; }

    // 3) 读 CONNACK: [0x20][0x02][flags][rc]; rc==0 表示接受
    uint8_t ack[4];
    if (!recv_all(ack, 4)) { disconnect(); return false; }
    if (ack[0] != 0x20 || ack[3] != 0x00) { disconnect(); return false; }
    return true;
}

bool MqttPublisher::connect(const std::string& host, int port,
                            const std::string& client_id,
                            const std::string& user, const std::string& pass,
                            int keepalive) {
    disconnect();
    host_      = host;
    port_      = port;
    client_id_ = client_id;
    user_      = user;
    pass_      = pass;
    keepalive_ = keepalive;
    return do_connect();
}

bool MqttPublisher::publish(const std::string& topic, const std::string& payload, int qos) {
    if (qos != 0 && qos != 1) qos = 1;

    for (int attempt = 0; attempt < 2; ++attempt) {
        if (fd_ < 0 && !do_connect()) return false;

        uint16_t pid = 0;
        std::vector<uint8_t> vh;
        put_str(vh, topic);
        if (qos == 1) {
            pid = ++packet_id_;
            if (pid == 0) pid = ++packet_id_; // packet id 不能为 0
            put_u16(vh, pid);
        }
        vh.insert(vh.end(), payload.begin(), payload.end());

        std::vector<uint8_t> pkt;
        pkt.push_back(static_cast<uint8_t>(0x30 | ((qos & 1) << 1))); // PUBLISH, DUP=0 RETAIN=0
        put_remlen(pkt, vh.size());
        pkt.insert(pkt.end(), vh.begin(), vh.end());

        if (!send_all(pkt.data(), pkt.size())) { disconnect(); continue; }

        if (qos == 1) {
            // 读 PUBACK: [0x40][0x02][pid_hi][pid_lo]
            uint8_t ack[4];
            if (!recv_all(ack, 4)) { disconnect(); continue; }
            uint16_t apid = (static_cast<uint16_t>(ack[2]) << 8) | ack[3];
            if (ack[0] != 0x40 || apid != pid) { disconnect(); continue; }
        }
        return true;
    }
    return false;
}

void MqttPublisher::disconnect() {
    if (fd_ >= 0) {
        uint8_t d[2] = {0xE0, 0x00}; // DISCONNECT, best-effort
        ::send(fd_, d, 2, MSG_NOSIGNAL);
        ::close(fd_);
        fd_ = -1;
    }
}

} // namespace dw
