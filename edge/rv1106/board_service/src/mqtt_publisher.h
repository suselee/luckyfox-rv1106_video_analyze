#pragma once
#include <string>
#include <cstdint>

namespace dw {

// 极简 MQTT 3.1.1 发布端 (publish-only), 无外部依赖, 仅用 POSIX socket。
// 满足本项目 "命中时向 broker 发一条 JSON" 的需求; 不实现订阅。
// 支持 QoS 0 / QoS 1 (等待 PUBACK), 断线时自动重连一次再发。
class MqttPublisher {
public:
    ~MqttPublisher();

    // 连接 broker 并发送 CONNECT。user/pass 为空则不带认证。keepalive 单位秒。
    bool connect(const std::string& host, int port,
                 const std::string& client_id,
                 const std::string& user = "",
                 const std::string& pass = "",
                 int keepalive = 60);

    // 发布一条消息。qos 取 0 或 1。返回是否成功。
    bool publish(const std::string& topic, const std::string& payload, int qos = 1);

    void disconnect();
    bool connected() const { return fd_ >= 0; }

private:
    bool send_all(const uint8_t* buf, size_t len);
    bool recv_all(uint8_t* buf, size_t len);
    bool do_connect();

    int      fd_        = -1;
    uint16_t packet_id_ = 0;

    // 保存连接参数以便断线重连
    std::string host_, client_id_, user_, pass_;
    int port_ = 1883, keepalive_ = 60;
};

} // namespace dw
