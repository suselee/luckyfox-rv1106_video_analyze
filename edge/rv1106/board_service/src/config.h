#pragma once
#include <string>
#include <map>

namespace dw {

// 极简配置解析: key=value, 每行一条, '#' 或 ';' 起始为注释。
// 支持 [section] 分组, 内部键名扁平化为 "section.key"。
// 无 section 时键名即为 "key"。
class Config {
public:
    bool load(const std::string& path);

    bool        has(const std::string& key) const;
    std::string get(const std::string& key, const std::string& def = "") const;
    int         get_int(const std::string& key, int def) const;
    double      get_double(const std::string& key, double def) const;
    bool        get_bool(const std::string& key, bool def) const;

private:
    std::map<std::string, std::string> kv_;
};

} // namespace dw
