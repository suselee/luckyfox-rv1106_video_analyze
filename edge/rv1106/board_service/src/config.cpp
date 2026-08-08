#include "config.h"
#include <cstdio>
#include <cstdlib>
#include <cctype>

namespace dw {
namespace {

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

} // namespace

bool Config::load(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "r");
    if (!fp) return false;

    kv_.clear();
    std::string section;
    char line[1024];
    while (std::fgets(line, sizeof(line), fp)) {
        std::string s = trim(line);
        if (s.empty() || s[0] == '#' || s[0] == ';') continue;

        if (s.front() == '[' && s.back() == ']') {
            section = trim(s.substr(1, s.size() - 2));
            continue;
        }

        size_t eq = s.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(s.substr(0, eq));
        std::string val = trim(s.substr(eq + 1));
        if (key.empty()) continue;
        if (!section.empty()) key = section + "." + key;
        kv_[key] = val;
    }
    std::fclose(fp);
    return true;
}

bool Config::has(const std::string& key) const {
    return kv_.find(key) != kv_.end();
}

std::string Config::get(const std::string& key, const std::string& def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : it->second;
}

int Config::get_int(const std::string& key, int def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : std::atoi(it->second.c_str());
}

double Config::get_double(const std::string& key, double def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : std::atof(it->second.c_str());
}

bool Config::get_bool(const std::string& key, bool def) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return def;
    const std::string& v = it->second;
    return v == "1" || v == "true" || v == "yes" || v == "on" ||
           v == "True" || v == "TRUE";
}

} // namespace dw
