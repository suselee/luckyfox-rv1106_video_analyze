#pragma once

#include <vector>

namespace dw {

struct SystemStats {
    double cpu_percent;
    long available_memory_kb;
    double temperature_c;
};

class SystemMonitor {
public:
    SystemMonitor();
    SystemStats sample();
    static double percentile95(const std::vector<double>& values);

private:
    unsigned long long previous_total_;
    unsigned long long previous_idle_;
};

class PerformanceGuard {
public:
    PerformanceGuard(double max_cpu, long min_memory_kb, double max_temperature,
                     double max_detector_p95_ms);
    int update(const SystemStats& stats, double detector_p95_ms);
    int level() const { return level_; }

private:
    double max_cpu_;
    long min_memory_kb_;
    double max_temperature_;
    double max_detector_p95_ms_;
    int level_;
    int good_samples_;
};

} // namespace dw
