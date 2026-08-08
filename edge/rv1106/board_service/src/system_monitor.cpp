#include "system_monitor.h"

#include <algorithm>
#include <stdio.h>
#include <string.h>

namespace dw {

SystemMonitor::SystemMonitor() : previous_total_(0), previous_idle_(0) {}

SystemStats SystemMonitor::sample() {
    SystemStats stats = {};
    FILE* file = fopen("/proc/stat", "r");
    if (file) {
        unsigned long long user = 0, nice = 0, system = 0, idle = 0;
        unsigned long long iowait = 0, irq = 0, softirq = 0, steal = 0;
        if (fscanf(file, "cpu %llu %llu %llu %llu %llu %llu %llu %llu",
                   &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal) == 8) {
            unsigned long long idle_all = idle + iowait;
            unsigned long long total = user + nice + system + idle + iowait + irq + softirq + steal;
            if (previous_total_ && total > previous_total_)
                stats.cpu_percent = 100.0 * (1.0 -
                    (double)(idle_all - previous_idle_) / (double)(total - previous_total_));
            previous_total_ = total;
            previous_idle_ = idle_all;
        }
        fclose(file);
    }

    file = fopen("/proc/meminfo", "r");
    if (file) {
        char key[64];
        long value = 0;
        char unit[16];
        while (fscanf(file, "%63s %ld %15s", key, &value, unit) == 3) {
            if (strcmp(key, "MemAvailable:") == 0) {
                stats.available_memory_kb = value;
                break;
            }
        }
        fclose(file);
    }

    file = fopen("/sys/class/thermal/thermal_zone0/temp", "r");
    if (file) {
        double value = 0;
        if (fscanf(file, "%lf", &value) == 1)
            stats.temperature_c = value > 1000.0 ? value / 1000.0 : value;
        fclose(file);
    }
    return stats;
}

double SystemMonitor::percentile95(const std::vector<double>& values) {
    if (values.empty()) return 0;
    std::vector<double> sorted(values);
    std::sort(sorted.begin(), sorted.end());
    size_t index = (size_t)((sorted.size() - 1) * 0.95);
    return sorted[index];
}

PerformanceGuard::PerformanceGuard(double max_cpu, long min_memory_kb,
                                   double max_temperature, double max_detector_p95_ms)
    : max_cpu_(max_cpu), min_memory_kb_(min_memory_kb),
      max_temperature_(max_temperature), max_detector_p95_ms_(max_detector_p95_ms),
      level_(0), good_samples_(0) {}

int PerformanceGuard::update(const SystemStats& stats, double detector_p95_ms) {
    bool bad = (stats.cpu_percent > 0 && stats.cpu_percent > max_cpu_) ||
               (stats.available_memory_kb > 0 && stats.available_memory_kb < min_memory_kb_) ||
               (stats.temperature_c > 0 && stats.temperature_c >= max_temperature_) ||
               (detector_p95_ms > max_detector_p95_ms_);
    if (bad) {
        good_samples_ = 0;
        if (level_ < 2) level_++;
    } else if (level_ > 0) {
        if (++good_samples_ >= 10) {
            level_--;
            good_samples_ = 0;
        }
    }
    return level_;
}

} // namespace dw
