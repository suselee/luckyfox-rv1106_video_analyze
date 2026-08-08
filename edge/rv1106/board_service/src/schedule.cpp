#include "schedule.h"

#include <cstdlib>

namespace dw {

bool parse_hhmm(const std::string& value, int* minute) {
    if (!minute) return false;
    size_t colon = value.find(':');
    if (colon == std::string::npos || value.find(':', colon + 1) != std::string::npos)
        return false;
    std::string hour_text = value.substr(0, colon);
    std::string minute_text = value.substr(colon + 1);
    if (hour_text.empty() || minute_text.empty()) return false;
    for (size_t i = 0; i < hour_text.size(); ++i)
        if (hour_text[i] < '0' || hour_text[i] > '9') return false;
    for (size_t i = 0; i < minute_text.size(); ++i)
        if (minute_text[i] < '0' || minute_text[i] > '9') return false;
    int hour = std::atoi(hour_text.c_str());
    int value_minute = std::atoi(minute_text.c_str());
    if (hour < 0 || hour > 23 || value_minute < 0 || value_minute > 59)
        return false;
    *minute = hour * 60 + value_minute;
    return true;
}

bool schedule_active_at(double epoch_seconds, const ActiveSchedule& schedule) {
    if (!schedule.enabled) return true;
    long long adjusted = (long long)epoch_seconds +
                         (long long)schedule.utc_offset_minutes * 60LL;
    long long second_of_day = adjusted % 86400LL;
    if (second_of_day < 0) second_of_day += 86400LL;
    int current_minute = (int)(second_of_day / 60LL);
    if (schedule.start_minute == schedule.end_minute) return true;
    if (schedule.start_minute < schedule.end_minute)
        return current_minute >= schedule.start_minute &&
               current_minute < schedule.end_minute;
    return current_minute >= schedule.start_minute ||
           current_minute < schedule.end_minute;
}

} // namespace dw
