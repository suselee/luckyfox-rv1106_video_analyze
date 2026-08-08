#pragma once

#include <string>

namespace dw {

struct ActiveSchedule {
    bool enabled;
    int start_minute;
    int end_minute;
    int utc_offset_minutes;
};

bool parse_hhmm(const std::string& value, int* minute);
bool schedule_active_at(double epoch_seconds, const ActiveSchedule& schedule);

} // namespace dw
