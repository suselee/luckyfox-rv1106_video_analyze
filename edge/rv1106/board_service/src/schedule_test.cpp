#include <assert.h>
#include <stdio.h>

#include "schedule.h"

using namespace dw;

static double utc_time(int hour, int minute) {
    return (double)(hour * 3600 + minute * 60);
}

int main() {
    int minute = -1;
    assert(parse_hhmm("07:00", &minute) && minute == 420);
    assert(parse_hhmm("21:59", &minute) && minute == 1319);
    assert(!parse_hhmm("24:00", &minute));
    assert(!parse_hhmm("bad", &minute));

    ActiveSchedule china = {true, 7 * 60, 21 * 60, 8 * 60};
    assert(!schedule_active_at(utc_time(22, 59), china)); // 06:59 UTC+8
    assert(schedule_active_at(utc_time(23, 0), china));   // 07:00 UTC+8
    assert(schedule_active_at(utc_time(12, 59), china));  // 20:59 UTC+8
    assert(!schedule_active_at(utc_time(13, 0), china));  // 21:00 UTC+8

    ActiveSchedule overnight = {true, 21 * 60, 6 * 60, 0};
    assert(schedule_active_at(utc_time(22, 0), overnight));
    assert(schedule_active_at(utc_time(5, 59), overnight));
    assert(!schedule_active_at(utc_time(6, 0), overnight));
    assert(!schedule_active_at(utc_time(12, 0), overnight));

    ActiveSchedule disabled = {false, 0, 0, 0};
    assert(schedule_active_at(utc_time(12, 0), disabled));

    printf("SCHEDULE_TEST_OK\n");
    return 0;
}
