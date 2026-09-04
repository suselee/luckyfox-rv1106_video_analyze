#include <assert.h>
#include <stdio.h>

#include "track_fusion.h"

using namespace dw;

static FusionConfig test_config() {
    FusionConfig config = {};
    config.probable_min_seconds = 4;
    config.probable_min_observations = 3;
    config.child_max_height_ratio = 0.55;
    config.relative_child_height_ratio = 0.75;
    config.face_check_interval_seconds = 2;
    config.face_hit_window_seconds = 3;
    config.confirmed_ttl_seconds = 8;
    config.track_lost_seconds = 3;
    config.probable_hold_seconds = 3;
    config.confirm_child_hold_seconds = 60;
    config.mqtt_update_seconds = 5;
    config.probable_min_activity = 0.0f;  // 测试关闭活动量门槛
    config.adult_tall_observations = 2;
    config.face_threshold = 0.35f;
    config.face_high_threshold = 0.55f;
    return config;
}

static IvaResult one_person(uint32_t id, float x, float height) {
    IvaResult result;
    IvaObject person = {};
    person.id = id;
    person.score = 0.9f;
    person.x1 = x;
    person.x2 = x + 0.2f;
    person.y1 = 0.4f;
    person.y2 = person.y1 + height;
    result.people.push_back(person);
    return result;
}

int main() {
    FusionConfig config = test_config();
    TrackFusion fusion(config);

    fusion.observe(0, one_person(7, 0.1f, 0.35f));
    std::vector<TrackSnapshot> snaps = fusion.snapshot(0);
    assert(snaps.size() == 1 && snaps[0].child_like);
    assert(fusion.should_check_face(snaps[0].id, 0));
    fusion.mark_face_checked(snaps[0].id, 0);
    assert(!fusion.should_check_face(snaps[0].id, 1));
    assert(fusion.should_check_face(snaps[0].id, 2));
    fusion.observe(1, one_person(17, 0.11f, 0.35f));
    fusion.observe(2, one_person(27, 0.12f, 0.35f));
    fusion.observe(4, one_person(37, 0.13f, 0.35f));
    std::vector<FusionEvent> events = fusion.collect_events(4);
    assert(events.size() == 1 && events[0].event == "start");
    assert(events[0].identity == "probable");
    assert(events[0].best_box.x1 == events[0].box.x1);
    uint32_t logical_track = events[0].track_id;

    // A brief child-size classification wobble keeps the existing probable
    // session alive, then ends it once the configured hold expires.
    TrackFusion held(config);
    held.observe(0, one_person(1, 0.1f, 0.35f));
    held.observe(1, one_person(1, 0.11f, 0.35f));
    held.observe(2, one_person(1, 0.12f, 0.35f));
    held.observe(4, one_person(1, 0.13f, 0.35f));
    events = held.collect_events(4);
    assert(events.size() == 1 && events[0].event == "start");
    held.observe(5, one_person(1, 0.13f, 0.70f));
    assert(held.collect_events(5).empty());
    held.observe(7.5, one_person(1, 0.13f, 0.70f));
    events = held.collect_events(7.5);
    assert(events.size() == 1 && events[0].event == "end");

    fusion.apply_face_score(logical_track, 0.60f, 5);
    events = fusion.collect_events(5);
    assert(events.size() == 1 && events[0].event == "update");
    assert(events[0].identity == "confirmed");
    assert(fusion.confirmed_sessions() == 1);

    TrackFusion repeated(config);
    repeated.observe(0, one_person(1, 0.1f, 0.35f));
    repeated.apply_face_score(1, 0.40f, 1);
    assert(repeated.collect_events(1).empty());
    repeated.apply_face_score(1, 0.40f, 2);
    events = repeated.collect_events(2);
    assert(events.size() == 1 && events[0].identity == "confirmed");
    assert(repeated.confirmed_sessions() == 1);

    // Two hits outside the configured 3s window must not confirm...
    TrackFusion expired(config);
    expired.observe(0, one_person(1, 0.1f, 0.35f));
    expired.apply_face_score(1, 0.40f, 1);
    expired.apply_face_score(1, 0.40f, 5);
    assert(expired.collect_events(5).empty());
    assert(expired.confirmed_sessions() == 0);

    // ...but the same spacing confirms with a wider 6s window (track kept
    // alive by continued observations).
    FusionConfig wide = test_config();
    wide.face_hit_window_seconds = 6;
    TrackFusion relaxed(wide);
    relaxed.observe(0, one_person(1, 0.1f, 0.35f));
    relaxed.apply_face_score(1, 0.40f, 1);
    relaxed.observe(2, one_person(1, 0.11f, 0.35f));
    relaxed.observe(4, one_person(1, 0.12f, 0.35f));
    relaxed.apply_face_score(1, 0.40f, 5);
    events = relaxed.collect_events(5);
    assert(events.size() == 1 && events[0].identity == "confirmed");

    // Face scores apply even while the track is ambiguous (overlapping):
    // RockIVA face anchoring disambiguates attribution for the caller.
    TrackFusion overlap(config);
    IvaResult pair;
    {
        IvaObject big = {};
        big.id = 1; big.score = 0.9f;
        big.x1 = 0.1f; big.x2 = 0.5f; big.y1 = 0.1f; big.y2 = 0.95f;
        IvaObject small = {};
        small.id = 2; small.score = 0.9f;
        small.x1 = 0.15f; small.x2 = 0.45f; small.y1 = 0.25f; small.y2 = 0.85f;
        pair.people.push_back(big);
        pair.people.push_back(small);
    }
    overlap.observe(0, pair);
    snaps = overlap.snapshot(0);
    assert(snaps.size() == 2);
    uint32_t child_track = 0;
    for (size_t i = 0; i < snaps.size(); ++i) {
        assert(snaps[i].ambiguous);
        if (snaps[i].child_like) child_track = snaps[i].id;
    }
    assert(child_track != 0);
    overlap.apply_face_score(child_track, 0.60f, 1);
    events = overlap.collect_events(1);
    assert(events.size() == 1 && events[0].identity == "confirmed");
    assert(events[0].track_id == child_track);

    // Face-anchor matching prefers the smallest containing box and accepts
    // faces down to 75% of the person height (held child).
    assert(overlap.track_for_face(0.3f, 0.62f) == child_track);

    events = overlap.finish_sessions(2);
    assert(events.size() == 1 && events[0].event == "end");
    assert(overlap.active_tracks() == 0);
    assert(overlap.confirmed_sessions() == 1);

    // 成人轨迹 (身高比超过 child_max) 低阈值命中两次不得确认: 避免保存
    // 只有大人的画面。
    TrackFusion adult(config);
    adult.observe(0, one_person(1, 0.1f, 0.75f));
    adult.apply_face_score(1, 0.40f, 1);
    adult.apply_face_score(1, 0.40f, 2);
    adult.observe(2.5, one_person(1, 0.1f, 0.75f));
    assert(adult.collect_events(2.8).empty());
    assert(adult.confirmed_sessions() == 0);
    // 成人轨迹只有强相似度 (>= face_high_threshold) 才能确认。
    adult.apply_face_score(1, 0.60f, 3);
    adult.observe(3.5, one_person(1, 0.1f, 0.75f));
    events = adult.collect_events(4);
    assert(events.size() == 1 && events[0].event == "start");
    assert(events[0].identity == "confirmed");

    // 儿童轨迹在确认后身份衰减, end 事件仍携带 confirmed (确认过的会话
    // 必须保存, 不能因 ttl 衰减丢失)。
    TrackFusion decay(config);
    decay.observe(0, one_person(1, 0.1f, 0.35f));
    decay.apply_face_score(1, 0.60f, 1);
    events = decay.collect_events(1);
    assert(events.size() == 1 && events[0].identity == "confirmed");
    decay.observe(20, one_person(1, 0.1f, 0.35f));
    decay.observe(22, IvaResult());  // track lost shortly after
    events = decay.collect_events(30);
    assert(events.size() == 1 && events[0].event == "end");
    assert(events[0].identity == "confirmed");

    fusion.observe(6, IvaResult());
    events = fusion.collect_events(9);
    assert(events.size() == 1 && events[0].event == "end");
    assert(events[0].identity == "confirmed");

    // 8/24 误报回归: 成人走入画面 (站立身高连续可见) 后坐下翻找。
    // 坐下瞬间身高跌破 child_max 而活动量尚未衰减, 旧逻辑会单帧锁存
    // probable 并保存纯成人片段 (face_score=0)。adult_prior 否决后
    // 不得升级、不得开会话。
    TrackFusion sitdown(config);
    sitdown.observe(0, one_person(1, 0.10f, 0.80f));  // 站立: tall_streak=1
    sitdown.observe(1, one_person(1, 0.10f, 0.80f));  // 站立: streak=2 -> 成人先验
    sitdown.observe(2, one_person(1, 0.14f, 0.35f));  // 坐下: child_like, 活动中
    sitdown.observe(3, one_person(1, 0.22f, 0.35f));
    sitdown.observe(4, one_person(1, 0.30f, 0.35f));
    sitdown.observe(5, one_person(1, 0.38f, 0.35f));  // age=5>=4, obs=6>=3
    assert(sitdown.collect_events(5).empty());
    assert(sitdown.probable_tracks() == 0);
    // 先验粘滞: 持续活动也保持 UNKNOWN。
    sitdown.observe(7, one_person(1, 0.46f, 0.35f));
    sitdown.observe(9, one_person(1, 0.54f, 0.35f));
    assert(sitdown.collect_events(9).empty());
    assert(sitdown.probable_tracks() == 0);
    // 强人脸确认 (>= high_threshold) 穿透成人先验。
    sitdown.apply_face_score(1, 0.60f, 10);
    sitdown.observe(10, one_person(1, 0.54f, 0.35f));
    events = sitdown.collect_events(11);
    assert(events.size() == 1 && events[0].identity == "confirmed");

    // wobble 保活回归: probable 会话在 child_like 短暂丢失 (儿童站起/
    // 被抱高) 的 probable_hold_seconds 内不得掉级断会话。
    TrackFusion wobblerecover(config);
    wobblerecover.observe(0, one_person(1, 0.1f, 0.35f));
    wobblerecover.observe(1, one_person(1, 0.11f, 0.35f));
    wobblerecover.observe(2, one_person(1, 0.12f, 0.35f));
    wobblerecover.observe(4, one_person(1, 0.13f, 0.35f));
    events = wobblerecover.collect_events(4);
    assert(events.size() == 1 && events[0].event == "start");
    assert(events[0].identity == "probable");
    wobblerecover.observe(5, one_person(1, 0.13f, 0.70f));   // 抖动帧 (身高超阈)
    assert(wobblerecover.collect_events(5).empty());          // 会话保持
    wobblerecover.observe(6, one_person(1, 0.13f, 0.35f));   // 回到儿童体型
    wobblerecover.observe(7, one_person(1, 0.13f, 0.35f));
    events = wobblerecover.collect_events(8);
    for (size_t i = 0; i < events.size(); ++i)
        assert(events[i].event != "end");                     // 未断会话

    // Q1/Q2 数据回归: 峰值时刻锁定活动最高的观测, 最后活跃时刻随活跃
    // 观测刷新且静止后不再前进。
    FusionConfig acfg = test_config();
    acfg.probable_min_activity = 0.30f;
    TrackFusion act(acfg);
    // 静止 -> 大幅移动 3 帧 (EMA 升至峰) -> 静止 2 帧 (先仍活跃后衰减)
    act.observe(0, one_person(1, 0.10f, 0.35f));   // 新轨迹 act=0
    act.observe(1, one_person(1, 0.40f, 0.35f));   // act≈0.20
    act.observe(2, one_person(1, 0.70f, 0.35f));   // act≈0.36 活跃
    act.observe(3, one_person(1, 0.55f, 0.35f));   // act≈0.49 <- 峰值
    act.observe(4, one_person(1, 0.56f, 0.35f));   // act≈0.41 活跃
    act.observe(5, one_person(1, 0.57f, 0.35f));   // act≈0.34 活跃
    events = act.collect_events(5.5);              // 打开会话
    assert(events.size() == 1 && events[0].event == "start");
    act.observe(6, one_person(1, 0.57f, 0.35f));   // act≈0.29 静止衰减
    events = act.collect_events(9);
    assert(events.size() == 1 && events[0].event == "end");
    double peak_ts = events[0].activity_peak_ts;
    double last_active = events[0].last_active_ts;
    assert(peak_ts == 3.0);                        // 锁定 EMA 最高帧
    assert(last_active == 5.0);                    // t=6 已跌破阈值不刷新

    printf("TRACK_FUSION_TEST_OK\n");
    return 0;
}
