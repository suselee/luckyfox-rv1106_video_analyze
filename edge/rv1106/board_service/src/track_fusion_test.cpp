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
    config.mqtt_update_seconds = 5;
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

    fusion.observe(6, IvaResult());
    events = fusion.collect_events(9);
    assert(events.size() == 1 && events[0].event == "end");
    assert(events[0].identity == "confirmed");

    printf("TRACK_FUSION_TEST_OK\n");
    return 0;
}
