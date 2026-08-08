#pragma once

#include <stdint.h>
#include <map>
#include <string>
#include <vector>

#include "rockiva_detector.h"

namespace dw {

enum IdentityLevel {
    IDENTITY_UNKNOWN = 0,
    IDENTITY_PROBABLE = 1,
    IDENTITY_CONFIRMED = 2,
};

struct FusionConfig {
    double probable_min_seconds;
    int probable_min_observations;
    double child_max_height_ratio;
    double relative_child_height_ratio;
    double face_check_interval_seconds;
    double face_hit_window_seconds;
    double confirmed_ttl_seconds;
    double track_lost_seconds;
    double probable_hold_seconds;
    double mqtt_update_seconds;
    float face_threshold;
    float face_high_threshold;
};

struct FusionEvent {
    std::string event;
    std::string identity;
    std::string session_id;
    uint32_t track_id;
    double timestamp;
    double session_start;
    double best_timestamp;
    float score;
    float face_score;
    float person_score;
    float activity_score;
    IvaObject box;
    IvaObject best_box;
    int people_count;
};

// Read-only view of one live track, used by the main loop to schedule
// face-recognition ROIs without exposing the internal Track storage.
struct TrackSnapshot {
    uint32_t id;
    IvaObject box;
    double last_seen;
    double last_face_check;
    bool child_like;
    bool ambiguous;
    IdentityLevel identity;
};

class TrackFusion {
public:
    explicit TrackFusion(const FusionConfig& config);

    void observe(double now, const IvaResult& detections);
    std::vector<TrackSnapshot> snapshot(double now) const;
    uint32_t track_for_face(float cx, float cy) const;
    bool should_check_face(uint32_t track_id, double now) const;
    void mark_face_checked(uint32_t track_id, double now);
    // The caller guarantees the similarity is attributed to the correct track
    // (e.g. via a RockIVA face box anchored inside the track box), so the
    // score is applied even while the track overlaps another person.
    void apply_face_score(uint32_t track_id, float similarity, double now);
    std::vector<FusionEvent> collect_events(double now);
    // End all published sessions and clear live tracks when the active camera
    // window closes. Cumulative confirmation/session counters are preserved.
    std::vector<FusionEvent> finish_sessions(double now);
    int active_tracks() const;
    int confirmed_tracks() const;
    int probable_tracks() const;
    // Cumulative number of times any track reached CONFIRMED identity.
    int confirmed_sessions() const { return confirmed_sessions_; }

private:
    struct Track {
        uint32_t id;
        uint32_t source_id;
        IvaObject box;
        IvaObject best_box;
        double first_seen;
        double last_seen;
        double last_face_check;
        double last_confirmed;
        double last_publish;
        double last_child_like;
        double session_start;
        double best_timestamp;
        float face_score;
        float person_score;
        float activity_score;
        float best_selection;
        float previous_cx;
        float previous_cy;
        int observations;
        int face_hits;
        double first_face_hit;
        bool ambiguous;
        bool needs_revalidation;
        bool child_like;
        bool session_active;
        IdentityLevel identity;
        IdentityLevel published_identity;
        std::string session_id;
    };

    static float iou(const IvaObject& a, const IvaObject& b);
    static const char* identity_name(IdentityLevel identity);
    void update_identity(Track& track, double now);
    FusionEvent make_event(const Track& track, const char* event, double now) const;

    FusionConfig config_;
    std::map<uint32_t, Track> tracks_;
    int people_count_;
    int confirmed_sessions_;
    uint64_t session_sequence_;
    uint32_t next_track_id_;
};

} // namespace dw
