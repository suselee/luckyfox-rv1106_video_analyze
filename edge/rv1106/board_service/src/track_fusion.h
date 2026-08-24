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
    double confirm_child_hold_seconds;
    double mqtt_update_seconds;
    // 无脸儿童通道: child_like 轨迹在无露脸证据时, 需要活动量至少达到
    // 该阈值才升 probable (0.0=关闭活动量要求)。区分"坐着的成人"与
    // "活动中/走动中的儿童"。
    float probable_min_activity;
    // 成人先验否决: 同一轨迹连续 >= 该次数观测到身高超过
    // child_max_height_ratio (站立/行走体态), 则判定为成人先验并粘滞,
    // 之后该轨迹不得再经无脸几何通道升级 probable (坐下后体型近似儿童,
    // 活动量瞬时达标即锁存是本次误报根因)。仅 >= face_high_threshold 的
    // 强人脸确认不受影响。0=禁用。
    int adult_tall_observations;
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
    // 精彩度选峰: 活动量 EMA 峰值时刻 / 最后一次活跃时刻 (秒, 进程时钟)。
    double activity_peak_ts;
    double last_active_ts;
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
        // 成人先验状态: 连续 tall_streak 次身高超阈值后 adult_prior 粘滞。
        int tall_streak;
        bool adult_prior;
        // 精彩度选峰: 活动 EMA 峰值与最后活跃时刻 (Q1/Q2)。
        float activity_peak;
        double activity_peak_ts;
        double last_active_ts;
        // 最近一次满足 probable 证据 (人脸命中或 child_like 且活动量达标)
        // 的时刻; probable 在该时刻的 probable_hold_seconds 内保持粘滞,
        // 容忍短暂分类抖动 (wobble) 而不掉级、不断会话。
        double last_probable_evidence;
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
