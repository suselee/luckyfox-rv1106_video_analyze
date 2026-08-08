#include "track_fusion.h"

#include <algorithm>
#include <math.h>
#include <stdio.h>

namespace dw {

TrackFusion::TrackFusion(const FusionConfig& config)
    : config_(config), people_count_(0), confirmed_sessions_(0),
      session_sequence_(0), next_track_id_(0) {}

float TrackFusion::iou(const IvaObject& a, const IvaObject& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    float total = area_a + area_b - intersection;
    return total > 0 ? intersection / total : 0;
}

void TrackFusion::observe(double now, const IvaResult& detections) {
    people_count_ = (int)detections.people.size();
    float tallest = 0;
    for (size_t i = 0; i < detections.people.size(); ++i)
        tallest = std::max(tallest, detections.people[i].y2 - detections.people[i].y1);

    std::vector<uint32_t> matched_ids;
    for (size_t i = 0; i < detections.people.size(); ++i) {
        const IvaObject& person = detections.people[i];
        Track* selected = NULL;
        float best_match = -1e9f;
        for (std::map<uint32_t, Track>::iterator candidate = tracks_.begin();
             candidate != tracks_.end(); ++candidate) {
            Track& track = candidate->second;
            if (std::find(matched_ids.begin(), matched_ids.end(), track.id) != matched_ids.end())
                continue;
            if (now - track.last_seen > config_.track_lost_seconds) continue;
            float overlap = iou(track.box, person);
            float old_cx = (track.box.x1 + track.box.x2) * 0.5f;
            float old_cy = (track.box.y1 + track.box.y2) * 0.5f;
            float new_cx = (person.x1 + person.x2) * 0.5f;
            float new_cy = (person.y1 + person.y2) * 0.5f;
            float distance = sqrtf((old_cx - new_cx) * (old_cx - new_cx) +
                                   (old_cy - new_cy) * (old_cy - new_cy));
            float old_h = std::max(0.01f, track.box.y2 - track.box.y1);
            float new_h = std::max(0.01f, person.y2 - person.y1);
            float size_ratio = new_h / old_h;
            bool same_vendor_id = track.source_id == person.id;
            bool spatial_match = overlap >= 0.12f ||
                (distance <= 0.20f && size_ratio >= 0.50f && size_ratio <= 2.0f);
            if (!same_vendor_id && !spatial_match) continue;
            float match = (same_vendor_id ? 2.0f : 0.0f) + overlap - distance * 0.5f;
            if (match > best_match) {
                best_match = match;
                selected = &track;
            }
        }
        if (!selected) {
            Track track = {};
            track.id = ++next_track_id_;
            track.source_id = person.id;
            track.box = person;
            track.best_box = person;
            track.first_seen = track.last_seen = now;
            track.last_face_check = -1e9;
            track.last_confirmed = -1e9;
            track.last_publish = -1e9;
            track.last_child_like = -1e9;
            track.first_face_hit = -1e9;
            track.best_timestamp = now;
            track.person_score = person.score;
            track.previous_cx = (person.x1 + person.x2) * 0.5f;
            track.previous_cy = (person.y1 + person.y2) * 0.5f;
            track.observations = 1;
            track.identity = IDENTITY_UNKNOWN;
            tracks_[track.id] = track;
            selected = &tracks_[track.id];
        } else {
            Track& track = *selected;
            float cx = (person.x1 + person.x2) * 0.5f;
            float cy = (person.y1 + person.y2) * 0.5f;
            float movement = sqrtf((cx - track.previous_cx) * (cx - track.previous_cx) +
                                   (cy - track.previous_cy) * (cy - track.previous_cy));
            track.activity_score = std::min(1.0f,
                track.activity_score * 0.8f + std::min(1.0f, movement * 8.0f) * 0.2f);
            track.previous_cx = cx;
            track.previous_cy = cy;
            track.box = person;
            track.source_id = person.id;
            track.last_seen = now;
            track.person_score = std::max(track.person_score, person.score);
            track.observations++;
        }
        Track& track = *selected;
        matched_ids.push_back(track.id);
        float height = person.y2 - person.y1;
        bool absolute_child = height >= 0.12f && height <= config_.child_max_height_ratio;
        bool relative_child = detections.people.size() > 1 && tallest > 0 &&
                              height <= tallest * config_.relative_child_height_ratio;
        track.child_like = relative_child ||
                           (detections.people.size() == 1 && absolute_child);
        if (track.child_like) track.last_child_like = now;
        track.ambiguous = false;
    }

    for (size_t i = 0; i < matched_ids.size(); ++i) {
        for (size_t j = i + 1; j < matched_ids.size(); ++j) {
            if (iou(tracks_[matched_ids[i]].box, tracks_[matched_ids[j]].box) >= 0.35f) {
                tracks_[matched_ids[i]].ambiguous = true;
                tracks_[matched_ids[j]].ambiguous = true;
                tracks_[matched_ids[i]].needs_revalidation = true;
                tracks_[matched_ids[j]].needs_revalidation = true;
            }
        }
    }

    for (std::map<uint32_t, Track>::iterator it = tracks_.begin(); it != tracks_.end(); ++it)
        update_identity(it->second, now);
}

std::vector<TrackSnapshot> TrackFusion::snapshot(double now) const {
    std::vector<TrackSnapshot> out;
    for (std::map<uint32_t, Track>::const_iterator it = tracks_.begin();
         it != tracks_.end(); ++it) {
        const Track& track = it->second;
        if (now - track.last_seen >= config_.track_lost_seconds) continue;
        TrackSnapshot snap;
        snap.id = track.id;
        snap.box = track.box;
        snap.last_seen = track.last_seen;
        snap.last_face_check = track.last_face_check;
        snap.child_like = track.child_like;
        snap.ambiguous = track.ambiguous;
        snap.identity = track.identity;
        out.push_back(snap);
    }
    return out;
}

uint32_t TrackFusion::track_for_face(float cx, float cy) const {
    uint32_t best = 0;
    float best_area = 1e9f;
    for (std::map<uint32_t, Track>::const_iterator it = tracks_.begin(); it != tracks_.end(); ++it) {
        const Track& track = it->second;
        // Faces live in the upper part of a person box; the relaxed 0.75
        // factor also covers a child held at chest height in a merged box.
        float upper_bottom = track.box.y1 + (track.box.y2 - track.box.y1) * 0.75f;
        if (cx >= track.box.x1 && cx <= track.box.x2 &&
            cy >= track.box.y1 && cy <= upper_bottom) {
            float area = (track.box.x2 - track.box.x1) * (track.box.y2 - track.box.y1);
            if (area < best_area) {
                best_area = area;
                best = track.id;
            }
        }
    }
    return best;
}

bool TrackFusion::should_check_face(uint32_t track_id, double now) const {
    std::map<uint32_t, Track>::const_iterator it = tracks_.find(track_id);
    if (it == tracks_.end()) return false;
    return now - it->second.last_face_check >= config_.face_check_interval_seconds;
}

void TrackFusion::mark_face_checked(uint32_t track_id, double now) {
    std::map<uint32_t, Track>::iterator it = tracks_.find(track_id);
    if (it != tracks_.end()) it->second.last_face_check = now;
}

void TrackFusion::apply_face_score(uint32_t track_id, float similarity, double now) {
    std::map<uint32_t, Track>::iterator it = tracks_.find(track_id);
    if (it == tracks_.end()) return;
    Track& track = it->second;
    track.face_score = std::max(track.face_score, similarity);
    if (similarity >= config_.face_high_threshold) {
        track.face_hits = 2;
        track.first_face_hit = now;
    } else if (similarity >= config_.face_threshold) {
        if (track.face_hits == 0 ||
            now - track.first_face_hit > config_.face_hit_window_seconds) {
            track.face_hits = 1;
            track.first_face_hit = now;
        } else {
            track.face_hits++;
        }
    }
    if (track.face_hits >= 2) {
        if (track.identity != IDENTITY_CONFIRMED) confirmed_sessions_++;
        track.last_confirmed = now;
        track.identity = IDENTITY_CONFIRMED;
        track.needs_revalidation = false;
    }
    update_identity(track, now);
}

void TrackFusion::update_identity(Track& track, double now) {
    if (now - track.first_face_hit > config_.face_hit_window_seconds &&
        now - track.last_confirmed > config_.confirmed_ttl_seconds) {
        track.face_hits = 0;
        track.first_face_hit = -1e9;
    }
    if (!track.needs_revalidation &&
        now - track.last_confirmed <= config_.confirmed_ttl_seconds) {
        track.identity = IDENTITY_CONFIRMED;
    } else if (!track.ambiguous &&
               (track.child_like ||
                now - track.last_child_like <= config_.probable_hold_seconds) &&
               track.observations >= config_.probable_min_observations &&
               now - track.first_seen >= config_.probable_min_seconds) {
        track.identity = IDENTITY_PROBABLE;
    } else {
        track.identity = IDENTITY_UNKNOWN;
    }
    float identity_score = track.identity == IDENTITY_CONFIRMED
        ? std::max(track.face_score, config_.face_threshold)
        : (track.identity == IDENTITY_PROBABLE
            ? std::min(0.75f, 0.50f + 0.02f * track.observations)
            : 0.0f);
    float selection = identity_score * 0.75f + track.activity_score * 0.25f;
    if (selection >= track.best_selection) {
        track.best_selection = selection;
        track.best_timestamp = now;
        track.best_box = track.box;
    }
}

const char* TrackFusion::identity_name(IdentityLevel identity) {
    if (identity == IDENTITY_CONFIRMED) return "confirmed";
    if (identity == IDENTITY_PROBABLE) return "probable";
    return "unknown";
}

FusionEvent TrackFusion::make_event(const Track& track, const char* event, double now) const {
    FusionEvent out;
    out.event = event;
    IdentityLevel reported =
        event[0] == 'e' && track.identity == IDENTITY_UNKNOWN
            ? track.published_identity
            : track.identity;
    out.identity = identity_name(reported);
    out.session_id = track.session_id;
    out.track_id = track.id;
    out.timestamp = now;
    out.session_start = track.session_start;
    out.best_timestamp = track.best_timestamp;
    out.face_score = track.face_score;
    out.person_score = track.person_score;
    out.activity_score = track.activity_score;
    out.score = reported == IDENTITY_CONFIRMED
        ? std::max(track.face_score, config_.face_threshold)
        : std::min(0.75f, 0.50f + 0.02f * track.observations);
    out.box = track.box;
    out.best_box = track.best_box;
    out.people_count = people_count_;
    return out;
}

std::vector<FusionEvent> TrackFusion::collect_events(double now) {
    std::vector<FusionEvent> events;
    std::vector<uint32_t> erase_ids;
    for (std::map<uint32_t, Track>::iterator it = tracks_.begin(); it != tracks_.end(); ++it) {
        Track& track = it->second;
        bool missing = now - track.last_seen >= config_.track_lost_seconds;
        bool eligible = track.identity != IDENTITY_UNKNOWN && !missing;
        if (eligible && !track.session_active) {
            char session[64];
            snprintf(session, sizeof(session), "%u-%llu", track.id,
                     (unsigned long long)++session_sequence_);
            track.session_id = session;
            track.session_start = now;
            track.last_publish = now;
            track.session_active = true;
            track.published_identity = track.identity;
            events.push_back(make_event(track, "start", now));
        } else if (eligible && track.session_active &&
                   (track.identity != track.published_identity ||
                    now - track.last_publish >= config_.mqtt_update_seconds)) {
            track.last_publish = now;
            track.published_identity = track.identity;
            events.push_back(make_event(track, "update", now));
        } else if ((!eligible || missing) && track.session_active) {
            events.push_back(make_event(track, "end", now));
            track.session_active = false;
            track.session_id.clear();
        }
        if (missing && now - track.last_seen >= config_.track_lost_seconds + 10.0)
            erase_ids.push_back(track.id);
    }
    for (size_t i = 0; i < erase_ids.size(); ++i) tracks_.erase(erase_ids[i]);
    return events;
}

std::vector<FusionEvent> TrackFusion::finish_sessions(double now) {
    std::vector<FusionEvent> events;
    for (std::map<uint32_t, Track>::iterator it = tracks_.begin();
         it != tracks_.end(); ++it) {
        Track& track = it->second;
        if (track.session_active)
            events.push_back(make_event(track, "end", now));
    }
    tracks_.clear();
    people_count_ = 0;
    return events;
}

int TrackFusion::active_tracks() const { return (int)tracks_.size(); }

int TrackFusion::confirmed_tracks() const {
    int count = 0;
    for (std::map<uint32_t, Track>::const_iterator it = tracks_.begin(); it != tracks_.end(); ++it)
        if (it->second.identity == IDENTITY_CONFIRMED) count++;
    return count;
}

int TrackFusion::probable_tracks() const {
    int count = 0;
    for (std::map<uint32_t, Track>::const_iterator it = tracks_.begin(); it != tracks_.end(); ++it)
        if (it->second.identity == IDENTITY_PROBABLE) count++;
    return count;
}

} // namespace dw
