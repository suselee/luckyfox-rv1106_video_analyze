import json
from dataclasses import replace

from nas_video_summarizer.archive import rebuild_day_archive
from nas_video_summarizer.config import load_settings
from nas_video_summarizer.database import Database
from nas_video_summarizer.daughter_detector import (
    DaughterDetector,
    DaughterObservation,
)


def _observation(offset, confidence=0.9, box=None, people=1):
    return DaughterObservation(
        offset_seconds=offset,
        confidence=confidence,
        boxes=[box or [0.2, 0.2, 0.2, 0.3]],
        person_count=people,
    )


def test_temporal_daughter_observations_become_quiet_candidate():
    settings = replace(
        load_settings("/nonexistent.env"),
        daughter_scan_fps=0.5,
        daughter_event_min_hits=2,
        daughter_event_min_seconds=4,
    )
    detector = DaughterDetector(settings)

    candidates = detector.candidates([
        _observation(2),
        _observation(4),
        _observation(6),
    ])

    assert len(candidates) == 1
    assert candidates[0].analysis_backend == "daughter_detector"
    assert candidates[0].category == "quiet"
    assert candidates[0].start_offset_seconds == 2
    assert candidates[0].end_offset_seconds == 8


def test_multi_person_category_wins_over_activity():
    settings = replace(load_settings("/nonexistent.env"), daughter_scan_fps=0.5)
    detector = DaughterDetector(settings)

    candidates = detector.candidates([
        _observation(0, box=[0.1, 0.1, 0.2, 0.3], people=2),
        _observation(2, box=[0.4, 0.3, 0.2, 0.3], people=2),
    ])

    assert candidates[0].category == "multi_person"


def test_short_or_isolated_detection_is_rejected():
    settings = replace(
        load_settings("/nonexistent.env"),
        daughter_scan_fps=0.5,
        daughter_event_min_hits=2,
        daughter_event_min_seconds=4,
    )
    detector = DaughterDetector(settings)

    assert detector.candidates([_observation(2)]) == []


def test_detector_reset_clears_cached_heuristic_state():
    detector = DaughterDetector(load_settings("/nonexistent.env"))
    detector._cached_child_score = 0.9
    detector._cached_child_until = 10
    detector.reset_segment()
    assert detector._cached_child_score == 0.0
    assert detector._cached_child_until == -1.0


def test_relative_body_size_fallback_selects_smaller_person():
    detector = DaughterDetector(load_settings("/nonexistent.env"))
    adult = [0.1, 0.1, 0.30, 0.70]
    child = [0.6, 0.4, 0.14, 0.35]
    assert detector._relative_child_box([adult, child]) == child


def test_relative_body_size_fallback_rejects_similar_adults():
    detector = DaughterDetector(load_settings("/nonexistent.env"))
    first = [0.1, 0.1, 0.25, 0.60]
    second = [0.5, 0.1, 0.23, 0.56]
    assert detector._relative_child_box([first, second]) is None


def _mock_board_frame_info(monkeypatch, detector, infos):
    values = iter(infos)

    class FakePersonFilter:
        def detect(self, encoded, *, classify_age=True):
            return next(values)

    monkeypatch.setattr(detector, "_get_person_filter", lambda: FakePersonFilter())
    monkeypatch.setattr(detector, "_encoded", lambda path: "encoded")


def _frame_info(
    *, person=0.9, faces=0, child=0.0, adult=0.0, adult_only=False
):
    return {
        "person_score": person,
        "person_boxes": [[0.1, 0.1, 0.2, 0.3]] if person > 0 else [],
        "face_count": faces,
        "child_score": child,
        "adult_score": adult,
        "adult_only": adult_only,
    }


def test_board_probable_accepts_one_child_face(monkeypatch, tmp_path):
    detector = DaughterDetector(load_settings("/nonexistent.env"))
    paths = [tmp_path / f"frame-{index}.jpg" for index in range(3)]
    _mock_board_frame_info(
        monkeypatch,
        detector,
        [
            _frame_info(),
            _frame_info(faces=1, child=0.8),
            _frame_info(),
        ],
    )

    result = detector.verify_board_probable_paths(paths, required_frames=2)

    assert result.accepted is True
    assert result.decision == "child_face"
    assert result.child_face_frames == 1
    assert "child_face" in result.evidence


def test_board_probable_rejects_repeated_adult_faces(monkeypatch, tmp_path):
    detector = DaughterDetector(load_settings("/nonexistent.env"))
    paths = [tmp_path / f"frame-{index}.jpg" for index in range(5)]
    _mock_board_frame_info(
        monkeypatch,
        detector,
        [
            _frame_info(faces=1, adult=0.95, adult_only=True),
            _frame_info(),
            _frame_info(faces=1, adult=0.96, adult_only=True),
            _frame_info(),
            _frame_info(),
        ],
    )

    result = detector.verify_board_probable_paths(paths, required_frames=2)

    assert result.accepted is False
    assert result.decision == "adult_face"
    assert result.adult_face_frames == 2


def test_board_probable_accepts_stable_faceless_person(monkeypatch, tmp_path):
    detector = DaughterDetector(load_settings("/nonexistent.env"))
    paths = [tmp_path / f"frame-{index}.jpg" for index in range(5)]
    _mock_board_frame_info(
        monkeypatch,
        detector,
        [_frame_info(), _frame_info(), _frame_info(), _frame_info(person=0), _frame_info(person=0)],
    )

    result = detector.verify_board_probable_paths(
        paths,
        required_person_frames=3,
        board_score=0.72,
        board_person_score=0.80,
    )

    assert result.accepted is True
    assert result.decision == "faceless_person"
    assert result.person_frames == 3


def test_board_probable_faceless_path_requires_strong_board_score(monkeypatch, tmp_path):
    detector = DaughterDetector(load_settings("/nonexistent.env"))
    paths = [tmp_path / f"frame-{index}.jpg" for index in range(5)]
    _mock_board_frame_info(
        monkeypatch,
        detector,
        [_frame_info() for _ in paths],
    )

    result = detector.verify_board_probable_paths(
        paths,
        required_person_frames=3,
        board_score=0.69,
        board_person_score=0.90,
    )

    assert result.accepted is False
    assert result.decision == "inconclusive"


def test_archive_contract_contains_manifest_and_ready(tmp_path):
    settings = replace(load_settings("/nonexistent.env"), output_dir=tmp_path / "out")
    database = Database(tmp_path / "app.sqlite3")
    database.migrate()
    day_dir = settings.output_dir / "2026-07-17"
    day_dir.mkdir(parents=True)
    clip = day_dir / "090000_daughter-quiet.mp4"
    metadata = clip.with_suffix(".json")
    clip.write_bytes(b"video")
    metadata.write_text("{}", encoding="utf-8")
    database.create_moment(
        camera_name="home-camera",
        title="Daughter quiet activity",
        summary="女儿持续出现在画面中。",
        tags=["daughter", "quiet"],
        confidence=0.9,
        source_low_segment_id=None,
        source_started_at="2026-07-17T09:00:00+08:00",
        source_ended_at="2026-07-17T09:02:00+08:00",
        clip_path=clip,
        metadata_path=metadata,
        analysis_backend="daughter_detector",
        category="quiet",
        selection_score=0.8,
        clip_started_at="2026-07-17T09:00:00+08:00",
        clip_ended_at="2026-07-17T09:00:30+08:00",
    )

    rebuild_day_archive(settings, database, "2026-07-17", ready=True)

    manifest = json.loads((day_dir / "manifest.json").read_text())
    ready = json.loads((day_dir / "_READY.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["clips"][0]["category"] == "quiet"
    assert manifest["clips"][0]["clip_size_bytes"] == 5
    assert ready["manifest_revision"] == manifest["revision"]
    assert "Daughter quiet activity" in (day_dir / "summary.md").read_text()
