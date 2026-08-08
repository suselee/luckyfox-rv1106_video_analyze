from dataclasses import replace

from nas_video_summarizer.cli import (
    _daughter_detector_check,
    _person_filter_check,
    _redact_url,
    build_checks,
)
from nas_video_summarizer.config import load_settings


def test_redact_url_hides_rtsp_credentials():
    value = _redact_url("rtsp://alice:secret@example.local:554/live/low?profile=1")

    assert value == "rtsp://<credentials>@example.local:554/live/low?profile=1"
    assert "alice" not in value
    assert "secret" not in value


def test_redact_url_keeps_url_without_credentials():
    value = _redact_url("http://127.0.0.1:8080/v1")

    assert value == "http://127.0.0.1:8080/v1"


def test_person_filter_check_is_optional_when_disabled():
    check = _person_filter_check(load_settings("/nonexistent.env"))

    assert check.ok is True
    assert check.required is False
    assert check.detail == "disabled"


def test_person_filter_check_explains_freebsd_venv_fix(monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"), person_filter_enabled=True
    )

    def fail_prepare(self):
        raise ImportError("No module named 'cv2'")

    monkeypatch.setattr(
        "nas_video_summarizer.cli.PersonFilter.prepare", fail_prepare
    )

    check = _person_filter_check(settings)

    assert check.ok is False
    assert check.required is True
    assert "--system-site-packages" in check.detail


def test_rv1106_backend_skips_nas_person_filter(monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        analysis_backend="rv1106",
        person_filter_enabled=True,
    )

    check = _person_filter_check(settings)
    backend = next(item for item in build_checks(settings) if item.name == "analysis backend")

    assert check.ok is True
    assert check.detail == "inactive in rv1106 mode"
    assert backend.ok is True


def test_high_only_mode_does_not_require_low_rtsp():
    settings = replace(
        load_settings("/nonexistent.env"),
        analysis_enabled=False,
        analysis_stream_role="high",
        rtsp_low_url="",
        rtsp_high_url="rtsp://camera/high",
    )

    low = next(item for item in build_checks(settings) if item.name == "low RTSP stream")

    assert low.ok is False
    assert low.required is False


def test_reject_policy_with_analysis_disabled_does_not_load_daughter_detector(
    monkeypatch,
):
    settings = replace(
        load_settings("/nonexistent.env"),
        analysis_enabled=False,
        analysis_backend="daughter_detector",
        rv1106_probable_policy="reject",
    )

    def fail_prepare(self):
        raise AssertionError("inactive detector should not be prepared")

    monkeypatch.setattr(
        "nas_video_summarizer.cli.DaughterDetector.prepare", fail_prepare
    )

    check = _daughter_detector_check(settings)

    assert check.ok is True
    assert check.required is False
    assert check.detail == "inactive"
