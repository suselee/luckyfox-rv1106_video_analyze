from nas_video_summarizer.config import load_env_file, load_settings, with_rtsp_credentials


def test_load_settings_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_ANALYSIS_TEMPERATURE", raising=False)
    monkeypatch.delenv("LLAMA_VERIFICATION_TEMPERATURE", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "APP_PORT=9001",
                "CAMERA_NAME=test-camera",
                "RTSP_USERNAME=alice",
                "RTSP_PASSWORD=s3cret!",
                "RTSP_LOW_URL=rtsp://example/low",
                "RTSP_HIGH_URL=rtsp://example/high",
                "NEXTCLOUD_OUTPUT_DIR=./out",
                "ANALYSIS_IMAGE_MODE=contact_sheet",
                "ANALYSIS_FRAME_WIDTH=320",
                "CONTACT_SHEET_COLUMNS=2",
                "MAX_MOMENT_SECONDS=180",
                "LLAMA_ANALYSIS_TEMPERATURE=0.7",
                "LLAMA_VERIFICATION_TEMPERATURE=0.2",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_file)

    assert settings.app_port == 9001
    assert settings.camera_name == "test-camera"
    assert settings.rtsp_username == "alice"
    assert settings.rtsp_password == "s3cret!"
    assert settings.rtsp_low_url == "rtsp://example/low"
    assert settings.rtsp_high_url == "rtsp://example/high"
    assert settings.rtsp_low_url_for_ffmpeg == "rtsp://alice:s3cret%21@example/low"
    assert settings.output_dir.name == "out"
    assert settings.analysis_image_mode == "contact_sheet"
    assert settings.analysis_frame_width == 320
    assert settings.contact_sheet_columns == 2
    assert settings.max_moment_seconds == 180
    assert settings.llama_analysis_temperature == 0.7
    assert settings.llama_verification_temperature == 0.2
    assert settings.analysis_stream_role == "low"
    assert settings.analysis_window_start == ""
    assert settings.analysis_window_end == ""
    assert settings.record_window_start == ""
    assert settings.record_window_end == ""
    assert settings.ffmpeg_hwaccel == ""
    assert settings.analysis_cooldown_seconds == 5
    assert settings.segment_at_clocktime is True
    assert settings.stream_alignment_tolerance_seconds == 2.0
    assert settings.stream_alignment_sample_count == 5


def test_new_window_and_quota_defaults(monkeypatch):
    monkeypatch.delenv("LLAMA_ANALYSIS_TEMPERATURE", raising=False)
    monkeypatch.delenv("LLAMA_VERIFICATION_TEMPERATURE", raising=False)
    settings = load_settings("/nonexistent.env")
    assert settings.moment_keep_threshold == 0.5
    assert settings.max_moments_per_day == 0
    assert settings.max_moments_per_period == 0
    assert settings.moment_period_boundaries == "07:00,12:00,17:00,21:00"
    assert settings.record_window_start == ""
    assert settings.record_window_end == ""
    assert settings.person_filter_model_dir == settings.data_dir / "person_filter_models"
    assert settings.person_filter_face_threshold == 0.7
    assert settings.person_filter_adult_threshold == 0.9
    assert settings.person_filter_child_threshold == 0.6
    assert settings.llama_analysis_temperature is None
    assert settings.llama_verification_temperature is None
    assert settings.analysis_backend == "vlm"
    assert settings.daughter_detector_mode == "heuristic"
    assert settings.daughter_scan_fps == 0.5
    assert settings.daughter_event_min_hits == 2
    assert settings.daughter_age_check_every == 3
    assert settings.daughter_body_fallback_enabled is True
    assert settings.daughter_body_height_ratio == 0.72
    assert settings.daughter_body_area_ratio == 0.5
    assert settings.moment_category_targets == "active:3,multi_person:3,quiet:2"
    assert settings.mqtt_enabled is False
    assert settings.mqtt_port == 1883
    assert settings.mqtt_daughter_topic == "homecam/daughter/hit"
    assert settings.mqtt_status_topic == "homecam/daughter/status"
    assert settings.rv1106_session_timeout_seconds == 20.0
    assert settings.rv1106_save_wait_seconds == 180.0
    assert settings.rv1106_probable_policy == "accept"
    assert settings.rv1106_accept_probable is True
    assert settings.rv1106_verify_roi_width_scale == 3.0
    assert settings.rv1106_verify_roi_height_scale == 2.0
    assert settings.rv1106_verify_frame_width == 960
    assert settings.rv1106_verify_person_frames == 3
    assert settings.rv1106_verify_board_score == 0.70
    assert settings.rv1106_verify_board_person_score == 0.70
    assert settings.rv1106_verify_faceless_score_multiplier == 0.75


def test_new_probable_policy_overrides_legacy_boolean(tmp_path, monkeypatch):
    monkeypatch.setenv("RV1106_ACCEPT_PROBABLE", "false")
    monkeypatch.setenv("RV1106_PROBABLE_POLICY", "verify")

    settings = load_settings(tmp_path / "missing.env")

    assert settings.rv1106_probable_policy == "verify"
    assert settings.rv1106_accept_probable is True


def test_invalid_probable_policy_is_rejected(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("RV1106_PROBABLE_POLICY", "maybe")
    with pytest.raises(ValueError, match="RV1106_PROBABLE_POLICY"):
        load_settings(tmp_path / "missing.env")


def test_env_overrides_window_and_quota():
    env_file = tmp_path_dummy = None
    import os
    from nas_video_summarizer.config import load_settings

    os.environ["RECORD_WINDOW_START"] = "07:00"
    os.environ["RECORD_WINDOW_END"] = "21:00"
    os.environ["MAX_MOMENTS_PER_DAY"] = "20"
    os.environ["MOMENT_KEEP_THRESHOLD"] = "0.6"
    try:
        settings = load_settings("/nonexistent.env")
        assert settings.record_window_start == "07:00"
        assert settings.record_window_end == "21:00"
        assert settings.max_moments_per_day == 20
        assert settings.moment_keep_threshold == 0.6
    finally:
        for k in ("RECORD_WINDOW_START", "RECORD_WINDOW_END", "MAX_MOMENTS_PER_DAY", "MOMENT_KEEP_THRESHOLD"):
            os.environ.pop(k, None)


def test_with_rtsp_credentials_does_not_override_embedded_credentials():
    url = with_rtsp_credentials(
        "rtsp://camera-user:camera-password@example/stream",
        "other-user",
        "other-password",
    )

    assert url == "rtsp://camera-user:camera-password@example/stream"


def test_with_rtsp_credentials_handles_blank_credentials():
    url = with_rtsp_credentials("rtsp://example/stream", "", "")

    assert url == "rtsp://example/stream"


def test_load_env_file_handles_quoted_values(tmp_path, monkeypatch):
    monkeypatch.delenv("RTSP_PASSWORD", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('RTSP_PASSWORD="abc#123&xyz"\n', encoding="utf-8")

    load_env_file(env_file)

    import os

    assert os.environ["RTSP_PASSWORD"] == "abc#123&xyz"
