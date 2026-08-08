from nas_video_summarizer.llm import (
    AnalysisResult,
    _parse_daughter_verification,
    _snap_highlight_offsets,
)


def _result(
    *,
    keep=False,
    confidence=0.85,
    title="",
    summary="",
    tags=None,
    local_child_confirmed=False,
    local_child_score=0.0,
):
    return AnalysisResult(
        keep=keep,
        title=title,
        summary=summary,
        tags=tags or [],
        confidence=confidence,
        start_offset_seconds=0,
        end_offset_seconds=1,
        raw={},
        local_child_confirmed=local_child_confirmed,
        local_child_score=local_child_score,
    )


def test_repairs_high_confidence_child_activity_with_false_keep():
    result = _result(
        title="Child playing",
        summary="A young child is engaged in a quiet activity at home.",
        tags=["child", "play"],
        local_child_confirmed=True,
        local_child_score=0.82,
    )

    assert result.keep_consistency_repaired(0.5) is True
    assert result.should_save(0.5) is True


def test_does_not_repair_without_local_child_evidence():
    result = _result(
        title="Child playing",
        summary="A young child is engaged in an activity.",
        tags=["child", "play"],
    )

    assert result.keep_consistency_repaired(0.5) is False
    assert result.should_save(0.5) is False


def test_does_not_repair_scene_without_child_evidence():
    result = _result(
        title="Quiet Study Scene",
        summary="A desk and workspace in an empty room.",
        tags=["desk", "study"],
    )

    assert result.keep_consistency_repaired(0.5) is False
    assert result.should_save(0.5) is False


def test_does_not_repair_explicit_exclusion_or_low_confidence():
    sleeping = _result(
        summary="A young child is sleeping after an activity.",
        tags=["child"],
    )
    uncertain = _result(
        confidence=0.6,
        summary="A young child is playing.",
        tags=["child", "play"],
    )
    idle = _result(
        summary="A young child is not engaged in any activity.",
        tags=["child", "activity"],
    )

    assert sleeping.should_save(0.5) is False
    assert uncertain.should_save(0.5) is False
    assert idle.should_save(0.5) is False


def test_original_true_keep_behavior_is_unchanged():
    assert _result(keep=True, confidence=0.8).should_save(0.5) is True
    assert _result(keep=True, confidence=0.3).should_save(0.5) is False


def test_snaps_model_offsets_to_actual_sample_times():
    offsets = [15.0, 30.0, 50.0, 70.0, 90.0, 110.0]

    assert _snap_highlight_offsets(1, 3, offsets, 120) == (15, 30)
    assert _snap_highlight_offsets(16, 31, offsets, 120) == (15, 30)
    assert _snap_highlight_offsets(10, 120, offsets, 120) == (15, 110)


def test_repairs_verification_boolean_when_description_sees_child():
    verification = _parse_daughter_verification(
        {
            "has_daughter": False,
            "confidence": 0.85,
            "description": "A young girl is visible playing in the room.",
        },
        "raw",
    )

    assert verification.visible is True
    assert verification.repaired is True


def test_verification_negative_description_is_not_repaired():
    verification = _parse_daughter_verification(
        {
            "has_daughter": False,
            "confidence": 0.9,
            "description": "No young girl is visible; the room is empty.",
        },
        "raw",
    )

    assert verification.visible is False
    assert verification.repaired is False
