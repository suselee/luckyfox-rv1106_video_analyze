import asyncio
from dataclasses import replace

from nas_video_summarizer.config import load_settings
from nas_video_summarizer.llm import AnalysisResult, _extract_json
from nas_video_summarizer.llm import LlamaAnalyzer


def _result(keep: bool, confidence: float) -> AnalysisResult:
    return AnalysisResult(
        keep=keep,
        title="t",
        summary="s",
        tags=[],
        confidence=confidence,
        start_offset_seconds=0,
        end_offset_seconds=1,
        raw={},
    )


async def _to_thread_inline(function, *args):
    return function(*args)


def test_extract_json_from_plain_json():
    data = _extract_json('{"keep": true, "confidence": 0.9}')

    assert data["keep"] is True
    assert data["confidence"] == 0.9


def test_extract_json_from_wrapped_text():
    data = _extract_json('Here is the result:\n{"keep": false, "tags": ["quiet"]}\nDone.')

    assert data["keep"] is False
    assert data["tags"] == ["quiet"]


def test_should_save_requires_both_keep_and_confidence():
    # A weak "keep=true" guess must NOT save when confidence is low -
    # this is the regression guard for the static/empty-room problem:
    # 2B models on low-res contact sheets often emit keep=true for
    # rooms that contain toys but no visible child.
    assert not _result(keep=True, confidence=0.3).should_save(0.55)
    assert _result(keep=True, confidence=0.6).should_save(0.55)
    # Even very high confidence does not override an explicit keep=false.
    assert not _result(keep=False, confidence=0.99).should_save(0.55)


def test_analyze_prompt_uses_actual_motion_offsets(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        analysis_image_mode="frames",
        llama_analysis_temperature=None,
    )
    frame_paths = []
    for index in range(3):
        frame_path = tmp_path / f"frame_{index}.jpg"
        frame_path.write_bytes(b"jpeg")
        frame_paths.append(frame_path)

    captured: dict[str, object] = {}

    def fake_post_json(endpoint, headers, payload, timeout):
        captured["instructions"] = payload["messages"][1]["content"][0]["text"]
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"keep": false, "title": "quiet", "summary": "none", '
                            '"tags": [], "confidence": 0.1, '
                            '"start_offset_seconds": 0, "end_offset_seconds": 1}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr("nas_video_summarizer.llm._post_json", fake_post_json)
    monkeypatch.setattr(
        "nas_video_summarizer.llm.asyncio.to_thread", _to_thread_inline
    )

    asyncio.run(
        LlamaAnalyzer(settings).analyze(
            video_path=tmp_path / "segment.mp4",
            image_paths=frame_paths,
            duration_seconds=120,
            frame_offsets_seconds=[12.4, 70.6, 97.2],
        )
    )

    instructions = captured["instructions"]
    assert "frame #1: ~12.4s" in instructions
    assert "frame #2: ~70.6s" in instructions
    assert "frame #3: ~97.2s" in instructions
    assert "frame #2: ~60s" not in instructions
    assert "temperature" not in captured["payload"]


def test_analyze_sends_configured_temperature(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"), llama_analysis_temperature=0.7
    )
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"jpeg")
    captured = {}

    def fake_post_json(endpoint, headers, payload, timeout):
        captured["temperature"] = payload.get("temperature")
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"keep": false, "title": "quiet", "summary": "none", '
                            '"tags": [], "confidence": 0.1, '
                            '"start_offset_seconds": 0, "end_offset_seconds": 1}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr("nas_video_summarizer.llm._post_json", fake_post_json)
    monkeypatch.setattr(
        "nas_video_summarizer.llm.asyncio.to_thread", _to_thread_inline
    )

    asyncio.run(
        LlamaAnalyzer(settings).analyze(
            video_path=tmp_path / "segment.mp4",
            image_paths=[frame_path],
            duration_seconds=120,
        )
    )

    assert captured["temperature"] == 0.7
