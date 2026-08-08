from io import BytesIO

from nas_video_summarizer import person_filter as pf


def test_custom_model_dir_is_used(tmp_path, monkeypatch):
    model = tmp_path / "yolov11n.onnx"
    monkeypatch.setattr(pf, "_download", lambda url, dest: dest.write_bytes(b"model"))

    assert pf._ensure_models("yolov11n", model_dir=tmp_path) == model
    assert model.read_bytes() == b"model"


def test_download_replaces_partial_file_atomically(tmp_path, monkeypatch):
    destination = tmp_path / "model.onnx"
    partial = tmp_path / "model.onnx.part"
    partial.write_bytes(b"incomplete")

    monkeypatch.setattr(pf.request, "urlopen", lambda url: BytesIO(b"complete"))

    pf._download("https://example.invalid/model.onnx", destination)

    assert destination.read_bytes() == b"complete"
    assert not partial.exists()


def test_age_model_assets_use_configured_model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pf, "_download", lambda url, destination: destination.write_bytes(b"model")
    )

    paths = pf._ensure_age_models(tmp_path)

    assert {path.name for path in paths} == {
        "opencv_face_detector.pbtxt",
        "opencv_face_detector_uint8.pb",
        "age_deploy.prototxt",
        "age_net.caffemodel",
    }
    assert all(path.parent == tmp_path and path.exists() for path in paths)


def test_face_matches_smallest_containing_person_box():
    boxes = [
        [0.0, 0.0, 1.0, 1.0],
        [0.2, 0.2, 0.3, 0.3],
    ]

    assert pf.PersonFilter._matching_person((0.3, 0.3), boxes) == 1
    assert pf.PersonFilter._matching_person((1.2, 0.3), boxes) is None
