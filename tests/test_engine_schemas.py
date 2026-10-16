"""Job request parsing — the panel sends JSON, so this is the trust boundary."""

from __future__ import annotations

import pytest

from src.engine.schemas import JobRequest, RequestError


def test_minimal_request_defaults():
    req = JobRequest.from_dict({"input_path": "C:/clip.mov"})
    assert req.input_path == "C:/clip.mov"
    assert req.frame_in == 0
    assert req.frame_out == -1
    assert req.preset == "anime"
    assert req.quality == "balanced"
    assert req.model is None
    assert req.keep_audio is False


def test_full_request_round_trips():
    req = JobRequest.from_dict(
        {
            "input_path": "clip.mov",
            "frame_in": 5,
            "frame_out": 42,
            "fps": 23.976,
            "width": 1920,
            "height": 1080,
            "preset": "anime",
            "quality": "best",
            "model": "small",
            "output_dir": "C:/out",
            "roi": [0.1, 0.2, 0.5, 0.5],
            "keep_audio": True,
        }
    )
    assert req.frame_out == 42
    assert req.model == "small"
    assert req.roi == (0.1, 0.2, 0.5, 0.5)
    assert req.keep_audio is True
    assert set(req.to_dict()) >= {"input_path", "frame_out", "model", "keep_audio"}


@pytest.mark.parametrize("body", [{}, {"input_path": ""}, {"input_path": 5}])
def test_input_path_required(body):
    with pytest.raises(RequestError):
        JobRequest.from_dict(body)


@pytest.mark.parametrize("preset", ["car", "", "ANIME2"])
def test_bad_preset_rejected(preset):
    with pytest.raises(RequestError):
        JobRequest.from_dict({"input_path": "a", "preset": preset})


def test_preset_is_case_insensitive():
    assert JobRequest.from_dict({"input_path": "a", "preset": "Anime"}).preset == "anime"


@pytest.mark.parametrize("quality", ["ultra", "max"])
def test_bad_quality_rejected(quality):
    with pytest.raises(RequestError):
        JobRequest.from_dict({"input_path": "a", "quality": quality})


def test_bad_model_rejected():
    with pytest.raises(RequestError):
        JobRequest.from_dict({"input_path": "a", "model": "gigantic"})


def test_frame_range_validated():
    with pytest.raises(RequestError):
        JobRequest.from_dict({"input_path": "a", "frame_in": -1})
    with pytest.raises(RequestError):
        JobRequest.from_dict({"input_path": "a", "frame_in": 10, "frame_out": 3})


def test_frame_out_minus_one_means_to_end():
    req = JobRequest.from_dict({"input_path": "a", "frame_in": 4, "frame_out": -1})
    assert req.frame_out == -1


def test_roi_shape_validated():
    with pytest.raises(RequestError):
        JobRequest.from_dict({"input_path": "a", "roi": [0.1, 0.2]})


def test_output_dir_must_be_string():
    with pytest.raises(RequestError):
        JobRequest.from_dict({"input_path": "a", "output_dir": 12})


def test_request_error_carries_kind():
    err = RequestError("nope")
    assert err.kind == "bad_request"


def test_matting_defaults_off():
    assert JobRequest.from_dict({"input_path": "a"}).matting is False


def test_matting_and_correction_fields_round_trip():
    req = JobRequest.from_dict(
        {
            "input_path": "a",
            "matting": True,
            "resume_dir": "C:/out/job1",
            "correction_add": "AAAA",
            "correction_remove": "BBBB",
        }
    )
    assert req.matting is True
    assert req.resume_dir == "C:/out/job1"
    assert req.correction_add == "AAAA"
    assert req.correction_remove == "BBBB"

    body = req.to_dict()
    assert body["matting"] is True
    assert body["resume_dir"] == "C:/out/job1"
    # The strokes are one-shot payloads, never echoed into every progress poll.
    assert "correction_add" not in body


def test_resume_dir_must_be_a_string():
    with pytest.raises(RequestError):
        JobRequest.from_dict({"input_path": "a", "resume_dir": 3})


def test_correction_frame_round_trips_and_is_validated():
    req = JobRequest.from_dict({"input_path": "a", "correction_frame": 12})
    assert req.correction_frame == 12
    assert req.to_dict()["correction_frame"] == 12

    with pytest.raises(RequestError):
        JobRequest.from_dict({"input_path": "a", "correction_frame": "12"})
