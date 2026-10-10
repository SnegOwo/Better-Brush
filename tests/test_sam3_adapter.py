"""Tests for the lazily-imported SAM 3 adapter.

Everything runs against fake ``sam3`` builders injected into the adapters, so
the suite works on machines without the optional ``sam3`` package or CUDA.

Run with:
    python -m pytest tests/test_sam3_adapter.py -v
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from src.models import sam3_adapter
from src.models.sam3_adapter import (
    PointPrompt,
    Sam3AdapterError,
    Sam3CheckpointError,
    Sam3DependencyError,
    Sam3ImageAdapter,
    Sam3StateError,
    Sam3VideoAdapter,
)

_H = 40
_W = 60


def _frame(h: int = _H, w: int = _W) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "sam3.1_multiplex.pt"
    path.write_bytes(b"not-a-real-checkpoint")
    return path


class FakeImageModel:
    """Stand-in for ``Sam3Image`` exposing the documented ``predict_inst``."""

    def __init__(self, masks, scores, logits, keyword: str = "state"):
        self._masks = np.asarray(masks)
        self._scores = np.asarray(scores, dtype=np.float32)
        self._logits = logits
        self._keyword = keyword
        self.calls: list[dict] = []

    def predict_inst(self, **kwargs):
        self.calls.append(kwargs)
        if self._keyword not in kwargs:
            raise TypeError(
                f"predict_inst() got an unexpected keyword argument '{self._keyword}'"
            )
        return self._masks, self._scores, self._logits


class FakeProcessor:
    """Stand-in for ``Sam3Processor`` that records the encoded frames."""

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self.frames: list[np.ndarray] = []

    def set_image(self, image, state=None):
        self.frames.append(np.asarray(image))
        return {"original_height": image.shape[0], "original_width": image.shape[1]}


def _candidates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.zeros((_H, _W), dtype=bool)
    left[5:20, 5:20] = True
    right = np.zeros((_H, _W), dtype=bool)
    right[10:35, 30:55] = True
    masks = np.stack([left, right])
    scores = np.array([0.9, 0.7], dtype=np.float32)
    logits = np.zeros((2, 256, 256), dtype=np.float32)
    return masks, scores, logits


def _image_adapter_with_model(model, tmp_path) -> Sam3ImageAdapter:
    def _builder(**build_kwargs):
        model.build_kwargs = build_kwargs
        return model

    return Sam3ImageAdapter(
        checkpoint_path=_checkpoint(tmp_path),
        device="cpu",
        model_builder=_builder,
        processor_factory=FakeProcessor,
    )


class FakeVideoPredictor:
    """Stand-in for ``Sam3MultiplexVideoPredictor`` request/response API."""

    def __init__(self, frames: dict[int, dict] | None = None):
        self.requests: list[dict] = []
        self.stream_requests: list[dict] = []
        self._frames = frames or {}
        self.build_kwargs: dict | None = None

    def handle_request(self, request):
        self.requests.append(request)
        kind = request["type"]
        if kind == "start_session":
            return {"session_id": "session-1"}
        if kind == "add_prompt":
            return {"frame_index": request["frame_index"], "outputs": {}}
        return {"is_success": True}

    def handle_stream_request(self, request):
        self.stream_requests.append(request)
        indices = sorted(
            self._frames, reverse=request.get("propagation_direction") == "backward"
        )
        for frame_index in indices:
            yield {
                "frame_index": frame_index,
                "outputs": self._frames[frame_index],
            }


def _video_frame(obj_ids, scores, masks) -> dict:
    return {
        "out_obj_ids": np.asarray(obj_ids, dtype=np.int64),
        "out_probs": np.asarray(scores, dtype=np.float32),
        "out_binary_masks": np.asarray(masks, dtype=bool),
    }


def _video_adapter(tmp_path, predictor, **kwargs) -> Sam3VideoAdapter:
    def _builder(**build_kwargs):
        predictor.build_kwargs = build_kwargs
        return predictor

    return Sam3VideoAdapter(
        checkpoint_path=_checkpoint(tmp_path),
        device="cpu",
        predictor_builder=_builder,
        **kwargs,
    )


# --------------------------------------------------------------------- #
# Module level
# --------------------------------------------------------------------- #
def test_module_imports_without_sam3(monkeypatch):
    monkeypatch.setitem(sys.modules, "sam3", None)
    assert callable(sam3_adapter.sam3_is_available)
    assert isinstance(sam3_adapter.sam3_is_available(), bool)


def test_missing_sam3_raises_dependency_error_with_hints(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "sam3" or name.startswith("sam3."):
            raise ImportError("No module named 'sam3'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.delitem(sys.modules, "sam3", raising=False)

    adapter = Sam3ImageAdapter(checkpoint_path="unused.pt")
    with pytest.raises(Sam3DependencyError) as excinfo:
        adapter.ensure_loaded()
    message = str(excinfo.value)
    assert "sam3" in message
    assert "facebookresearch/sam3" in message
    assert issubclass(Sam3DependencyError, Sam3AdapterError)


def test_missing_checkpoint_mentions_gated_hf_repo(tmp_path):
    adapter = Sam3ImageAdapter(
        checkpoint_path=tmp_path / "nope.pt",
        model_builder=lambda **kwargs: object(),
        processor_factory=FakeProcessor,
    )
    with pytest.raises(Sam3CheckpointError) as excinfo:
        adapter.ensure_loaded()
    message = str(excinfo.value)
    assert "facebook/sam3" in message
    assert "gated" in message.lower()
    assert "hf auth login" in message


def test_checkpoint_is_resolved_from_checkpoint_dir(tmp_path):
    _checkpoint(tmp_path)
    built: dict = {}

    def _builder(**kwargs):
        built.update(kwargs)
        return object()

    adapter = Sam3ImageAdapter(
        checkpoint_dir=tmp_path,
        device="cpu",
        model_builder=_builder,
        processor_factory=FakeProcessor,
    )
    adapter.ensure_loaded()
    assert built["checkpoint_path"].endswith("sam3.1_multiplex.pt")
    assert built["load_from_HF"] is False
    assert built["enable_inst_interactivity"] is True
    assert built["device"] == "cpu"


def test_missing_local_checkpoint_is_reported(tmp_path):
    adapter = Sam3ImageAdapter(
        checkpoint_dir=tmp_path / "empty",
        model_builder=lambda **kwargs: object(),
        processor_factory=FakeProcessor,
    )
    with pytest.raises(Sam3CheckpointError):
        adapter.ensure_loaded()


# --------------------------------------------------------------------- #
# Image adapter
# --------------------------------------------------------------------- #
def test_set_image_and_predict_contract(tmp_path):
    model = FakeImageModel(*_candidates(), keyword="state")
    adapter = _image_adapter_with_model(model, tmp_path)
    state = adapter.set_image(_frame())
    assert state == {"original_height": _H, "original_width": _W}
    assert adapter.frame_hw == (_H, _W)

    coords = np.array([[10, 10], [40, 20]], dtype=np.float32)
    labels = np.array([1, 0], dtype=np.int32)
    masks, scores, logits = adapter.predict(coords, labels, multimask_output=True)

    assert masks.shape == (2, _H, _W)
    assert masks.dtype == bool
    assert scores.shape == (2,) and scores.dtype == np.float32
    assert logits is not None and logits.shape == (2, 256, 256)

    call = model.calls[-1]
    np.testing.assert_allclose(call["point_coords"], coords)
    np.testing.assert_array_equal(call["point_labels"], labels)
    assert call["multimask_output"] is True
    assert call["state"]["original_width"] == _W


def test_predict_defaults_to_positive_labels_and_single_mask(tmp_path):
    masks, _scores, _logits = _candidates()
    single = masks[:1]
    model = FakeImageModel(single, np.array([0.8], np.float32), None)
    adapter = _image_adapter_with_model(model, tmp_path)
    adapter.set_image(_frame())
    out_masks, out_scores, out_logits = adapter.predict([[10, 10]])

    assert out_masks.shape == (1, _H, _W)
    assert out_scores.shape == (1,)
    assert out_logits is None
    np.testing.assert_array_equal(model.calls[-1]["point_labels"], np.array([1]))
    assert model.calls[-1]["multimask_output"] is False


def test_predict_falls_back_to_inference_state_keyword(tmp_path):
    _masks, _scores, _logits = _candidates()
    model = FakeImageModel(_masks, np.array([0.5], np.float32), None, keyword="state")
    adapter = _image_adapter_with_model(model, tmp_path)
    adapter.set_image(_frame())
    adapter.predict([[10, 10]])
    assert "state" in model.calls[-1]

    legacy = FakeImageModel(_masks, np.array([0.5], np.float32), None)
    legacy.predict_inst = _only_inference_state(legacy)
    legacy_adapter = _image_adapter_with_model(legacy, tmp_path)
    legacy_adapter.set_image(_frame())
    legacy_adapter.predict([[10, 10]])
    assert "inference_state" in legacy.calls[-1]


def _only_inference_state(model):
    def _predict_inst(inference_state=None, **kwargs):
        model.calls.append({"inference_state": inference_state, **kwargs})
        return model._masks, model._scores, model._logits

    return _predict_inst


def test_predict_requires_set_image(tmp_path):
    model = FakeImageModel(*_candidates())
    adapter = _image_adapter_with_model(model, tmp_path)
    with pytest.raises(Sam3StateError):
        adapter.predict([[1, 1]])


def test_predict_rejects_mismatched_labels(tmp_path):
    model = FakeImageModel(*_candidates())
    adapter = _image_adapter_with_model(model, tmp_path)
    adapter.set_image(_frame())
    with pytest.raises(ValueError):
        adapter.predict([[1, 1], [2, 2]], [1])


def test_set_image_validates_frame_shape(tmp_path):
    model = FakeImageModel(*_candidates())
    adapter = _image_adapter_with_model(model, tmp_path)
    with pytest.raises(ValueError):
        adapter.set_image(np.zeros((10, 10), dtype=np.uint8))


def test_predict_from_mask_ranks_by_iou(tmp_path):
    masks, _scores, _logits = _candidates()
    left, right = masks[0], masks[1]
    scores = np.array([0.99, 0.5], dtype=np.float32)
    model = FakeImageModel(np.stack([right, left]), scores, None)
    adapter = _image_adapter_with_model(model, tmp_path)

    best_mask, best_score = adapter.predict_from_mask(_frame(), left, n_points=3)

    assert best_score == pytest.approx(0.5)
    np.testing.assert_array_equal(best_mask, left)
    coords = model.calls[-1]["point_coords"]
    assert coords.shape[0] >= 1
    assert coords[:, 0].min() >= 5 and coords[:, 0].max() < 20


def test_predict_from_mask_falls_back_to_best_score(tmp_path):
    far = np.zeros((_H, _W), dtype=bool)
    far[35:39, 55:59] = True
    _masks, scores, _logits = _candidates()
    model = FakeImageModel(np.stack([far, far]), scores, None)
    adapter = _image_adapter_with_model(model, tmp_path)

    best_mask, best_score = adapter.predict_from_mask(_frame(), far, n_points=2)

    assert best_score == pytest.approx(0.9)
    np.testing.assert_array_equal(best_mask, far)


def test_predict_from_mask_rejects_empty_mask(tmp_path):
    model = FakeImageModel(*_candidates())
    adapter = _image_adapter_with_model(model, tmp_path)
    with pytest.raises(ValueError):
        adapter.predict_from_mask(_frame(), np.zeros((_H, _W), dtype=bool))


def test_unload_releases_state(tmp_path):
    model = FakeImageModel(*_candidates())
    adapter = _image_adapter_with_model(model, tmp_path)
    adapter.set_image(_frame())
    assert adapter.is_loaded
    adapter.unload()
    assert not adapter.is_loaded
    assert adapter.frame_hw == (0, 0)
    with pytest.raises(Sam3StateError):
        adapter.predict([[1, 1]])


def test_image_adapter_does_not_build_before_first_use(tmp_path):
    model = FakeImageModel(*_candidates())
    adapter = _image_adapter_with_model(model, tmp_path)
    assert not adapter.is_loaded
    assert model.calls == []
    adapter.set_image(_frame())
    assert adapter.is_loaded
    assert model.build_kwargs["load_from_HF"] is False
    assert model.build_kwargs["enable_inst_interactivity"] is True


# --------------------------------------------------------------------- #
# Video adapter
# --------------------------------------------------------------------- #
def _mask_box(x1, y1, x2, y2) -> np.ndarray:
    mask = np.zeros((_H, _W), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def test_init_video_starts_a_session(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    predictor = FakeVideoPredictor()
    adapter = _video_adapter(tmp_path, predictor)

    session_id = adapter.init_video(frames, frame_size=(_H, _W))

    assert session_id == "session-1"
    assert adapter.session_id == "session-1"
    assert adapter.frame_hw == (_H, _W)
    request = predictor.requests[-1]
    assert request["type"] == "start_session"
    assert request["resource_path"] == str(frames)
    assert predictor.build_kwargs["checkpoint_path"].endswith("sam3.1_multiplex.pt")


def test_init_video_rejects_missing_path(tmp_path):
    adapter = _video_adapter(tmp_path, FakeVideoPredictor())
    with pytest.raises(Sam3StateError):
        adapter.init_video(tmp_path / "nope.mp4")


def test_add_points_to_video_sends_normalized_tensors(tmp_path):
    predictor = FakeVideoPredictor()
    adapter = _video_adapter(tmp_path, predictor)
    adapter.init_video(tmp_path, frame_size=(_H, _W))

    adapter.add_points_to_video(
        3,
        2,
        [PointPrompt(30.0, 10.0, 1), (0.0, 0.0, 0), (600.0, 40.0, 1)],
    )

    request = predictor.requests[-1]
    assert request["type"] == "add_prompt"
    assert request["session_id"] == "session-1"
    assert request["frame_index"] == 3
    assert request["obj_id"] == 2
    assert isinstance(request["points"], torch.Tensor)
    assert isinstance(request["point_labels"], torch.Tensor)
    np.testing.assert_allclose(
        request["points"].numpy(),
        np.array([[0.5, 0.25], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        request["point_labels"].numpy(), np.array([1, 0, 1], dtype=np.int32)
    )


def test_add_points_to_video_requires_frame_size(tmp_path):
    adapter = _video_adapter(tmp_path, FakeVideoPredictor())
    adapter.init_video(tmp_path)
    with pytest.raises(Sam3StateError):
        adapter.add_points_to_video(0, 1, [PointPrompt(1.0, 1.0)])


def test_add_points_to_video_requires_session(tmp_path):
    adapter = _video_adapter(tmp_path, FakeVideoPredictor())
    adapter.ensure_loaded()
    with pytest.raises(Sam3StateError):
        adapter.add_points_to_video(0, 1, [PointPrompt(1.0, 1.0)])


def test_add_mask_to_video_samples_interior_points(tmp_path):
    predictor = FakeVideoPredictor()
    adapter = _video_adapter(tmp_path, predictor)
    adapter.init_video(tmp_path, frame_size=(_H, _W))

    painted = _mask_box(10, 5, 30, 25)
    adapter.add_mask_to_video(7, 1, painted, n_points=4)

    request = predictor.requests[-1]
    assert request["type"] == "add_prompt"
    assert request["frame_index"] == 7
    assert request["obj_id"] == 1
    points = request["points"].numpy()
    labels = request["point_labels"].numpy()
    assert points.shape[0] == labels.shape[0] >= 1
    assert set(labels.tolist()) == {1}
    assert points[:, 0].min() >= 10 / _W
    assert points[:, 0].max() <= 30 / _W


def test_propagate_extracts_matching_object(tmp_path):
    frames = {
        0: _video_frame(
            [1, 2], [0.5, 0.9], [_mask_box(0, 0, 5, 5), _mask_box(0, 0, 6, 6)]
        ),
        1: _video_frame(
            [2, 1], [0.7, 0.8], [_mask_box(0, 0, 7, 7), _mask_box(0, 0, 8, 8)]
        ),
    }
    predictor = FakeVideoPredictor(frames)
    adapter = _video_adapter(tmp_path, predictor)
    adapter.init_video(tmp_path, frame_size=(_H, _W))
    seen: list[tuple[int, int, np.ndarray]] = []

    results = adapter.propagate_in_video(
        start_frame_idx=0,
        on_frame=lambda f, o, m: seen.append((f, o, m)),
        obj_id=2,
    )

    request = predictor.stream_requests[-1]
    assert request["type"] == "propagate_in_video"
    assert request["propagation_direction"] == "forward"
    assert request["start_frame_index"] == 0
    assert [entry[0] for entry in results] == [0, 1]
    assert [entry[1] for entry in results] == [2, 2]
    assert results[0][3] == pytest.approx(0.9)
    assert results[0][2].dtype == bool
    assert len(seen) == 2
    np.testing.assert_array_equal(seen[0][2], _mask_box(0, 0, 6, 6))


def test_propagate_can_return_all_objects(tmp_path):
    frames = {
        0: _video_frame(
            [1, 2], [0.5, 0.9], [_mask_box(0, 0, 5, 5), _mask_box(0, 0, 6, 6)]
        )
    }
    adapter = _video_adapter(tmp_path, FakeVideoPredictor(frames))
    adapter.init_video(tmp_path, frame_size=(_H, _W))

    results = adapter.propagate_in_video(obj_id=None)

    assert [entry[1] for entry in results] == [1, 2]


def test_propagate_filters_range_and_max_frames(tmp_path):
    frames = {i: _video_frame([1], [0.5], [_mask_box(0, 0, 4, 4)]) for i in range(6)}
    adapter = _video_adapter(tmp_path, FakeVideoPredictor(frames))
    adapter.init_video(tmp_path, frame_size=(_H, _W))

    results = adapter.propagate_in_video(start_frame_idx=2, max_frames=2)

    assert [entry[0] for entry in results] == [2, 3]
    assert [entry[3] for entry in results] == [0.5, 0.5]


def test_propagate_reverse_collects_and_reverses(tmp_path):
    frames = {i: _video_frame([1], [0.5], [_mask_box(0, 0, 4, 4)]) for i in range(6)}
    predictor = FakeVideoPredictor(frames)
    adapter = _video_adapter(tmp_path, predictor)
    adapter.init_video(tmp_path, frame_size=(_H, _W))
    seen: list[int] = []

    results = adapter.propagate_in_video(
        start_frame_idx=4,
        max_frames=3,
        reverse=True,
        on_frame=lambda f, o, m: seen.append(f),
    )

    request = predictor.stream_requests[-1]
    assert request["propagation_direction"] == "backward"
    assert request["max_frame_num_to_track"] == 3
    assert [entry[0] for entry in results] == [2, 3, 4]
    assert seen == [2, 3, 4]


def test_propagate_stops_on_cancel_event(tmp_path):
    frames = {i: _video_frame([1], [0.5], [_mask_box(0, 0, 4, 4)]) for i in range(5)}
    adapter = _video_adapter(tmp_path, FakeVideoPredictor(frames))
    adapter.init_video(tmp_path, frame_size=(_H, _W))
    cancel = threading.Event()
    seen: list[int] = []

    def _on_frame(frame_idx, _obj_id, _mask):
        seen.append(frame_idx)
        if frame_idx == 1:
            cancel.set()

    results = adapter.propagate_in_video(on_frame=_on_frame, cancel_event=cancel)

    assert [entry[0] for entry in results] == [0, 1]
    assert seen == [0, 1]


def test_propagate_tolerates_missing_masks(tmp_path):
    frames = {
        0: {"out_obj_ids": np.zeros(0, np.int64)},
        1: _video_frame([1], [0.4], [_mask_box(0, 0, 4, 4)]),
    }
    adapter = _video_adapter(tmp_path, FakeVideoPredictor(frames))
    adapter.init_video(tmp_path, frame_size=(_H, _W))
    results = adapter.propagate_in_video()
    assert [entry[0] for entry in results] == [1]


def test_reset_and_close_are_idempotent(tmp_path):
    predictor = FakeVideoPredictor()
    adapter = _video_adapter(tmp_path, predictor)
    adapter.init_video(tmp_path, frame_size=(_H, _W))

    assert adapter.reset() is True
    assert adapter.close() is True
    assert adapter.close() is False
    assert adapter.reset() is False
    assert adapter.session_id is None
    types = [request["type"] for request in predictor.requests]
    assert types == ["start_session", "reset_session", "close_session"]


def test_unload_closes_session_and_drops_predictor(tmp_path):
    predictor = FakeVideoPredictor()
    adapter = _video_adapter(tmp_path, predictor)
    adapter.init_video(tmp_path, frame_size=(_H, _W))
    adapter.unload()
    assert not adapter.is_loaded
    assert adapter.session_id is None
    with pytest.raises(Sam3StateError):
        adapter.propagate_in_video()


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def test_sample_interior_points_stays_inside_and_dedupes():
    mask = _mask_box(10, 10, 20, 12)
    points = sam3_adapter._sample_interior_points(mask, 8)
    assert points.ndim == 2 and points.shape[1] == 2
    assert points.dtype == np.float32
    for x, y in points:
        assert 10 <= x < 20
        assert 10 <= y < 12
    assert len({tuple(p) for p in points}) == points.shape[0]


def test_sample_interior_points_falls_back_for_thin_strokes():
    mask = np.zeros((_H, _W), dtype=bool)
    mask[10, 5:25] = True
    points = sam3_adapter._sample_interior_points(mask, 3)
    assert points.shape[0] == 3
    assert np.all(points[:, 1] == 10)


def test_set_frame_size_validates_input(tmp_path):
    adapter = _video_adapter(tmp_path, FakeVideoPredictor())
    adapter.set_frame_size(_H, _W)
    assert adapter.frame_hw == (_H, _W)
    with pytest.raises(ValueError):
        adapter.set_frame_size(0, 10)


def test_mask_iou_and_prompt_arrays():
    a = np.zeros((4, 4), dtype=bool)
    a[:2, :2] = True
    b = np.zeros((4, 4), dtype=bool)
    b[:2, :2] = True
    assert sam3_adapter._mask_iou(a, b) == pytest.approx(1.0)
    assert sam3_adapter._mask_iou(a, np.zeros((4, 4), dtype=bool)) == 0.0

    coords, labels = sam3_adapter._prompt_arrays([(1, 2), (3, 4, 0)])
    np.testing.assert_allclose(coords, np.array([[1, 2], [3, 4]], dtype=np.float32))
    np.testing.assert_array_equal(labels, np.array([1, 0], dtype=np.int32))
    with pytest.raises(ValueError):
        sam3_adapter._prompt_arrays([])
