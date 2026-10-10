"""Tests for SAM 2 and SAM 3.1 checkpoint resolution."""

from __future__ import annotations

import json
import sys
import types

import pytest

from src.hardware.profiler import ModelSize
from src.models.checkpoint_manager import CheckpointManager


def test_sam3_1_checkpoint_uses_authenticated_hf_download(tmp_path, monkeypatch) -> None:
    source = tmp_path / "hub-cache.pt"
    source.write_bytes(b"sam3.1-weights")
    calls: list[dict] = []

    def _download(*, repo_id, filename, token=None):
        calls.append({"repo_id": repo_id, "filename": filename, "token": token})
        return str(source)

    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = _download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setenv("HF_TOKEN", "local-test-token")

    path = CheckpointManager(tmp_path).get_checkpoint_path(ModelSize.SAM3_1)

    assert path.name == "sam3.1_multiplex.pt"
    assert path.read_bytes() == b"sam3.1-weights"
    assert calls == [
        {
            "repo_id": "facebook/sam3.1",
            "filename": "sam3.1_multiplex.pt",
            "token": "local-test-token",
        }
    ]


def test_sam3_1_gated_error_explains_access(tmp_path, monkeypatch) -> None:
    module = types.ModuleType("huggingface_hub")

    def _denied(**_kwargs):
        raise RuntimeError("401 Client Error: gated repository")

    module.hf_hub_download = _denied
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    with pytest.raises(RuntimeError) as excinfo:
        CheckpointManager(tmp_path).get_checkpoint_path(ModelSize.SAM3_1)

    message = str(excinfo.value)
    assert "gated" in message
    assert "hf auth login" in message
    assert "HF_TOKEN" in message


def test_sam3_1_config_is_cached(tmp_path, monkeypatch) -> None:
    source = tmp_path / "config.json"
    source.write_text(json.dumps({"model": "sam3.1"}), encoding="utf-8")
    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = lambda **_kwargs: str(source)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    path = CheckpointManager(tmp_path).get_config_path(ModelSize.SAM3_1)

    assert path.name == "sam3.1_config.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"model": "sam3.1"}
