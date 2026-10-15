"""Tests for the Lugnica SAM fine-tuning kit (per-base config + installer)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = REPO_ROOT / "tools" / "sam2_training" / "colab"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load("bb_sam2_colab_finetune", COLAB_DIR / "sam2_colab_finetune.py")


@pytest.fixture(scope="module")
def installer():
    return _load("bb_install_lugnica", REPO_ROOT / "tools" / "sam2_training" / "install_lugnica.py")


def _template(driver) -> str:
    return driver.CONFIG_TEMPLATE.read_text(encoding="utf-8").replace("\r\n", "\n")


# --------------------------------------------------------------------------- #
# sam2_colab_finetune.py — per-base architecture
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("base", "embed_dim", "num_heads", "channels", "checkpoint"),
    [
        ("tiny", 96, 1, "[768, 384, 192, 96]", "sam2.1_hiera_tiny.pt"),
        ("small", 96, 1, "[768, 384, 192, 96]", "sam2.1_hiera_small.pt"),
        ("base_plus", 112, 2, "[896, 448, 224, 112]", "sam2.1_hiera_base_plus.pt"),
        ("large", 144, 2, "[1152, 576, 288, 144]", "sam2.1_hiera_large.pt"),
    ],
)
def test_apply_base_sets_architecture(
    driver, base, embed_dim, num_heads, channels, checkpoint
):
    out = driver._apply_base(_template(driver), base)
    assert f"embed_dim: {embed_dim}" in out
    assert f"num_heads: {num_heads}" in out
    assert f"backbone_channel_list: {channels}" in out
    assert f"checkpoint_path: ./checkpoints/{checkpoint}" in out


def test_apply_base_small_declares_trunk_shape_fields(driver):
    out = driver._apply_base(_template(driver), "small")
    assert "stages: [1, 2, 11, 2]" in out
    assert "global_att_blocks: [7, 10, 13]" in out
    assert "window_pos_embed_bkg_spatial_size: [7, 7]" in out


def test_apply_base_base_plus_is_identity(driver):
    template = _template(driver)
    assert driver._apply_base(template, "base_plus") == template


def test_apply_base_rejects_unknown(driver):
    with pytest.raises(ValueError):
        driver._apply_base(_template(driver), "nope")


def test_checkpoint_url_per_base(driver):
    assert driver.checkpoint_url().endswith("sam2.1_hiera_base_plus.pt")
    assert driver.checkpoint_url("small").endswith("sam2.1_hiera_small.pt")
    assert driver.checkpoint_url("tiny").endswith("sam2.1_hiera_tiny.pt")


def test_write_config_small_end_to_end(driver, tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "ImageSets").mkdir(parents=True)
    (dataset / "ImageSets" / "train.txt").write_text("", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()

    dest = driver.write_config(
        repo, dataset, base="small", amp_dtype="float16", epochs=1
    )
    text = dest.read_text(encoding="utf-8")

    assert dest == repo / "configs" / "sam2.1_training" / "anime_amv_colab.yaml"
    assert "stages: [1, 2, 11, 2]" in text
    assert "checkpoint_path: ./checkpoints/sam2.1_hiera_small.pt" in text
    assert f"img_folder: {dataset / 'JPEGImages'}" in text
    assert "amp_dtype: float16" in text
    assert "num_epochs: 1" in text


def test_write_config_requires_dataset(driver, tmp_path):
    with pytest.raises(FileNotFoundError):
        driver.write_config(tmp_path / "repo", tmp_path / "missing")


# --------------------------------------------------------------------------- #
# install_lugnica.py — sidecar written for the app
# --------------------------------------------------------------------------- #
def test_install_writes_checkpoint_and_sidecar(installer, tmp_path):
    src = tmp_path / "run" / "checkpoint_1.pt"
    src.parent.mkdir()
    src.write_bytes(b"weights")
    dest_dir = tmp_path / "custom"

    info = installer.install(
        src, base="small", label="Lugnica SAM", dest_dir=dest_dir
    )

    installed = dest_dir / installer.DEFAULT_NAME
    assert installed.read_bytes() == b"weights"
    sidecar = json.loads(
        (dest_dir / installer.SIDECAR_NAME).read_text(encoding="utf-8")
    )
    assert sidecar == {
        "checkpoint": installer.DEFAULT_NAME,
        "config": "sam2.1_hiera_s.yaml",
        "label": "Lugnica SAM",
    }
    assert info["installed"] == str(installed)


def test_install_appends_pt_extension(installer, tmp_path):
    src = tmp_path / "ckpt.pt"
    src.write_bytes(b"w")
    info = installer.install(src, name="mymodel", dest_dir=tmp_path / "custom")
    assert info["checkpoint"] == "mymodel.pt"
    assert (tmp_path / "custom" / "mymodel.pt").is_file()


def test_install_accepts_checkpoint_already_in_place(installer, tmp_path):
    dest_dir = tmp_path / "custom"
    dest_dir.mkdir()
    src = dest_dir / installer.DEFAULT_NAME
    src.write_bytes(b"w")
    info = installer.install(src, dest_dir=dest_dir)  # no self-copy crash
    assert Path(info["installed"]).read_bytes() == b"w"


def test_install_rejects_unknown_base_and_missing_file(installer, tmp_path):
    src = tmp_path / "ckpt.pt"
    src.write_bytes(b"w")
    with pytest.raises(ValueError):
        installer.install(src, base="huge", dest_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        installer.install(tmp_path / "nope.pt", dest_dir=tmp_path)


def test_installer_configs_match_app_sidecar_choices(installer):
    source = (REPO_ROOT / "src" / "models" / "sam2_manager.py").read_text(
        encoding="utf-8"
    )
    for base, config in installer.CONFIG_FOR_BASE.items():
        assert f"configs/sam2.1/{config}" in source, base
