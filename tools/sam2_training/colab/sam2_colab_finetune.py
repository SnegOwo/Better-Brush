"""Fine-tune SAM 2.1 for anime / AMV on Google Colab — one file driver.

This script automates the whole Colab workflow described in
``docs/FINE-TUNING.md`` and ``tools/sam2_training/colab/README.md``:

  1. clone the official SAM 2 repo (shallow) and ``pip install -e ".[dev]"``
  2. download only the SAM 2.1 Base+ checkpoint
  3. write a Colab-tuned config (``anime_amv_colab.yaml``) into the repo and
     patch the dataset paths + AMP dtype for the GPU it finds
  4. launch ``training/train.py`` for one GPU
  5. locate the trained checkpoint so you can copy it back to Better Brush

It runs both as a notebook cell (``%run sam2_colab_finetune.py ...`` /
``import sam2_colab_finetune as bb``) and as a plain script::

    python sam2_colab_finetune.py --dataset /content/drive/MyDrive/bb/dataset

Nothing here is Colab-specific except the default paths; on any Linux box with
a CUDA GPU you can run the same command.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

#: Official SAM 2 repository.
REPO_URL = "https://github.com/facebookresearch/sam2.git"
_CKPT_BASE = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
CKPT_URL = f"{_CKPT_BASE}sam2.1_hiera_base_plus.pt"

#: Per-size architecture. SAM 2.1's four Hiera sizes differ only in a handful of
#: fields, and ``anime_amv_colab.yaml`` is written for Base+.  Switching the base
#: means swapping the starting checkpoint *and* exactly these fields, so the
#: trained checkpoint always matches the architecture it was initialised from.
#: Values mirror ``sam2/configs/sam2.1/sam2.1_hiera_{t,s,b+,l}.yaml`` upstream.
#: ``stages``/``global_att_blocks``/``window`` are ``None`` for Base+ because its
#: values are the ``Hiera`` constructor defaults, which the template inherits.
_BASE_SPECS: dict[str, dict[str, object]] = {
    "tiny": {
        "embed_dim": 96,
        "num_heads": 1,
        "stages": "[1, 2, 7, 2]",
        "global_att_blocks": "[5, 7, 9]",
        "window": "[7, 7]",
        "window_spec": None,
        "channel_list": "[768, 384, 192, 96]",
        "checkpoint": "sam2.1_hiera_tiny.pt",
    },
    "small": {
        "embed_dim": 96,
        "num_heads": 1,
        "stages": "[1, 2, 11, 2]",
        "global_att_blocks": "[7, 10, 13]",
        "window": "[7, 7]",
        "window_spec": None,
        "channel_list": "[768, 384, 192, 96]",
        "checkpoint": "sam2.1_hiera_small.pt",
    },
    "base_plus": {
        "embed_dim": 112,
        "num_heads": 2,
        "stages": None,
        "global_att_blocks": None,
        "window": None,
        "window_spec": None,
        "channel_list": "[896, 448, 224, 112]",
        "checkpoint": "sam2.1_hiera_base_plus.pt",
    },
    "large": {
        "embed_dim": 144,
        "num_heads": 2,
        "stages": "[2, 6, 36, 4]",
        "global_att_blocks": "[23, 33, 43]",
        "window": "[7, 7]",
        "window_spec": "[8, 4, 16, 8]",
        "channel_list": "[1152, 576, 288, 144]",
        "checkpoint": "sam2.1_hiera_large.pt",
    },
}

#: Backbone used when ``--base`` is not given.
DEFAULT_BASE = "base_plus"

#: The exact Base+ Hiera block in ``anime_amv_colab.yaml``.  Kept as a literal so
#: ``_apply_base`` can swap it for another size without a YAML parser.
_OLD_TRUNK = (
    "      trunk:\n"
    "        _target_: sam2.modeling.backbones.hieradet.Hiera\n"
    "        embed_dim: 112\n"
    "        num_heads: 2\n"
    "        drop_path_rate: 0.0\n"
)

#: Where Colab usually keeps things.
DEFAULT_WORKDIR = Path("/content/sam2")
DEFAULT_DATASET = Path("/content/dataset")

#: The tuned config that ships next to this script.
_HERE = Path(__file__).resolve().parent
CONFIG_TEMPLATE = _HERE / "anime_amv_colab.yaml"


def _log(message: str) -> None:
    print(message, flush=True)


def run(cmd, *, cwd: Path | None = None, quiet: bool = False) -> None:
    """Run a command, echoing it first, and raise on a non-zero exit."""
    shown = " ".join(str(part) for part in cmd)
    _log(f"$ {shown}")
    subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
    )


def detect_amp_dtype() -> str:
    """Return ``bfloat16`` on Ampere+ GPUs, else ``float16`` (T4)."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is always present on Colab
        return "float16"
    if not torch.cuda.is_available():
        return "float16"
    major, _minor = torch.cuda.get_device_capability(0)
    # bfloat16 needs compute capability >= 8 (A100, L4, 30/40-series).  The
    # Colab-free Tesla T4 is Turing (7.5) and has no bf16 -> float16.
    return "bfloat16" if major >= 8 else "float16"


def detect_gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover
        pass
    return "CPU"


def clone_and_install(repo: Path = DEFAULT_WORKDIR, *, install: bool = True) -> Path:
    """Clone SAM 2 (if needed) and install it in editable dev mode."""
    repo = Path(repo)
    if not (repo / "training" / "train.py").is_file():
        if repo.exists():
            shutil.rmtree(repo)
        run(["git", "clone", "--depth", "1", REPO_URL, repo])
    if install:
        # Colab ships a CUDA-enabled torch; pip keeps it because it satisfies
        # the ">=2.3.1" pin, so we only pull the training extras (hydra, iopath,
        # fvcore, ...).
        run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{repo}[dev]"])
    return repo


def download_checkpoint(repo: Path = DEFAULT_WORKDIR, url: str = CKPT_URL) -> Path:
    """Download the base-plus checkpoint into ``<repo>/checkpoints``."""
    ckpt_dir = Path(repo) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dest = ckpt_dir / Path(url).name
    if dest.is_file() and dest.stat().st_size > 0:
        _log(f"checkpoint already present: {dest}")
        return dest
    wget = shutil.which("wget")
    if wget:
        run([wget, "-q", "--show-progress", "-O", dest, url])
    else:  # pragma: no cover - fallback when wget is unavailable
        import urllib.request

        _log(f"downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    return dest


def checkpoint_url(base: str = DEFAULT_BASE) -> str:
    """Official download URL for the starting checkpoint of *base*."""
    return _CKPT_BASE + str(_BASE_SPECS[base]["checkpoint"])


def _apply_base(text: str, base: str = DEFAULT_BASE) -> str:
    """Swap the template's Base+ Hiera block for *base*'s architecture.

    The trained checkpoint must be initialised from — and match the shape of —
    the same base it was trained on, so the trunk fields and ``checkpoint_path``
    move together.  Values come from the upstream ``sam2.1_hiera_<size>.yaml``
    configs.  ``base_plus`` reproduces the template byte-for-byte, so the
    default path is unchanged.
    """
    if base not in _BASE_SPECS:
        raise ValueError(
            f"unknown base {base!r}; expected one of {sorted(_BASE_SPECS)}"
        )
    spec = _BASE_SPECS[base]
    trunk = [
        "      trunk:",
        "        _target_: sam2.modeling.backbones.hieradet.Hiera",
        f"        embed_dim: {spec['embed_dim']}",
        f"        num_heads: {spec['num_heads']}",
    ]
    if spec["stages"] is not None:
        trunk.append(f"        stages: {spec['stages']}")
        trunk.append(f"        global_att_blocks: {spec['global_att_blocks']}")
    if spec["window"] is not None:
        trunk.append(f"        window_pos_embed_bkg_spatial_size: {spec['window']}")
    if spec["window_spec"] is not None:
        trunk.append(f"        window_spec: {spec['window_spec']}")
    trunk.append("        drop_path_rate: 0.0")
    if _OLD_TRUNK not in text:
        raise ValueError(
            "config template trunk block not found — was anime_amv_colab.yaml "
            "edited? _apply_base anchors on the Base+ Hiera block."
        )
    text = text.replace(_OLD_TRUNK, "\n".join(trunk) + "\n", 1)
    text = re.sub(
        r"^(\s*backbone_channel_list:\s*)\S.*$",
        lambda mo: mo.group(1) + str(spec["channel_list"]),
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^(\s*checkpoint_path:\s*)\S.*$",
        lambda mo: mo.group(1) + f"./checkpoints/{spec['checkpoint']}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    return text


def _patch_config(
    text: str,
    dataset_root: Path,
    *,
    resolution: int,
    num_frames: int,
    amp_dtype: str,
    epochs: int | None,
    num_workers: int,
) -> str:
    """Rewrite the handful of values that Colab needs without a YAML parser.

    The template is a known, tightly-controlled file, so anchoring on the exact
    key names keeps this dependency-free (no PyYAML needed in the runtime).
    Replacements use a callback, not a template string, so a Windows path with
    backslashes can never be misread as a regex escape.
    """
    values = {
        "resolution": resolution,
        "num_frames": num_frames,
        "num_train_workers": num_workers,
        "amp_dtype": amp_dtype,
        "img_folder": str(dataset_root / "JPEGImages"),
        "gt_folder": str(dataset_root / "Annotations"),
        "file_list_txt": str(dataset_root / "ImageSets" / "train.txt"),
    }
    if epochs is not None:
        values["num_epochs"] = epochs
    for key, value in values.items():
        pattern = rf"^(\s*{key}:\s*)\S+"
        text = re.sub(
            pattern,
            lambda mo, v=value: mo.group(1) + str(v),
            text,
            count=1,
            flags=re.MULTILINE,
        )
    return text


def write_config(
    repo: Path = DEFAULT_WORKDIR,
    dataset_root: Path = DEFAULT_DATASET,
    *,
    resolution: int = 384,
    num_frames: int = 2,
    amp_dtype: str | None = None,
    epochs: int | None = None,
    num_workers: int = 2,
    name: str = "anime_amv_colab",
    base: str = DEFAULT_BASE,
) -> Path:
    """Write the Colab config into the repo's ``configs/sam2.1_training``."""
    repo = Path(repo)
    dataset_root = Path(dataset_root)
    if not (dataset_root / "ImageSets" / "train.txt").is_file():
        raise FileNotFoundError(
            f"{dataset_root}/ImageSets/train.txt not found — export a dataset "
            "from Better Brush (File > Save Training Data) and upload it first."
        )
    if amp_dtype is None:
        amp_dtype = detect_amp_dtype()
    text = CONFIG_TEMPLATE.read_text(encoding="utf-8").replace("\r\n", "\n")
    text = _apply_base(text, base)
    text = _patch_config(
        text,
        dataset_root,
        resolution=resolution,
        num_frames=num_frames,
        amp_dtype=amp_dtype,
        epochs=epochs,
        num_workers=num_workers,
    )
    dest = repo / "configs" / "sam2.1_training" / f"{name}.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    _log(f"wrote config: {dest}  (base={base}, amp_dtype={amp_dtype}, "
         f"{resolution}px, {num_frames} frames)")
    return dest


def train(
    repo: Path = DEFAULT_WORKDIR,
    config: Path | str = "configs/sam2.1_training/anime_amv_colab.yaml",
    *,
    num_gpus: int = 1,
) -> None:
    """Launch ``training/train.py`` for a local single-node run."""
    run(
        [
            sys.executable,
            "training/train.py",
            "-c",
            str(config),
            "--use-cluster",
            "0",
            "--num-gpus",
            str(num_gpus),
        ],
        cwd=Path(repo),
    )


def find_checkpoints(repo: Path = DEFAULT_WORKDIR) -> list[Path]:
    """Return every saved training checkpoint, newest first."""
    repo = Path(repo)
    found = [
        path
        for path in (repo / "sam2_logs").rglob("*.pt")
        if path.is_file() and path.stat().st_size > 0
    ]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument(
        "--base",
        choices=sorted(_BASE_SPECS),
        default=DEFAULT_BASE,
        help="SAM 2.1 backbone to fine-tune (default: %(default)s)",
    )
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--amp-dtype",
        default=None,
        help="float16 (T4) or bfloat16 (L4/A100); auto-detected by default",
    )
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument(
        "--no-train", action="store_true", help="prepare everything, do not train"
    )
    args = parser.parse_args(argv)

    _log(f"GPU: {detect_gpu_name()}  |  amp_dtype -> {args.amp_dtype or detect_amp_dtype()}")
    _log(f"base: {args.base}  ({_BASE_SPECS[args.base]['checkpoint']})")
    repo = clone_and_install(args.workdir, install=not args.skip_install)
    download_checkpoint(repo, checkpoint_url(args.base))
    config = write_config(
        repo,
        args.dataset,
        resolution=args.resolution,
        num_frames=args.num_frames,
        amp_dtype=args.amp_dtype,
        epochs=args.epochs,
        num_workers=args.num_workers,
        base=args.base,
    )
    if args.no_train:
        _log("--no-train set; stopping before training.")
        return 0
    train(repo, config)
    saved = find_checkpoints(repo)
    if saved:
        _log("\nTrained checkpoints (newest first):")
        for path in saved[:5]:
            _log(f"  {path}")
        _log(
            "\nDownload the newest one, then install it on the machine that runs "
            "Better Brush with:\n"
            f"  python tools/sam2_training/install_lugnica.py <downloaded.pt> "
            f"--base {args.base} --label \"Lugnica SAM\"\n"
            "It lands in checkpoints/custom/ and shows up in AI Model."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
