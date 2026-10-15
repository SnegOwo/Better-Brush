"""Install a fine-tuned SAM 2 checkpoint into Better Brush as "Lugnica SAM".

Run this on the machine that runs Better Brush (your PC), *after* downloading
the checkpoint Colab produced.  It copies the weights into
``checkpoints/custom/`` and writes the ``sam2_custom.json`` sidecar that tells
the app which base architecture to build and what to call the model.

    python tools/sam2_training/install_lugnica.py path/to/checkpoint.pt
    python tools/sam2_training/install_lugnica.py ckpt.pt --base small \\
        --label "Lugnica SAM (small)"

Then pick it under **AI Model** in Better Brush (restart the app if it was open).

This is the last step of the loop in ``docs/FINE-TUNING.md``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

#: Sidecar ``config`` value the app expects for each base.  Each must match a
#: file name in ``src/models/sam2_manager.py::_HYDRA_CONFIGS``.
CONFIG_FOR_BASE: dict[str, str] = {
    "tiny": "sam2.1_hiera_t.yaml",
    "small": "sam2.1_hiera_s.yaml",
    "base_plus": "sam2.1_hiera_b+.yaml",
    "large": "sam2.1_hiera_l.yaml",
}

DEFAULT_LABEL = "Lugnica SAM"
DEFAULT_NAME = "lugnica_sam_v1.pt"
SIDECAR_NAME = "sam2_custom.json"


def repo_root() -> Path:
    """Best-effort Better Brush root (``tools/sam2_training/install_lugnica.py``)."""
    return Path(__file__).resolve().parents[2]


def custom_dir() -> Path:
    """``checkpoints/custom`` as the running app resolves it."""
    try:
        sys.path.insert(0, str(repo_root()))
        from src.core.constants import CHECKPOINTS_DIR  # type: ignore[import-not-found]

        return Path(CHECKPOINTS_DIR) / "custom"
    except ImportError:  # pragma: no cover - only when src/ is not importable
        return repo_root() / "checkpoints" / "custom"


def install(
    checkpoint: Path | str,
    *,
    base: str = "base_plus",
    label: str = DEFAULT_LABEL,
    name: str = DEFAULT_NAME,
    dest_dir: Path | str | None = None,
) -> dict[str, str]:
    """Copy *checkpoint* into ``checkpoints/custom`` and write the sidecar.

    Returns the sidecar contents plus the written paths.  Idempotent: running it
    again with a new file overwrites the checkpoint and sidecar in place.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if base not in CONFIG_FOR_BASE:
        raise ValueError(
            f"unknown base {base!r}; expected one of {sorted(CONFIG_FOR_BASE)}"
        )
    if not name.endswith(".pt"):
        name = f"{name}.pt"
    out = Path(dest_dir) if dest_dir is not None else custom_dir()
    out.mkdir(parents=True, exist_ok=True)
    dest = out / name
    if checkpoint.resolve() != dest.resolve():
        shutil.copy2(checkpoint, dest)
    sidecar = {
        "checkpoint": name,
        "config": CONFIG_FOR_BASE[base],
        "label": label,
    }
    sidecar_path = out / SIDECAR_NAME
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return {**sidecar, "installed": str(dest), "sidecar": str(sidecar_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="downloaded .pt from training")
    parser.add_argument(
        "--base",
        choices=sorted(CONFIG_FOR_BASE),
        default="base_plus",
        help="architecture the checkpoint was fine-tuned from (default: %(default)s)",
    )
    parser.add_argument(
        "--label", default=DEFAULT_LABEL, help="name shown in the app"
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help="file name inside checkpoints/custom (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    info = install(args.checkpoint, base=args.base, label=args.label, name=args.name)
    print(f"installed '{info['label']}' -> {info['installed']}")
    print(f"sidecar          -> {info['sidecar']}  (config={info['config']})")
    print("Restart Better Brush, then choose it under AI Model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
