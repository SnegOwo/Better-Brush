"""Pure-Python fallback for SAM 2's compiled ``_C`` extension.

SAM 2's mask post-processing loop (hole-filling and small-island removal)
calls ``sam2.utils.misc.get_connected_components``, which requires the
compiled ``sam2._C`` extension. When that extension is missing — CPU-only
build, no CUDA toolkit on the machine — sam2 silently *skips* the
post-processing on every frame, leaving masks full of holes and jagged thin
bits (hair strands get cut, masks look rough).

We install a scipy-backed implementation of ``get_connected_components`` so
the post-processing actually runs. The compiled extension is still preferred
when it is available.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

_installed = False


def install_cc_fallback() -> None:
    """Patch ``get_connected_components`` in sam2 (and vendored EfficientTAM).

    Both packages (the pip-installed ``sam2`` and the vendored
    ``src.vendor.efficienttam`` fork) route their mask post-processing through
    a compiled ``_C`` extension that is absent on CPU-only builds. A
    scipy-backed replacement is installed for each so hole-filling and
    small-island removal keep working.
    """
    global _installed
    if _installed:
        return
    try:
        import sam2.utils.misc as _misc
        import scipy.ndimage  # noqa: F401 - presence probe for the fallback
    except Exception as exc:  # noqa: BLE001
        logger.warning("SAM 2 post-processing fallback unavailable: %s", exc)
        return

    _patch_connected_components(_misc)
    _installed = True
    logger.info(
        "Installed scipy connected-components fallback for SAM 2 post-processing."
    )

    # The vendored EfficientTAM package (src/vendor/efficienttam) is a fork of
    # the same codebase and calls the identical compiled extension.  Its
    # ``utils.misc.get_connected_components`` is imported at call time by both
    # consumers, so patching the one attribute covers both call sites.
    try:
        import src.vendor.efficienttam.utils.misc as _etam_misc
    except Exception as exc:  # noqa: BLE001 - vendored package optional
        logger.debug("EfficientTAM connected-components fallback skipped: %s", exc)
        return
    _patch_connected_components(_etam_misc)


def _patch_connected_components(misc_module) -> None:
    """Install the scipy fallback on *misc_module*'s ``get_connected_components``.

    Args:
        misc_module: A module exposing ``get_connected_components`` backed by
            a compiled extension that may be unavailable at runtime.
    """
    import scipy.ndimage

    fast_path = misc_module.get_connected_components

    def _connected_components(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return fast_path(mask)
        except Exception as exc:  # noqa: BLE001 - compiled extension missing/failed
            logger.debug("Connected components unavailable (%s)", exc)
        return _scipy_connected_components(mask, scipy.ndimage)

    misc_module.get_connected_components = _connected_components


def _scipy_connected_components(
    mask: torch.Tensor, ndimage
) -> tuple[torch.Tensor, torch.Tensor]:
    """8-connected component labeling + per-pixel area, matching ``sam2._C``.

    Args:
        mask: Binary (N, 1, H, W) tensor; non-zero is foreground.

    Returns:
        (labels, areas): int32 tensors shaped like ``mask``. ``labels`` holds
        the component id per pixel (0 = background) and ``areas`` the area of
        that component per pixel (0 for background).
    """
    device = mask.device
    structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    label_parts: list[torch.Tensor] = []
    area_parts: list[torch.Tensor] = []
    for i in range(mask.shape[0]):
        m = mask[i, 0].detach().cpu().numpy()
        labels, _ = ndimage.label(m, structure=structure)
        if labels.max() == 0:
            areas = labels
        else:
            counts = np.bincount(labels.ravel())
            areas = counts[labels]
        label_parts.append(
            torch.as_tensor(labels, dtype=torch.int32, device=device)[None, None]
        )
        area_parts.append(
            torch.as_tensor(areas, dtype=torch.int32, device=device)[None, None]
        )
    return torch.cat(label_parts, dim=0), torch.cat(area_parts, dim=0)