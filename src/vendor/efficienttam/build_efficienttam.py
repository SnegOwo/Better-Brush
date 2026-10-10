# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path

import src.vendor.efficienttam

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Better Brush modification: the upstream builder resolves configs through
# Hydra's global search path (``initialize_config_module``), which is a
# process-wide singleton.  Better Brush also ships Meta's ``sam2`` package,
# which registers its own search path at import time, so whichever package
# imported first would win and the other one's configs would go missing.
# Configs are therefore loaded straight from this package's ``configs/``
# directory with OmegaConf (identical result, no global state).

_CONFIG_ROOT = Path(src.vendor.efficienttam.__path__[0]) / "configs"


def _load_config(config_file, overrides):
    """Resolve *config_file* inside this package and apply *overrides*.

    Accepts a path relative to the package root
    (``configs/efficienttam/efficienttam_s_512x512.yaml``), a bare config
    name (``efficienttam_s_512x512.yaml``) or an absolute path.
    """
    candidates = [
        Path(config_file),
        _CONFIG_ROOT / config_file,
        _CONFIG_ROOT / "efficienttam" / Path(config_file).name,
        _CONFIG_ROOT / "efficienttam" / config_file,
    ]
    for candidate in candidates:
        if candidate.is_file():
            config_path = candidate
            break
    else:
        raise FileNotFoundError(
            f"EfficientTAM config not found for {config_file!r} "
            f"(looked in {_CONFIG_ROOT})"
        )
    cfg = OmegaConf.load(config_path)
    if overrides:
        dotlist = [
            item[2:] if item.startswith("++") else item.lstrip("+")
            for item in overrides
        ]
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    OmegaConf.resolve(cfg)
    return cfg

# Working on putting efficient track anything models on Facebook Hugging Face Hub.
# This is just for demonstration.
# Please download efficient track anything models from https://huggingface.co/yunyangx/efficient-track-anything.
# and use build_efficienttam/build_efficienttam_video_predictor for loading them.
HF_MODEL_ID_TO_FILENAMES = {
    "facebook/efficienttam_s": (
        "configs/efficienttam/efficienttam_s.yaml",
        "efficienttam_s.pt",
    ),
    "facebook/efficienttam_s_512x512": (
        "configs/efficienttam/efficienttam_s_512x512.yaml",
        "efficienttam_s_512x512.pt",
    ),
    "facebook/efficienttam_s_1": (
        "configs/efficienttam/efficienttam_s_1.yaml",
        "efficienttam_s_1.pt",
    ),
    "facebook/efficienttam_s_2": (
        "configs/efficienttam/efficienttam_s_2.yaml",
        "efficienttam_s_2.pt",
    ),
    "facebook/efficienttam_ti": (
        "configs/efficienttam/efficienttam_ti.yaml",
        "efficienttam_ti.pt",
    ),
    "facebook/efficienttam_ti_512x512": (
        "configs/efficienttam/efficienttam_ti_512x512.yaml",
        "efficienttam_ti_512x512.pt",
    ),
    "facebook/efficienttam_ti_1": (
        "configs/efficienttam/efficienttam_ti_1.yaml",
        "efficienttam_ti_1.pt",
    ),
    "facebook/efficienttam_ti_2": (
        "configs/efficienttam/efficienttam_ti_2.yaml",
        "efficienttam_ti_2.pt",
    ),
}


def build_efficienttam(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = _load_config(config_file, hydra_overrides_extra)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_efficienttam_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).major < 8:
        print("Disable torch compile due to unsupported GPU.")
        hydra_overrides_extra = ["++model.compile_image_encoder=False"]
        vos_optimized = False

    hydra_overrides = [
        "++model._target_=src.vendor.efficienttam.efficienttam_video_predictor.EfficientTAMVideoPredictor",
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=src.vendor.efficienttam.efficienttam_video_predictor.EfficientTAMVideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let efficienttam_base handle this
        ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = _load_config(config_file, hydra_overrides)
    model = instantiate(cfg.model, _recursive_=True)
    if ckpt_path is not None:
        _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_efficienttam_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_efficienttam(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_efficienttam_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_efficienttam_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
