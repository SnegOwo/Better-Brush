"""Tests for per-shot and per-object processing profiles."""

from __future__ import annotations

import numpy as np

from src.core.profiles import ShotProfile, ShotProfileMap, shot_ranges
from src.core.project import ObjectRecord, ProjectData, ProjectManager
from src.hardware.profiler import ModelSize


def test_scene_cuts_create_inclusive_shot_ranges() -> None:
    ranges = shot_ranges(100, [0, 30, 30, 99])
    assert [(item.shot_id, item.start_frame, item.end_frame) for item in ranges] == [
        (1, 0, 29),
        (2, 30, 98),
        (3, 99, 99),
    ]


def test_object_shot_profile_overrides_object_default() -> None:
    default = ShotProfile(name="Default", edge_smoothness=2)
    object_default = ShotProfile(name="Object", edge_smoothness=4)
    profiles = ShotProfileMap(default=default)
    shot_profile = ShotProfile(
        name="Anime close-up",
        model_size=ModelSize.SMALL,
        edge_smoothness=8,
        temporal_smoothing=False,
        detail_preservation=0.9,
    )
    profiles.set_for_shot(3, 2, shot_profile)

    assert profiles.get_for_frame(40, 100, [30], 3, object_default).name == "Anime close-up"
    assert profiles.get_for_frame(10, 100, [30], 3, object_default).name == "Object"
    assert profiles.get_for_frame(10, 100, [30], None).name == "Default"


def test_profiles_round_trip_in_project(tmp_path) -> None:
    profile = ShotProfile(
        name="Hair",
        model_size=ModelSize.BASE_PLUS,
        edge_smoothness=6,
        temporal_smoothing=False,
        detail_preservation=0.95,
        mask_inset=2,
    )
    data = ProjectData(
        video_path=str(tmp_path / "clip.mp4"),
        masks={0: np.ones((4, 4), dtype=bool)},
        default_profile=ShotProfile(name="Project default", edge_smoothness=4),
        shot_profiles=ShotProfileMap().to_records(20, [10]),
        objects=[
            ObjectRecord(
                obj_id=1,
                name="Character",
                color="#00b4f0",
                profile=profile,
            )
        ],
    )
    data.shot_profiles = ShotProfileMap().to_records(20, [10])
    profiles = ShotProfileMap()
    profiles.set_for_shot(1, 1, profile)
    data.shot_profiles = profiles.to_records(20, [10])

    loaded = ProjectManager.load(ProjectManager.save(tmp_path / "project.bbproj", data))

    assert loaded.default_profile.edge_smoothness == 4
    assert loaded.objects[0].profile is not None
    assert loaded.objects[0].profile.model_size is ModelSize.BASE_PLUS
    assert loaded.objects[0].profile.detail_preservation == 0.95
    assert loaded.objects[0].profile.mask_inset == 2
    assert loaded.shot_profiles[0].profile.mask_inset == 2
    assert loaded.shot_profiles[0].object_id == 1
    assert loaded.shot_profiles[0].shot_id == 1


def test_mask_inset_defaults_to_zero_and_is_clamped() -> None:
    assert ShotProfile().mask_inset == 0
    assert ShotProfile(mask_inset=7).mask_inset == 7
    assert ShotProfile(mask_inset=-3).mask_inset == 0
    assert ShotProfile(mask_inset=99).mask_inset == 10
    assert ShotProfile(mask_inset="4").mask_inset == 4
    assert ShotProfile(mask_inset=None).mask_inset == 0


def test_mask_inset_survives_dict_round_trip() -> None:
    profile = ShotProfile.from_dict({"name": "Inset", "mask_inset": 5})

    assert profile.mask_inset == 5
    assert profile.to_dict()["mask_inset"] == 5
    assert profile.copy().mask_inset == 5
    # Legacy projects without the key keep the default.
    assert ShotProfile.from_dict({"name": "Legacy"}).mask_inset == 0
    # Legacy "inset" alias is accepted.
    assert ShotProfile.from_dict({"inset": 3}).mask_inset == 3
