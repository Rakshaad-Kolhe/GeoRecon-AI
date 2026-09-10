import pytest

from georecon.cli import _deep_merge, parse_set_overrides
from georecon.config import JobConfig


def test_parse_set_overrides_types_and_nesting():
    out = parse_set_overrides([
        "sfm.seq_overlap=20",
        "sfm.mapper=incremental",
        "sfm.quadratic_overlap=false",
        "masking.classes=[0, 2, 7]",
        "preset=accurate",
    ])
    assert out == {
        "sfm": {"seq_overlap": 20, "mapper": "incremental", "quadratic_overlap": False},
        "masking": {"classes": [0, 2, 7]},
        "preset": "accurate",
    }


def test_parse_set_overrides_bad_item():
    with pytest.raises(ValueError):
        parse_set_overrides(["missing_equals"])


def test_overrides_apply_to_jobconfig():
    merged = _deep_merge(
        dict(preset="fast", video_path="v.mp4"),
        parse_set_overrides(["sfm.seq_overlap=20", "sfm.min_model_size=4"]),
    )
    cfg = JobConfig(**merged)
    assert cfg.sfm.seq_overlap == 20
    assert cfg.sfm.min_model_size == 4
    assert cfg.sfm.camera_model == "SIMPLE_RADIAL"      # untouched default
    assert cfg.preset == "fast"
