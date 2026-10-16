"""
WS2: experimental geometric posture measurements.

These tests check the arithmetic on synthetic poses only. They do not
show that the measurements detect real compensation.
"""

import pytest

from src.compensation_detection.geometric_measures import (
    ExperimentalPostureMonitor,
)
from src.simulation.synthetic_pose import (
    ArmPose,
    BodyPose,
    LEFT_HIP,
    empty_results,
    make_results,
)


def run(moving_pose, side="LEFT", baseline=20, moving=30):
    monitor = ExperimentalPostureMonitor(side, baseline_frames=15)
    for _ in range(baseline):
        monitor.update(make_results(BodyPose()), movement_active=False)
    for _ in range(moving):
        monitor.update(make_results(moving_pose), movement_active=True)
    return monitor.summary()


def measure(summary, name):
    return summary["measures"][name]["max_abs_deviation"]


def raised_arm(**trunk):
    return BodyPose(left_arm=ArmPose(flexion=90), **trunk)


def test_neutral_movement_gives_near_zero_deviation():
    summary = run(raised_arm())
    assert summary["experimental"] is True
    assert summary["validated"] is False
    for name in summary["measures"]:
        assert measure(summary, name) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("field,value,name,expected", [
    ("trunk_forward_lean", 15.0, "trunk_forward_lean", 15.0),
    ("trunk_lateral_lean", 10.0, "trunk_lateral_lean", 10.0),
    ("trunk_rotation", 20.0, "trunk_rotation", 20.0),
    ("left_shoulder_raise", 0.03, "shoulder_elevation", 0.03 / 0.36),
])
def test_simulated_posture_change_is_measured(field, value, name, expected):
    summary = run(raised_arm(**{field: value}))
    assert measure(summary, name) == pytest.approx(expected, abs=0.02)


def test_contralateral_shoulder_raise_is_negative_elevation_for_affected():
    # Raising the RIGHT shoulder lowers the LEFT (affected) relative height.
    summary = run(raised_arm(right_shoulder_raise=0.03), side="LEFT")
    assert measure(summary, "shoulder_elevation") == pytest.approx(
        0.03 / 0.36, abs=0.02
    )


def test_no_baseline_means_no_measures_not_zero():
    monitor = ExperimentalPostureMonitor("LEFT", baseline_frames=15)
    for _ in range(30):
        monitor.update(make_results(raised_arm()), movement_active=True)
    summary = monitor.summary()
    assert summary["baseline_established"] is False
    assert summary["measures"] is None


def test_hidden_hips_and_missing_pose_are_counted_unavailable():
    monitor = ExperimentalPostureMonitor("LEFT", baseline_frames=15)
    hidden = make_results(BodyPose(), visibility_overrides={LEFT_HIP: 0.1})
    for _ in range(10):
        monitor.update(hidden, movement_active=False)
    monitor.update(empty_results(), movement_active=False)
    assert monitor.frames_unavailable == 11
    assert monitor.summary()["measures"] is None
