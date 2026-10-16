"""
Step A: the shared_v1 feature extractor.

The key property is PARITY: the same body expressed as Kinect (Toronto)
joints and as MediaPipe world landmarks must yield identical features.
These tests use synthetic geometry; they do not show that real Kinect
and MediaPipe landmarks agree (that needs Step B data).
"""

from pathlib import Path

import numpy as np
import pytest

from src.compensation_detection.shared_features import (
    FEATURE_NAMES,
    FRAME_FEATURES,
    KINECT,
    MEDIAPIPE,
    active_side_from_trial,
    causal_smooth,
    kinect_to_shared,
    mediapipe_to_shared,
    resample,
    sequence_features,
)
from src.paths import DATA_DIR
from src.simulation.session_simulator import body_pose
from src.simulation.synthetic_pose import world_points


def mediapipe_sequence(side, frames=40, **trunk):
    values = np.linspace(0, 120, frames)
    return np.stack([
        world_points(body_pose("shoulder_flexion", side, v, **trunk))
        for v in values
    ])


def as_kinect(mp_sequence):
    """Express MediaPipe-convention points as Toronto Kinect joints."""
    kinect = np.zeros((len(mp_sequence), 25, 3))
    for side in ("LEFT", "RIGHT"):
        for part, k_index in KINECT[side].items():
            mp_index = MEDIAPIPE[side][part]
            # Inverse of kinect_to_shared's (-x, -y, z) mapping.
            kinect[:, k_index] = mp_sequence[:, mp_index] * [-1, -1, 1]
    return kinect


@pytest.mark.parametrize("side", ["LEFT", "RIGHT"])
@pytest.mark.parametrize("trunk", [
    {}, {"trunk_forward_lean": 12}, {"trunk_rotation": 15},
    {"left_shoulder_raise": 0.02},
])
def test_kinect_and_mediapipe_paths_give_identical_features(side, trunk):
    mp = mediapipe_sequence(side, **trunk)
    from_mp = sequence_features(mediapipe_to_shared(mp, side), side)
    from_kinect = sequence_features(
        kinect_to_shared(as_kinect(mp), side), side
    )
    assert from_mp.shape == (len(mp), len(FEATURE_NAMES))
    np.testing.assert_allclose(from_mp, from_kinect, atol=1e-9)


def test_features_are_mirror_symmetric_between_sides():
    left = sequence_features(
        mediapipe_to_shared(mediapipe_sequence("LEFT"), "LEFT"), "LEFT")
    right = sequence_features(
        mediapipe_to_shared(mediapipe_sequence("RIGHT"), "RIGHT"), "RIGHT")
    np.testing.assert_allclose(left, right, atol=1e-9)


def test_features_do_not_depend_on_future_frames():
    mp = mediapipe_sequence("LEFT")
    altered = mp.copy()
    altered[30:] += 0.2  # change only the future
    a = sequence_features(mediapipe_to_shared(mp, "LEFT"), "LEFT")
    b = sequence_features(mediapipe_to_shared(altered, "LEFT"), "LEFT")
    np.testing.assert_allclose(a[:30], b[:30], atol=1e-12)


def test_features_invariant_to_horizontal_camera_rotation():
    # Only the vertical axis is assumed; rotating the scene about the
    # vertical axis must not change any feature.
    mp = mediapipe_sequence("LEFT", trunk_forward_lean=10)
    angle = np.radians(35)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    rotated = mp @ rotation.T
    a = sequence_features(mediapipe_to_shared(mp, "LEFT"), "LEFT")
    b = sequence_features(mediapipe_to_shared(rotated, "LEFT"), "LEFT")
    # acos of nearly parallel vectors limits precision to ~1e-6 degrees.
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_features_are_unit_free():
    mp = mediapipe_sequence("LEFT")
    a = sequence_features(mediapipe_to_shared(mp, "LEFT"), "LEFT")
    b = sequence_features(mediapipe_to_shared(mp * 2.5, "LEFT"), "LEFT")
    # acos of nearly parallel vectors limits precision to ~1e-6 degrees.
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_expected_feature_responses():
    def static(**trunk):
        seq = mediapipe_sequence("LEFT", frames=6, **trunk)
        f = sequence_features(mediapipe_to_shared(seq, "LEFT"), "LEFT")[-1]
        return dict(zip(FRAME_FEATURES, f[:len(FRAME_FEATURES)]))

    neutral = static()
    assert static(trunk_forward_lean=15)["trunk_sagittal_inclination"] == (
        pytest.approx(15.0, abs=1e-6))
    assert static(trunk_rotation=20)["shoulder_hip_yaw"] == pytest.approx(
        20.0, abs=1e-6)
    assert static(left_shoulder_raise=0.03)["shoulder_line_tilt"] > (
        neutral["shoulder_line_tilt"] + 4)


def test_causal_smooth_is_trailing_mean():
    x = np.arange(6, dtype=float)[:, None]
    np.testing.assert_allclose(
        causal_smooth(x, 3)[:, 0], [0, 0.5, 1, 2, 3, 4])


def test_resample_to_30hz():
    times = np.array([0.0, 0.1, 0.2, 0.3])          # 10 Hz
    values = np.array([[0.0], [1.0], [2.0], [3.0]])
    new_times, new_values = resample(times, values, rate=30.0)
    assert len(new_times) == 10
    np.testing.assert_allclose(new_values[:, 0], new_times * 10, atol=1e-9)


@pytest.mark.parametrize("name,expected", [
    ("Rch_Fwr_Bck_L", "LEFT"), ("Rch_Sd2Sd_Bck_TrRot_R", "RIGHT"),
    ("Rch_Fwr_Bck_L_2", "LEFT"), ("Rch_Fwr_Bck_R_1", "RIGHT"),
])
def test_active_side_parsing(name, expected):
    assert active_side_from_trial(name) == expected


@pytest.mark.skipif(not (DATA_DIR / "toronto" / "data_new").exists(),
                    reason="Toronto data not present")
def test_every_toronto_trial_name_has_a_side():
    trials = Path(DATA_DIR / "toronto" / "data_new").glob("*/*")
    for trial in trials:
        if trial.is_dir():
            assert active_side_from_trial(trial.name) in ("LEFT", "RIGHT")
