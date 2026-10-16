"""
Shared-joint compensation features (Step A, "shared_v1").

One feature definition used for BOTH Kinect (Toronto) skeletons and
MediaPipe world landmarks, so a model trained on one can be applied to
the other without a separate, divergent feature path.

Design constraints (see README, "Compensation detection status"):
  * Only the 8 joints both sensors provide: shoulders, elbows, wrists,
    hips. No Kinect spine joints, no MediaPipe face/hand points.
  * Only the VERTICAL axis of the sensor frame is used (Toronto +y up;
    MediaPipe -y up, assuming an upright camera). Everything else is
    rotation-invariant geometry (angles, ratios to shoulder width), so
    the unknown horizontal axis conventions of the pre-normalized
    Toronto data do not matter.
  * Unit-free: angles in degrees, distances in shoulder widths.
  * Expressed relative to the ACTIVE (exercising) arm, so left- and
    right-arm trials share one model.
  * Causal: smoothing and temporal features use only past frames.

Remaining, unverified assumptions: Kinect and MediaPipe hip landmarks
are at comparable anatomical positions; Toronto data were recorded at
about 30 Hz (live data must be resampled to the same rate).
"""

import math

import numpy as np

FEATURE_SET = "shared_v1"
TARGET_RATE_HZ = 30.0

JOINTS = (
    "shoulder_active", "shoulder_other",
    "elbow_active", "elbow_other",
    "wrist_active", "wrist_other",
    "hip_active", "hip_other",
)

# Kinect v2 joint indices (Toronto Joint_Positions.csv order).
KINECT = {
    "LEFT": {"shoulder": 4, "elbow": 5, "wrist": 6, "hip": 12},
    "RIGHT": {"shoulder": 8, "elbow": 9, "wrist": 10, "hip": 16},
}

# MediaPipe Pose landmark indices.
MEDIAPIPE = {
    "LEFT": {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23},
    "RIGHT": {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24},
}

FRAME_FEATURES = (
    "trunk_inclination",          # trunk axis vs vertical (deg)
    "trunk_lateral_inclination",  # towards the active side positive
    "trunk_sagittal_inclination", # forward positive (handedness-based)
    "shoulder_line_tilt",         # active shoulder higher -> positive
    "shoulder_height_difference", # vertical, in shoulder widths
    "shoulder_hip_yaw",           # shoulder line vs hip line (deg)
    "trunk_length",               # hip mid -> shoulder mid / width
    "elbow_angle_active",
    "elbow_angle_other",
    "arm_elevation_active",       # upper arm vs trunk axis (deg)
    "arm_elevation_other",
)

TEMPORAL_WINDOW = 5

FEATURE_NAMES = (
    list(FRAME_FEATURES)
    + [f"{name}_delta" for name in FRAME_FEATURES]
    + [f"{name}_rolling_std" for name in FRAME_FEATURES]
)


def other(side):
    return "RIGHT" if side.upper() == "LEFT" else "LEFT"


def _order(points_by_index, mapping, active_side):
    """Return (T, 8, 3) ordered as JOINTS from an indexable array."""

    active = mapping[active_side.upper()]
    passive = mapping[other(active_side)]
    order = []
    for name in JOINTS:
        part, role = name.split("_")
        order.append((active if role == "active" else passive)[part])
    return points_by_index[:, order, :]


def kinect_to_shared(positions, active_side):
    """
    Toronto (T, 25, 3) Kinect positions -> (T, 8, 3) shared joints in a
    y-DOWN frame (to match MediaPipe). Only the vertical axis matters to
    the features; x is also negated so the mapping is a proper rotation
    (handedness preserved).
    """

    positions = np.asarray(positions, dtype=np.float64)
    flipped = positions * np.array([-1.0, -1.0, 1.0])
    return _order(flipped, KINECT, active_side)


def mediapipe_to_shared(world_points, active_side):
    """MediaPipe world landmarks (T, 33, 3) -> (T, 8, 3) shared joints."""

    world_points = np.asarray(world_points, dtype=np.float64)
    if world_points.ndim == 2:
        world_points = world_points[None]
    return _order(world_points, MEDIAPIPE, active_side)


def causal_smooth(sequence, window=5):
    """Trailing moving average over up to `window` frames (causal)."""

    sequence = np.asarray(sequence, dtype=np.float64)
    out = np.empty_like(sequence)
    cumulative = np.cumsum(sequence, axis=0)
    for t in range(len(sequence)):
        start = max(0, t - window + 1)
        total = cumulative[t] - (cumulative[start - 1] if start > 0 else 0)
        out[t] = total / (t - start + 1)
    return out


def resample(times, sequence, rate=TARGET_RATE_HZ):
    """
    Linearly resample an irregularly timed (T, ...) sequence to a fixed
    rate, so temporal features match the ~30 Hz training data.
    Returns (new_times, new_sequence).
    """

    times = np.asarray(times, dtype=np.float64)
    sequence = np.asarray(sequence, dtype=np.float64)

    if len(times) < 2:
        return times, sequence

    new_times = np.arange(times[0], times[-1] + 1e-9, 1.0 / rate)
    flat = sequence.reshape(len(times), -1)
    resampled = np.stack(
        [np.interp(new_times, times, flat[:, i]) for i in range(flat.shape[1])],
        axis=1,
    )
    return new_times, resampled.reshape((len(new_times),) + sequence.shape[1:])


def _angle(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-12:
        return float("nan")
    return math.degrees(math.acos(max(-1.0, min(1.0, float(v1 @ v2 / n)))))


UP = np.array([0.0, -1.0, 0.0])  # y-down frame


def frame_features(joints, active_side):
    """
    Features for one frame of shared joints (8, 3), y-down frame.
    Returns an array ordered as FRAME_FEATURES (NaN if degenerate).
    """

    sa, so, ea, eo, wa, wo, ha, ho = joints

    width = np.linalg.norm(sa - so)
    if width < 1e-9:
        return np.full(len(FRAME_FEATURES), np.nan)

    shoulder_mid = (sa + so) / 2
    hip_mid = (ha + ho) / 2
    trunk = shoulder_mid - hip_mid

    # Horizontal lateral axis from the hips pointing to the ACTIVE side.
    lateral = ha - ho
    lateral = lateral - (lateral @ UP) * UP
    lateral_norm = np.linalg.norm(lateral)
    if lateral_norm < 1e-9 or np.linalg.norm(trunk) < 1e-9:
        return np.full(len(FRAME_FEATURES), np.nan)
    lateral = lateral / lateral_norm

    # Forward must come from a FIXED anatomical axis (towards the
    # subject's left), not the active side, or its sign would flip
    # between left- and right-arm trials. In a right-handed y-down
    # frame, left x up points the way the subject faces.
    left = lateral if active_side.upper() == "LEFT" else -lateral
    forward = np.cross(left, UP)

    trunk_unit = trunk / np.linalg.norm(trunk)

    inclination = _angle(trunk, UP)
    lateral_incl = math.degrees(math.atan2(trunk @ lateral, trunk @ UP))
    sagittal_incl = math.degrees(math.atan2(trunk @ forward, trunk @ UP))

    shoulder_line = sa - so
    tilt = math.degrees(math.atan2(
        shoulder_line @ UP,
        np.linalg.norm(shoulder_line - (shoulder_line @ UP) * UP),
    ))
    height_difference = float((sa - so) @ UP) / width

    def horizontal(v):
        return v - (v @ UP) * UP

    yaw = _angle(horizontal(sa - so), horizontal(ha - ho))

    trunk_length = float(np.linalg.norm(trunk)) / width

    elbow_active = _angle(sa - ea, wa - ea)
    elbow_other = _angle(so - eo, wo - eo)
    arm_active = _angle(ea - sa, -trunk_unit)
    arm_other = _angle(eo - so, -trunk_unit)

    return np.array([
        inclination, lateral_incl, sagittal_incl, tilt, height_difference,
        yaw, trunk_length, elbow_active, elbow_other, arm_active, arm_other,
    ], dtype=np.float64)


def sequence_features(shared_joints, active_side, smooth_window=5):
    """
    (T, 8, 3) shared joints -> (T, len(FEATURE_NAMES)) features:
    causal smoothing, per-frame features, then causal delta and
    trailing rolling standard deviation (ddof=1, 0 for first frame).
    """

    smoothed = causal_smooth(shared_joints, smooth_window)
    base = np.stack([
        frame_features(frame, active_side) for frame in smoothed
    ])

    delta = np.zeros_like(base)
    delta[1:] = base[1:] - base[:-1]

    rolling = np.zeros_like(base)
    for t in range(1, len(base)):
        start = max(0, t - TEMPORAL_WINDOW + 1)
        rolling[t] = np.std(base[start:t + 1], axis=0, ddof=1)

    return np.concatenate([base, delta, rolling], axis=1)


def active_side_from_trial(trial_name):
    """Toronto trial names end in _L / _R, optionally followed by _1/_2."""

    parts = trial_name.split("_")
    for part in reversed(parts):
        if part in ("L", "R"):
            return "LEFT" if part == "L" else "RIGHT"
    raise ValueError(f"Cannot determine active side from {trial_name!r}")
