"""
Synthetic reference movements for the exercise catalogue.

References are generated from the rigid seated-body model in
src/simulation/synthetic_pose.py. They are illustrative animations,
NOT recordings of a clinician demonstration. The DTW reference curve is
computed by applying the exercise's own live angle function to the
synthetic poses, so reference and patient trajectories are measured in
exactly the same way.
"""

import numpy as np

from src.exercises.config import ReferenceMovement
from src.simulation.synthetic_pose import (
    ArmPose,
    BodyPose,
    make_results,
    world_points,
)

# Same timing as the original reference: rest -> mid -> peak -> mid -> rest
FRAME_COUNT = 30

# Display projection (normalized widget coordinates).
DISPLAY_SCALE = 0.6
DISPLAY_TOP = 0.78


def keyframe_values(peak, frame_count=FRAME_COUNT):
    """
    Linearly interpolated values over rest, half, peak, half, rest,
    matching the original keyframe interpolation scheme.
    """

    keys = [0.0, peak / 2.0, peak, peak / 2.0, 0.0]
    per_segment = max(1, frame_count // (len(keys) - 1))

    values = []
    for start, end in zip(keys[:-1], keys[1:]):
        for step in range(per_segment):
            alpha = step / per_segment
            values.append(start * (1.0 - alpha) + end * alpha)

    values = values[:frame_count]
    while len(values) < frame_count:
        values.append(keys[-1])

    return values


def project_for_display(points, view):
    """
    Project world points to 2D display coordinates.

    front: mirror-style view (subject's left on the screen's left),
           matching the mirrored live camera display.
    side:  profile view, subject facing the right of the screen.
    """

    projected = np.zeros((points.shape[0], 2), dtype=np.float32)

    if view == "side":
        projected[:, 0] = 0.5 - DISPLAY_SCALE * points[:, 2]
    else:
        projected[:, 0] = 0.5 - DISPLAY_SCALE * points[:, 0]

    projected[:, 1] = DISPLAY_TOP + DISPLAY_SCALE * points[:, 1]
    return projected


def arm_pose(side, **arm):
    pose = BodyPose()
    if side.upper() == "LEFT":
        pose.left_arm = ArmPose(**arm)
    else:
        pose.right_arm = ArmPose(**arm)
    return pose


def synthetic_reference(pose_fn, angle_fn, side, peak, view):
    """
    Build a ReferenceMovement.

    pose_fn(side, value) -> BodyPose for a movement value in [0, peak].
    """

    frames = []
    curve = []

    for value in keyframe_values(peak):
        pose = pose_fn(side, value)
        frames.append(project_for_display(world_points(pose), view))

        reading = angle_fn(make_results(pose), side)
        if reading is not None:
            curve.append(float(reading.value))

    return ReferenceMovement(
        frames=np.asarray(frames, dtype=np.float32),
        angle_curve=curve,
        synthetic=True,
    )
