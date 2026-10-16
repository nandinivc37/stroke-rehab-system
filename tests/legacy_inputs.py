"""
Scripted inputs reproducing the original shoulder-raise behaviour.

The original widget measured the angle between the shoulder->wrist
vector and straight down in normalized IMAGE coordinates (2D). These
helpers build MediaPipe-style results whose image landmarks produce an
exact requested 2D angle, so the original counting/segmentation/DTW
behaviour can be pinned down before refactoring.
"""

import math

import numpy as np

from src.simulation.synthetic_pose import (
    BodyPose,
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    empty_results,
    make_results,
    project_to_image,
    world_points,
)

ARM_LENGTH = 0.30


def legacy_results(angle, side="LEFT", arm_visibility=0.99):
    """Results whose 2D image-plane shoulder->wrist angle equals `angle`."""

    image = project_to_image(world_points(BodyPose()))

    if side == "LEFT":
        shoulder, elbow, wrist, outward = (
            LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, -1.0
        )
    else:
        shoulder, elbow, wrist, outward = (
            RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, 1.0
        )

    a = math.radians(angle)
    direction = np.array([outward * math.sin(a), math.cos(a), 0.0])

    image[elbow] = image[shoulder] + 0.5 * ARM_LENGTH * direction
    image[wrist] = image[shoulder] + ARM_LENGTH * direction

    overrides = {}
    if arm_visibility != 0.99:
        overrides = {shoulder: arm_visibility, elbow: arm_visibility,
                     wrist: arm_visibility}

    return make_results(image_points=image, visibility_overrides=overrides)


def ramp(start, end, frames):
    return list(np.linspace(start, end, frames))


def rep_cycle(peak=120.0):
    """Rest -> raise -> hold -> lower -> rest (70 frames)."""
    return (
        [0.0] * 10
        + ramp(0.0, peak, 20)
        + [peak] * 10
        + ramp(peak, 0.0, 20)
        + [0.0] * 10
    )


SCENARIOS = {
    # Three complete repetitions to 120 degrees.
    "three_full_reps": rep_cycle() * 3,
    # Raises that never pass the 80-degree "up" threshold.
    "partial_raises": rep_cycle(peak=70.0) * 2,
    # Single-frame spikes above threshold must not count as reps.
    "glitch_spikes": [0.0] * 10 + ([95.0] + [0.0] * 9) * 4,
    # One rep, then the session ends while the arm is still raised.
    "ends_mid_movement": rep_cycle() + ramp(0.0, 100.0, 20),
    # Up/down threshold hysteresis: hovering between 45 and 80 degrees.
    "hover_between_thresholds": [0.0] * 10 + [100.0] * 6 + [60.0] * 20
    + [100.0] * 6 + [30.0] * 10,
}


def scenario_results(name, side="LEFT"):
    return [legacy_results(a, side) for a in SCENARIOS[name]]


def occluded_arm_results():
    """A rep with the arm invisible in the middle, plus no-pose frames."""
    angles = rep_cycle()
    results = [legacy_results(a) for a in angles]
    for i in range(25, 35):
        results[i] = legacy_results(angles[i], arm_visibility=0.1)
    results[40] = empty_results()
    return results
