"""
Scripted synthetic exercise sessions for offline testing.

Generates sequences of MediaPipe-style results in which a seated body
model performs repetitions of a catalogue exercise. Optional Gaussian
landmark noise, hidden frames and trunk perturbations can be added.

Synthetic sessions exercise the software pipeline only. They are not
evidence of accuracy on real webcam data.
"""

import math

import numpy as np

from src.simulation.synthetic_pose import (
    ArmPose,
    BodyPose,
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    empty_results,
    make_results,
)


def _arm_for(exercise_id, value):
    if exercise_id == "shoulder_flexion":
        return ArmPose(flexion=value)
    if exercise_id == "shoulder_abduction":
        return ArmPose(abduction=value)
    if exercise_id == "elbow_flexion":
        return ArmPose(elbow_flexion=value)
    if exercise_id == "forward_reach":
        fraction = max(0.0, min(0.99, value / 100.0))
        return ArmPose(flexion=math.degrees(math.asin(fraction)))
    raise ValueError(f"No simulation model for {exercise_id!r}")


def body_pose(exercise_id, side, value, **trunk):
    pose = BodyPose(**trunk)
    arm = _arm_for(exercise_id, value)
    if side.upper() == "LEFT":
        pose.left_arm = arm
    else:
        pose.right_arm = arm
    return pose


def repetition_values(peak, rest=0.0, rise=20, hold=10, pause=10):
    """One repetition: rest, raise, hold, lower, rest."""
    up = list(np.linspace(rest, peak, rise))
    return (
        [rest] * pause + up + [peak] * hold + up[::-1] + [rest] * pause
    )


def simulate_session(
    exercise_id,
    side,
    peaks,
    rest=0.0,
    noise_m=0.0,
    seed=0,
    trunk=None,
    drop_every=None,
):
    """
    Build a list of results for one repetition per entry in `peaks`.

    noise_m: std-dev of Gaussian noise (metres) added to world
        landmarks of the moving arm (image landmarks follow).
    trunk: dict of BodyPose trunk fields applied while the arm is
        above rest (simulated compensation-like movement).
    drop_every: every n-th frame has no detected pose.
    """

    rng = np.random.default_rng(seed)
    results = []

    arm_indices = (
        (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        if side.upper() == "LEFT"
        else (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    )

    frame = 0
    for peak in peaks:
        for value in repetition_values(peak, rest=rest):
            frame += 1

            if drop_every and frame % drop_every == 0:
                results.append(empty_results())
                continue

            moving = value > rest + 1e-9
            pose = body_pose(
                exercise_id, side, value,
                **(trunk if (trunk and moving) else {}),
            )
            result = make_results(pose)

            if noise_m:
                for index in arm_indices:
                    delta = rng.normal(0.0, noise_m, 3)
                    world = result.pose_world_landmarks.landmark[index]
                    world.x += delta[0]
                    world.y += delta[1]
                    world.z += delta[2]

            results.append(result)

    return results


def timestamps(count, fps=30.0):
    return [i / fps for i in range(count)]
