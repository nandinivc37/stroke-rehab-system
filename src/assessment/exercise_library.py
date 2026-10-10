# src/assessment/exercise_library.py

import numpy as np


# =========================================================
# MediaPipe Pose landmark indices
# =========================================================

NOSE = 0

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LEFT_WRIST = 15
RIGHT_WRIST = 16

LEFT_HIP = 23
RIGHT_HIP = 24


def create_base_pose():
    pose = np.zeros((33, 2), dtype=np.float32)

    # -----------------------------
    # Seated upper-body reference
    # -----------------------------

    # Head
    pose[0] = [0.50, 0.12]

    # Shoulders
    pose[11] = [0.43, 0.28]   # left shoulder
    pose[12] = [0.57, 0.28]   # right shoulder

    # Arms - resting position
    pose[13] = [0.37, 0.43]   # left elbow
    pose[14] = [0.63, 0.43]   # right elbow

    pose[15] = [0.37, 0.58]   # left wrist
    pose[16] = [0.63, 0.58]   # right wrist

    # Small amount of torso so the reference
    # still looks like a seated person.
    pose[23] = [0.45, 0.58]    # left hip
    pose[24] = [0.55, 0.58]    # right hip

    return pose


def create_shoulder_flexion_sequence(
    affected_side="LEFT",
    frame_count=30,
):
    affected_side = affected_side.upper()

    if affected_side not in {"LEFT", "RIGHT"}:
        raise ValueError(
            "affected_side must be LEFT or RIGHT"
        )

    base_pose = create_base_pose()

    rest = base_pose.copy()
    middle = base_pose.copy()
    raised = base_pose.copy()

    # --------------------------------
    # LEFT shoulder flexion
    # --------------------------------

    if affected_side == "LEFT":

       # Rest: arm down ≈ 0°
        rest[LEFT_ELBOW] = [0.37, 0.43]
        rest[LEFT_WRIST] = [0.37, 0.58]

        # Middle: arm at ≈ 45°
        middle[LEFT_ELBOW] = [0.325, 0.386]
        middle[LEFT_WRIST] = [0.218, 0.492]

        # Raised: arm horizontal ≈ 90°
        raised[LEFT_ELBOW] = [0.355, 0.28]
        raised[LEFT_WRIST] = [0.18, 0.28]

    # --------------------------------
    # RIGHT shoulder flexion
    # --------------------------------

    else:

        # Rest
        # Rest: arm down ≈ 0°
        rest[RIGHT_ELBOW] = [0.63, 0.43]
        rest[RIGHT_WRIST] = [0.63, 0.58]

        # Middle: arm at ≈ 45°
        middle[RIGHT_ELBOW] = [0.675, 0.386]
        middle[RIGHT_WRIST] = [0.782, 0.492]

        # Raised: arm horizontal ≈ 90°
        raised[RIGHT_ELBOW] = [0.645, 0.28]
        raised[RIGHT_WRIST] = [0.82, 0.28]
    # --------------------------------
    # Complete movement
    # --------------------------------

    keyframes = [
        rest,
        middle,
        raised,
        middle,
        rest,
    ]

    frames_per_segment = max(
        1,
        frame_count // (len(keyframes) - 1),
    )

    frames = []

    for index in range(len(keyframes) - 1):

        start = keyframes[index]
        end = keyframes[index + 1]

        for step in range(frames_per_segment):

            alpha = step / frames_per_segment

            frame = (
                start * (1.0 - alpha)
                + end * alpha
            )

            frames.append(frame)

    frames = frames[:frame_count]

    while len(frames) < frame_count:
        frames.append(
            keyframes[-1].copy()
        )

    return np.asarray(
        frames,
        dtype=np.float32,
    )


# =========================================================
# Exercise library
# =========================================================

EXERCISE_LIBRARY = {

    "Shoulder Flexion": {
        "exercise_id": "shoulder_flexion",
        "name": "Shoulder Flexion",
        "joint": "Shoulder",
        "body_region": "Upper Limb",
        "default_difficulty": "Easy",
        "description": (
            "Raise the affected arm forward and upward "
            "while keeping the trunk stable."
        ),
        "reference_generator": create_shoulder_flexion_sequence,
    },

}


# =========================================================
# Library access
# =========================================================

def get_exercise(exercise_name):
    """
    Return exercise information from the library.
    """

    return EXERCISE_LIBRARY.get(exercise_name)


def get_reference_sequence(
    exercise_name,
    affected_side="LEFT",
):
    """
    Generate the reference pose sequence for an exercise.
    """

    exercise = get_exercise(exercise_name)

    if exercise is None:
        raise ValueError(
            f"Exercise not found: {exercise_name}"
        )

    generator = exercise["reference_generator"]

    return generator(
        affected_side=affected_side
    )