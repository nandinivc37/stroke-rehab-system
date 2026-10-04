import numpy as np


# Kinect v2 joint indices used by the Toronto dataset

SPINE_BASE = 0
SPINE_MID = 1
NECK = 2
HEAD = 3

SHOULDER_LEFT = 4
ELBOW_LEFT = 5
WRIST_LEFT = 6
HAND_LEFT = 7

SHOULDER_RIGHT = 8
ELBOW_RIGHT = 9
WRIST_RIGHT = 10
HAND_RIGHT = 11

HIP_LEFT = 12
HIP_RIGHT = 16

SPINE_SHOULDER = 20


def vector(a, b):
    """Vector from joint a to joint b."""
    return b - a


def vector_angle(v1, v2):
    """
    Calculate the angle between two 3D vectors in degrees.
    """

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


def joint_angle(a, b, c):
    """
    Calculate angle ABC using 3D joint positions.
    """

    ba = a - b
    bc = c - b

    return vector_angle(ba, bc)


def distance(a, b):
    """Euclidean distance between two 3D joints."""
    return np.linalg.norm(a - b)


def extract_frame_features(frame):
    """
    Extract interpretable biomechanical features
    from one Toronto skeleton frame.

    Args:
        frame: array of shape (25, 3)

    Returns:
        dictionary of feature values
    """

    spine_base = frame[SPINE_BASE]
    spine_mid = frame[SPINE_MID]
    spine_shoulder = frame[SPINE_SHOULDER]

    shoulder_left = frame[SHOULDER_LEFT]
    elbow_left = frame[ELBOW_LEFT]
    wrist_left = frame[WRIST_LEFT]

    shoulder_right = frame[SHOULDER_RIGHT]
    elbow_right = frame[ELBOW_RIGHT]
    wrist_right = frame[WRIST_RIGHT]

    hip_left = frame[HIP_LEFT]
    hip_right = frame[HIP_RIGHT]

    # -------------------------------------------------
    # Arm angles
    # -------------------------------------------------

    left_elbow_angle = joint_angle(
        shoulder_left,
        elbow_left,
        wrist_left
    )

    right_elbow_angle = joint_angle(
        shoulder_right,
        elbow_right,
        wrist_right
    )

    # -------------------------------------------------
    # Shoulder width
    # -------------------------------------------------

    shoulder_width = distance(
        shoulder_left,
        shoulder_right
    )

    # -------------------------------------------------
    # Trunk orientation
    # -------------------------------------------------

    trunk_vector = vector(
        spine_base,
        spine_shoulder
    )

    trunk_vertical_angle = vector_angle(
        trunk_vector,
        np.array([0.0, 1.0, 0.0])
    )

    # -------------------------------------------------
    # Shoulder orientation
    # -------------------------------------------------

    shoulder_vector = vector(
        shoulder_left,
        shoulder_right
    )

    shoulder_x_angle = vector_angle(
        shoulder_vector,
        np.array([1.0, 0.0, 0.0])
    )

    # -------------------------------------------------
    # Shoulder elevation proxy
    # -------------------------------------------------

    left_shoulder_height = shoulder_left[1]
    right_shoulder_height = shoulder_right[1]

    shoulder_height_difference = (
        left_shoulder_height -
        right_shoulder_height
    )

    # -------------------------------------------------
    # Forward trunk displacement
    # -------------------------------------------------

    trunk_forward_displacement = (
        spine_shoulder[2] -
        spine_base[2]
    )

    # -------------------------------------------------
    # Hip / trunk geometry
    # -------------------------------------------------

    hip_center = (
        hip_left + hip_right
    ) / 2.0

    shoulder_center = (
        shoulder_left + shoulder_right
    ) / 2.0

    trunk_length = distance(
        hip_center,
        shoulder_center
    )

    # -------------------------------------------------
    # Feature dictionary
    # -------------------------------------------------

    return {
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,

        "shoulder_width": shoulder_width,

        "trunk_vertical_angle": trunk_vertical_angle,

        "shoulder_x_angle": shoulder_x_angle,

        "shoulder_height_difference":
            shoulder_height_difference,

        "trunk_forward_displacement":
            trunk_forward_displacement,

        "trunk_length":
            trunk_length,
    }