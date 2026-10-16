"""
Joint-angle measurements from MediaPipe Pose results.

Two families of measurements exist:

* image_plane_*: the ORIGINAL 2D measurement in normalized image
  coordinates. Kept unchanged for regression tests and for interpreting
  historical sessions recorded with it.

* World-landmark (3D) measurements using results.pose_world_landmarks
  (metres, hip-centred). Angles are expressed in a trunk-fixed frame
  built from the shoulders and hips, so they do not depend on where the
  person sits in the image or on the image aspect ratio.

Assumed MediaPipe world axes: x image-right, y down, z away from the
camera. The forward axis of the trunk frame depends on this assumption.
It has NOT been verified with a live webcam.
"""

import math

import numpy as np

from src.exercises.config import AngleReading

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

ARM_LANDMARKS = {
    "LEFT": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "RIGHT": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
}

TRUNK_LANDMARKS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)

LANDMARK_NAMES = {
    LEFT_SHOULDER: "left shoulder",
    RIGHT_SHOULDER: "right shoulder",
    LEFT_ELBOW: "left elbow",
    RIGHT_ELBOW: "right elbow",
    LEFT_WRIST: "left wrist",
    RIGHT_WRIST: "right wrist",
    LEFT_HIP: "left hip",
    RIGHT_HIP: "right hip",
}


def arm_landmarks(side):
    return ARM_LANDMARKS[side.upper()]


def arm_and_trunk_landmarks(side):
    shoulder, elbow, wrist = arm_landmarks(side)
    return tuple(dict.fromkeys(
        (shoulder, elbow, wrist) + TRUNK_LANDMARKS
    ))


# ---------------------------------------------------------
# Original 2D measurement
# ---------------------------------------------------------

def image_plane_arm_elevation(results, side):
    """
    ORIGINAL measurement: angle between the shoulder->wrist vector and
    straight down, in normalized image coordinates.

    Image x and y are normalized by different pixel dimensions, so the
    value depends on the camera aspect ratio, and forward raises are
    foreshortened. Retained for regression and historical comparison.
    """

    shoulder, _, wrist = arm_landmarks(side)
    landmarks = results.pose_landmarks.landmark

    dx = landmarks[wrist].x - landmarks[shoulder].x
    dy = landmarks[wrist].y - landmarks[shoulder].y

    length = (dx ** 2 + dy ** 2) ** 0.5

    if length <= 0:
        return AngleReading(0.0)

    cosine = max(-1.0, min(1.0, dy / length))

    return AngleReading(math.degrees(math.acos(cosine)))


def image_plane_curve(frames_2d, side):
    """Original 2D angle for each frame of a (n, 33, 2) sequence."""

    shoulder, _, wrist = arm_landmarks(side)
    curve = []

    for frame in frames_2d:
        dx = frame[wrist][0] - frame[shoulder][0]
        dy = frame[wrist][1] - frame[shoulder][1]
        length = (dx ** 2 + dy ** 2) ** 0.5

        if length > 0:
            cosine = max(-1.0, min(1.0, dy / length))
            curve.append(math.degrees(math.acos(cosine)))

    return curve


# ---------------------------------------------------------
# World-landmark helpers
# ---------------------------------------------------------

def world_array(results):
    """(33, 3) float array of world landmarks, or None if unavailable."""

    world = getattr(results, "pose_world_landmarks", None)

    if world is None:
        return None

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in world.landmark],
        dtype=np.float64,
    )

    if points.shape != (33, 3) or not np.all(np.isfinite(points)):
        return None

    return points


CAMERA_UP = np.array([0.0, -1.0, 0.0])


def hips_visible(results, threshold=0.5):
    landmarks = results.pose_landmarks.landmark
    return all(
        landmarks[i].visibility >= threshold for i in (LEFT_HIP, RIGHT_HIP)
    )


def trunk_frame(points, use_hips=True):
    """
    Orthonormal trunk frame from world points.

    Returns (lateral, up, forward) unit vectors or None if degenerate.
        lateral: towards the subject's LEFT shoulder
        up:      hip midpoint -> shoulder midpoint (or the camera
                 vertical when use_hips is False), orthogonal to lateral
        forward: the direction the chest faces
    """

    lateral = points[LEFT_SHOULDER] - points[RIGHT_SHOULDER]

    if use_hips:
        shoulder_mid = (points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]) / 2.0
        hip_mid = (points[LEFT_HIP] + points[RIGHT_HIP]) / 2.0
        up = shoulder_mid - hip_mid
    else:
        up = CAMERA_UP.copy()

    lateral_norm = np.linalg.norm(lateral)

    if lateral_norm < 1e-6:
        return None

    lateral = lateral / lateral_norm
    up = up - (up @ lateral) * lateral
    up_norm = np.linalg.norm(up)

    if up_norm < 1e-6:
        return None

    up = up / up_norm
    forward = np.cross(lateral, up)

    return lateral, up, forward


def humeral_elevation(points, side, use_hips=True):
    """
    Elevation of the upper arm (shoulder -> elbow) relative to the
    trunk, and its plane of elevation.

    elevation: 0 = arm hanging along the trunk, 90 = horizontal,
               180 = overhead.
    plane:     0 = arm moving forward (flexion), 90 = sideways
               (abduction), negative = behind the body. None when the
               arm is too close to the trunk for the plane to be stable.
    """

    frame = trunk_frame(points, use_hips)

    if frame is None:
        return None

    lateral, up, forward = frame
    shoulder, elbow, _ = arm_landmarks(side)

    arm = points[elbow] - points[shoulder]
    arm_norm = np.linalg.norm(arm)

    if arm_norm < 1e-6:
        return None

    arm = arm / arm_norm

    down_component = -(arm @ up)
    elevation = math.degrees(math.acos(max(-1.0, min(1.0, down_component))))

    outward = lateral if side.upper() == "LEFT" else -lateral
    forward_component = arm @ forward
    outward_component = arm @ outward

    horizontal = math.hypot(forward_component, outward_component)

    plane = None
    if horizontal > math.sin(math.radians(15.0)):
        plane = math.degrees(math.atan2(outward_component, forward_component))

    return elevation, plane


# ---------------------------------------------------------
# World-landmark measurements used by exercise configs
# ---------------------------------------------------------

def world_humeral_elevation(results, side):
    points = world_array(results)

    if points is None:
        return None

    use_hips = hips_visible(results)
    measurement = humeral_elevation(points, side, use_hips)

    if measurement is None:
        return None

    elevation, plane = measurement
    return AngleReading(
        elevation, plane, "hips" if use_hips else "camera"
    )


def world_elbow_flexion(results, side):
    """
    Elbow flexion in degrees: 0 = straight arm, increasing as the elbow
    bends (180 minus the shoulder-elbow-wrist angle).
    """

    points = world_array(results)

    if points is None:
        return None

    shoulder, elbow, wrist = arm_landmarks(side)

    upper = points[shoulder] - points[elbow]
    fore = points[wrist] - points[elbow]

    norms = np.linalg.norm(upper) * np.linalg.norm(fore)

    if norms < 1e-9:
        return None

    cosine = max(-1.0, min(1.0, float(upper @ fore / norms)))
    interior = math.degrees(math.acos(cosine))

    return AngleReading(180.0 - interior)


def world_forward_reach(results, side):
    """
    Forward reach as a percentage of arm length: the wrist's
    displacement in front of the shoulder along the trunk's forward
    axis, divided by (upper-arm + forearm) length.

    Depends heavily on MediaPipe's depth estimate.
    """

    points = world_array(results)

    if points is None:
        return None

    use_hips = hips_visible(results)
    frame = trunk_frame(points, use_hips)

    if frame is None:
        return None

    _, _, forward = frame
    shoulder, elbow, wrist = arm_landmarks(side)

    arm_length = (
        np.linalg.norm(points[elbow] - points[shoulder])
        + np.linalg.norm(points[wrist] - points[elbow])
    )

    if arm_length < 1e-6:
        return None

    reach = (points[wrist] - points[shoulder]) @ forward

    return AngleReading(
        100.0 * float(reach) / float(arm_length),
        None,
        "hips" if use_hips else "camera",
    )
