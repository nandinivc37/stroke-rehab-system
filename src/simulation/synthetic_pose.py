"""
Synthetic MediaPipe-style pose results for offline tests and simulation.

This is NOT recorded data. Poses are generated from a simple rigid
seated-body model so that the analysis pipeline can be exercised without
a webcam. Nothing produced here may be used as ground-truth evidence of
real-world accuracy.

Coordinate convention (assumed to match MediaPipe Pose world landmarks;
NOT yet verified against a live webcam):
    x  -> image right (the subject's LEFT side when facing the camera)
    y  -> down
    z  -> away from the camera (the subject faces -z)
    origin at the hip midpoint, units in metres.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from mediapipe.framework.formats import landmark_pb2

NUM_LANDMARKS = 33

NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

UPPER_ARM_LENGTH = 0.30
FOREARM_LENGTH = 0.25

# Orthographic projection scale from metres to normalized image units.
IMAGE_SCALE = 0.6
IMAGE_CENTER = (0.5, 0.75)


@dataclass
class ArmPose:
    """Arm configuration in degrees, relative to the arm hanging down."""

    flexion: float = 0.0       # forward elevation in the sagittal plane
    abduction: float = 0.0     # sideways elevation in the frontal plane
    elbow_flexion: float = 0.0  # 0 = straight elbow


@dataclass
class BodyPose:
    left_arm: ArmPose = field(default_factory=ArmPose)
    right_arm: ArmPose = field(default_factory=ArmPose)

    # Trunk perturbations (degrees / metres), used to simulate
    # compensation-like movements in tests.
    trunk_forward_lean: float = 0.0
    trunk_lateral_lean: float = 0.0
    trunk_rotation: float = 0.0
    left_shoulder_raise: float = 0.0
    right_shoulder_raise: float = 0.0

    visibility: float = 0.99


def _rotation_x(degrees):
    a = math.radians(degrees)
    return np.array([
        [1, 0, 0],
        [0, math.cos(a), -math.sin(a)],
        [0, math.sin(a), math.cos(a)],
    ])


def _rotation_y(degrees):
    a = math.radians(degrees)
    return np.array([
        [math.cos(a), 0, math.sin(a)],
        [0, 1, 0],
        [-math.sin(a), 0, math.cos(a)],
    ])


def _rotation_z(degrees):
    a = math.radians(degrees)
    return np.array([
        [math.cos(a), -math.sin(a), 0],
        [math.sin(a), math.cos(a), 0],
        [0, 0, 1],
    ])


def _arm_direction(arm, outward_sign):
    """
    Unit vector of the upper arm in the trunk frame.

    Flexion rotates the hanging arm (0, 1, 0) towards the front (-z).
    Abduction rotates it towards the subject's side (outward_sign * x).
    Combined values are applied as abduction followed by flexion.
    """

    down = np.array([0.0, 1.0, 0.0])

    abduct = math.radians(arm.abduction)
    direction = np.array([
        outward_sign * math.sin(abduct),
        math.cos(abduct),
        0.0,
    ])

    if arm.flexion:
        flex = math.radians(arm.flexion)
        # Rotate within the plane spanned by "down" and "forward".
        forward = np.array([0.0, 0.0, -1.0])
        vertical = direction @ down
        lateral = direction - vertical * down
        direction = (
            lateral
            + vertical * (math.cos(flex) * down + math.sin(flex) * forward)
        )

    return direction / np.linalg.norm(direction)


def _forearm_direction(upper_direction, elbow_flexion):
    """
    Bend the forearm towards the front of the body by elbow_flexion.
    """

    if not elbow_flexion:
        return upper_direction

    forward = np.array([0.0, 0.0, -1.0])
    bend_axis = np.cross(upper_direction, forward)

    if np.linalg.norm(bend_axis) < 1e-6:
        # Upper arm points straight forward: bend upwards instead.
        bend_axis = np.array([1.0, 0.0, 0.0])

    bend_axis /= np.linalg.norm(bend_axis)

    a = math.radians(elbow_flexion)
    # Rodrigues rotation of upper_direction about bend_axis.
    v = upper_direction
    rotated = (
        v * math.cos(a)
        + np.cross(bend_axis, v) * math.sin(a)
        + bend_axis * (bend_axis @ v) * (1 - math.cos(a))
    )
    return rotated / np.linalg.norm(rotated)


def world_points(pose):
    """
    Return a (33, 3) array of world coordinates for a BodyPose.
    Landmarks not modelled are placed at the hip midpoint.
    """

    points = np.zeros((NUM_LANDMARKS, 3), dtype=np.float64)

    points[LEFT_HIP] = [0.10, 0.0, 0.0]
    points[RIGHT_HIP] = [-0.10, 0.0, 0.0]

    trunk = {
        LEFT_SHOULDER: np.array([0.18, -0.50, 0.0]),
        RIGHT_SHOULDER: np.array([-0.18, -0.50, 0.0]),
        NOSE: np.array([0.0, -0.75, -0.08]),
        LEFT_EAR: np.array([0.07, -0.73, 0.0]),
        RIGHT_EAR: np.array([-0.07, -0.73, 0.0]),
    }

    trunk[LEFT_SHOULDER][1] -= pose.left_shoulder_raise
    trunk[RIGHT_SHOULDER][1] -= pose.right_shoulder_raise

    # Lean forward rotates the trunk about the hip x-axis so the
    # shoulders move towards the camera (-z).
    rotation = (
        _rotation_y(pose.trunk_rotation)
        @ _rotation_z(pose.trunk_lateral_lean)
        @ _rotation_x(pose.trunk_forward_lean)
    )

    for index, point in trunk.items():
        points[index] = rotation @ point

    for shoulder, elbow, wrist, arm, outward in (
        (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, pose.left_arm, 1.0),
        (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, pose.right_arm, -1.0),
    ):
        upper = rotation @ _arm_direction(arm, outward)
        fore = _forearm_direction(upper, arm.elbow_flexion)

        points[elbow] = points[shoulder] + UPPER_ARM_LENGTH * upper
        points[wrist] = points[elbow] + FOREARM_LENGTH * fore

    return points


def project_to_image(points):
    """Orthographic projection of world points to normalized image coords."""

    image = np.empty_like(points)
    image[:, 0] = IMAGE_CENTER[0] + IMAGE_SCALE * points[:, 0]
    image[:, 1] = IMAGE_CENTER[1] + IMAGE_SCALE * points[:, 1]
    image[:, 2] = IMAGE_SCALE * points[:, 2]
    return image


class SyntheticResults:
    """Mimics the attributes of mediapipe Pose.process() results."""

    def __init__(self, pose_landmarks, pose_world_landmarks):
        self.pose_landmarks = pose_landmarks
        self.pose_world_landmarks = pose_world_landmarks


def make_results(pose=None, visibility_overrides=None, image_points=None):
    """
    Build a SyntheticResults object.

    Args:
        pose: BodyPose (default neutral seated pose)
        visibility_overrides: {landmark_index: visibility}
        image_points: optional (33, 3) normalized image coordinates
            overriding the orthographic projection (used by legacy
            2D characterization tests).
    """

    pose = pose or BodyPose()
    world = world_points(pose)
    image = project_to_image(world) if image_points is None else image_points

    overrides = visibility_overrides or {}

    normalized = landmark_pb2.NormalizedLandmarkList()
    metric = landmark_pb2.LandmarkList()

    for index in range(NUM_LANDMARKS):
        visibility = overrides.get(index, pose.visibility)

        normalized.landmark.add(
            x=float(image[index, 0]),
            y=float(image[index, 1]),
            z=float(image[index, 2]),
            visibility=float(visibility),
        )
        metric.landmark.add(
            x=float(world[index, 0]),
            y=float(world[index, 1]),
            z=float(world[index, 2]),
            visibility=float(visibility),
        )

    return SyntheticResults(normalized, metric)


def empty_results():
    """Results object for a frame in which no person was detected."""
    return SyntheticResults(None, None)


class FakeCamera:
    """Minimal cv2.VideoCapture stand-in that yields blank frames."""

    def __init__(self, frames, width=320, height=240):
        self._remaining = frames
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)

    def read(self):
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, self._frame.copy()

    def isOpened(self):
        return True

    def release(self):
        self._remaining = 0


class ScriptedPose:
    """Stand-in for mediapipe Pose that replays a list of results."""

    def __init__(self, results):
        self._results = list(results)
        self._index = 0

    def process(self, _rgb):
        result = self._results[self._index]
        self._index += 1
        return result

    def close(self):
        pass
