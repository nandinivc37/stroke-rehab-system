"""
EXPERIMENTAL geometric posture measurements (method "geometric_v1").

These are continuous measurements of trunk and shoulder posture relative
to the participant's own resting baseline at the start of a session.
They are intended for research and therapist review only.

They are NOT validated compensation detections:
  * no labelled MediaPipe data has been used to check them;
  * trunk rotation and forward lean depend on MediaPipe's depth (z)
    estimate, which is known to be noisy;
  * camera tilt or a non-frontal camera position changes the values.

They must not be shown to patients as diagnoses and must not drive
adaptive difficulty decisions until validated.
"""

import math

import numpy as np

from src.exercises.angles import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    world_array,
)

METHOD = "geometric_v1"

MEASURES = {
    "shoulder_elevation": (
        "shoulder widths",
        "Affected shoulder height relative to the other shoulder",
    ),
    "trunk_lateral_lean": (
        "degrees",
        "Sideways tilt of the hip-to-shoulder line",
    ),
    "trunk_forward_lean": (
        "degrees",
        "Forward tilt of the hip-to-shoulder line (uses depth)",
    ),
    "trunk_rotation": (
        "degrees",
        "Shoulder-line rotation relative to the hip line (uses depth)",
    ),
}

REQUIRED_LANDMARKS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)


def raw_posture(points, side):
    """
    Raw posture values from (33, 3) world points in camera axes
    (x right, y down, z away from camera). Returns None if degenerate.
    """

    left_shoulder = points[LEFT_SHOULDER]
    right_shoulder = points[RIGHT_SHOULDER]
    left_hip = points[LEFT_HIP]
    right_hip = points[RIGHT_HIP]

    shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))

    if shoulder_width < 1e-6:
        return None

    if side.upper() == "LEFT":
        affected, other = left_shoulder, right_shoulder
    else:
        affected, other = right_shoulder, left_shoulder

    trunk = (left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2

    if np.linalg.norm(trunk) < 1e-6:
        return None

    # Height increases upwards (world y points down).
    elevation = (other[1] - affected[1]) / shoulder_width

    lateral_lean = math.degrees(math.atan2(trunk[0], -trunk[1]))
    forward_lean = math.degrees(math.atan2(-trunk[2], -trunk[1]))

    shoulder_line = left_shoulder - right_shoulder
    hip_line = left_hip - right_hip
    rotation = math.degrees(
        math.atan2(shoulder_line[2], shoulder_line[0])
        - math.atan2(hip_line[2], hip_line[0])
    )
    # Wrap to (-180, 180].
    rotation = (rotation + 180.0) % 360.0 - 180.0

    return {
        "shoulder_elevation": float(elevation),
        "trunk_lateral_lean": float(lateral_lean),
        "trunk_forward_lean": float(forward_lean),
        "trunk_rotation": float(rotation),
    }


class ExperimentalPostureMonitor:
    """
    Accumulates posture deviations from a per-session resting baseline.

    The baseline is the median of the first `baseline_frames` usable
    frames recorded while no movement attempt is active. Deviations are
    only accumulated during active movement attempts.
    """

    def __init__(self, side, baseline_frames=15, visibility_threshold=0.5):
        self.side = side.upper()
        self.baseline_frames = baseline_frames
        self.visibility_threshold = visibility_threshold
        self.reset()

    def reset(self):
        self._baseline_samples = []
        self.baseline = None
        self.deviations = {name: [] for name in MEASURES}
        self.frames_unavailable = 0
        self.frames_used = 0

    def update(self, results, movement_active):
        values = self._measure(results)

        if values is None:
            self.frames_unavailable += 1
            return None

        if self.baseline is None:
            if not movement_active:
                self._baseline_samples.append(values)
                if len(self._baseline_samples) >= self.baseline_frames:
                    self.baseline = {
                        name: float(np.median(
                            [s[name] for s in self._baseline_samples]
                        ))
                        for name in MEASURES
                    }
            return None

        deviation = {
            name: values[name] - self.baseline[name] for name in MEASURES
        }

        if movement_active:
            self.frames_used += 1
            for name, value in deviation.items():
                self.deviations[name].append(value)

        return deviation

    def _measure(self, results):
        if results is None or results.pose_landmarks is None:
            return None

        landmarks = results.pose_landmarks.landmark

        if not all(
            landmarks[i].visibility >= self.visibility_threshold
            for i in REQUIRED_LANDMARKS
        ):
            return None

        points = world_array(results)

        if points is None:
            return None

        values = raw_posture(points, self.side)

        if values is None or not all(
            math.isfinite(v) for v in values.values()
        ):
            return None

        return values

    def summary(self):
        """
        Session summary, or a summary with measures=None when no baseline
        or no movement frames were available. Never reports zeros for
        missing data.
        """

        base = {
            "method": METHOD,
            "experimental": True,
            "validated": False,
            "baseline_established": self.baseline is not None,
            "frames_used": self.frames_used,
            "frames_unavailable": self.frames_unavailable,
        }

        if self.baseline is None or self.frames_used == 0:
            base["measures"] = None
            return base

        measures = {}
        for name, (unit, description) in MEASURES.items():
            values = np.abs(np.asarray(self.deviations[name]))
            measures[name] = {
                "unit": unit,
                "description": description,
                "p95_abs_deviation": round(float(np.percentile(values, 95)), 4),
                "max_abs_deviation": round(float(values.max()), 4),
            }

        base["measures"] = measures
        return base
