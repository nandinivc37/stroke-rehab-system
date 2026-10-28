"""
MediaPipe Pose settings shared by the application and the Step B
recorder, so that both always use the same pose model.

model_complexity selects the landmark model: 0 = lite, 1 = full,
2 = heavy. Profiling (README, "Frame rate") showed the heavy model
cannot reach the application's 10 fps assessment gate on the project
laptops (live sessions ran at 5.7-9.3 fps), so the default is 1 (full).
The value is stored with every session (diagnostics_json) so sessions
recorded with different models can be told apart.

Override for a run with the environment variable REHAB_POSE_COMPLEXITY
(0, 1 or 2), e.g. to compare models during validation.
"""

import os

DEFAULT_MODEL_COMPLEXITY = 1


def model_complexity():
    value = os.environ.get("REHAB_POSE_COMPLEXITY")
    if value is None or value.strip() == "":
        return DEFAULT_MODEL_COMPLEXITY
    if value.strip() not in ("0", "1", "2"):
        raise ValueError(
            f"REHAB_POSE_COMPLEXITY must be 0, 1 or 2, not {value!r}"
        )
    return int(value)


def pose_settings():
    """Keyword arguments for mediapipe.solutions.pose.Pose."""
    return {
        "static_image_mode": False,
        "model_complexity": model_complexity(),
        "smooth_landmarks": True,
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.7,
    }
