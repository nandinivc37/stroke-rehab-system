"""
Exercise catalogue: the single source of truth for which exercises
exist, how they are analysed and which parameters may be adapted.

Status of every exercise (see EXERCISE_STATUS / README):
  implemented + offline-tested with synthetic poses;
  webcam NOT TESTED; thresholds NOT clinically validated.

All thresholds are engineering defaults for webcam tracking and must be
reviewed by a therapist before clinical use.
"""

import math

from src.exercises.angles import (
    arm_landmarks,
    world_elbow_flexion,
    world_forward_reach,
    world_humeral_elevation,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
)
from src.exercises.config import (
    AdaptiveParameter,
    DtwConfig,
    ExerciseConfig,
    PlaneCheckConfig,
    RepCountingConfig,
    SegmentationConfig,
)
from src.exercises.reference import arm_pose, synthetic_reference

IMPAIRMENTS = (
    "Reduced ROM",
    "Reduced strength",
    "Poor movement coordination",
)


def _arm_and_shoulders(side):
    shoulder, elbow, wrist = arm_landmarks(side)
    return tuple(dict.fromkeys(
        (shoulder, elbow, wrist, LEFT_SHOULDER, RIGHT_SHOULDER)
    ))


def _upper_arm_and_shoulders(side):
    """
    Landmarks used by humeral elevation (shoulder -> elbow in a frame
    built from both shoulders). The wrist is not used by the angle, so a
    wrist outside the image (e.g. hanging below a waist-height crop at
    rest) must not reject the frame.
    """
    shoulder, elbow, _ = arm_landmarks(side)
    return tuple(dict.fromkeys(
        (shoulder, elbow, LEFT_SHOULDER, RIGHT_SHOULDER)
    ))


# Metrics v2 measurement settings shared by every catalogue exercise
# (see RepCountingConfig / DtwConfig). Engineering defaults, not
# clinically validated.
REP_MIN_DWELL_S = 0.13
REP_MAX_GAP_S = 1.0
METRICS_V2_DTW = DtwConfig(normalization="path")
# Glitch filter: generous upper bounds on how fast the measured value
# can change between consecutive measured frames. Chosen well above
# voluntary rehabilitation movement speeds so that only tracking
# artefacts (e.g. left/right landmark swaps) are rejected.
MAX_ANGULAR_SPEED = 1000.0  # degrees per second
MAX_REACH_SPEED = 500.0     # % of arm length per second


def _rep_config(up_threshold, down_threshold):
    return RepCountingConfig(
        up_threshold=up_threshold,
        down_threshold=down_threshold,
        min_dwell_s=REP_MIN_DWELL_S,
        max_gap_s=REP_MAX_GAP_S,
    )


def _target_reps(default=10):
    return AdaptiveParameter(
        name="target_reps", label="Target repetitions", unit="reps",
        default=default, minimum=3, maximum=30, step=2, integer=True,
    )


# ---------------------------------------------------------
# Shoulder flexion (forward raise)
# ---------------------------------------------------------

def _flexion_reference(side, parameters):
    return synthetic_reference(
        pose_fn=lambda s, v: arm_pose(s, flexion=v),
        angle_fn=world_humeral_elevation,
        side=side,
        peak=parameters["target_range"],
        view="side",
    )


SHOULDER_FLEXION = ExerciseConfig(
    exercise_id="shoulder_flexion",
    display_name="Shoulder Flexion",
    joint="Shoulder",
    body_region="Upper Limb",
    supported_impairments=IMPAIRMENTS,
    angle_label="Shoulder elevation",
    angle_unit="°",
    required_landmarks=_upper_arm_and_shoulders,
    angle_fn=world_humeral_elevation,
    reference_fn=_flexion_reference,
    rep=_rep_config(80.0, 30.0),
    dtw=METRICS_V2_DTW,
    metrics_version=2,
    max_speed=MAX_ANGULAR_SPEED,
    segmentation=SegmentationConfig(
        start_threshold=25.0, end_threshold=20.0, end_frames=5,
        min_segment_frames=6, min_range=12.0, min_active_frames=5,
    ),
    plane_check=PlaneCheckConfig(expected_plane=0.0, tolerance=30.0),
    adaptive_parameters=(
        _target_reps(),
        AdaptiveParameter(
            name="target_range", label="Target shoulder elevation",
            unit="°", default=80.0, minimum=50.0, maximum=150.0, step=5.0,
        ),
    ),
    rep_threshold_parameter="target_range",
    camera_view="side or 45° front-side recommended",
    reference_view="side",
    instructions=(
        "Raise the affected arm FORWARD and upward in front of the body, "
        "keeping the trunk still, then lower it slowly."
    ),
    notes=(
        "Measured as upper-arm elevation from MediaPipe 3D world "
        "landmarks. Forward raises depend on MediaPipe's depth estimate."
    ),
)


# ---------------------------------------------------------
# Shoulder abduction (sideways raise; the original reference movement)
# ---------------------------------------------------------

def _abduction_reference(side, parameters):
    return synthetic_reference(
        pose_fn=lambda s, v: arm_pose(s, abduction=v),
        angle_fn=world_humeral_elevation,
        side=side,
        peak=parameters["target_range"],
        view="front",
    )


SHOULDER_ABDUCTION = ExerciseConfig(
    exercise_id="shoulder_abduction",
    display_name="Shoulder Abduction",
    joint="Shoulder",
    body_region="Upper Limb",
    supported_impairments=IMPAIRMENTS,
    angle_label="Shoulder elevation",
    angle_unit="°",
    required_landmarks=_upper_arm_and_shoulders,
    angle_fn=world_humeral_elevation,
    reference_fn=_abduction_reference,
    rep=_rep_config(80.0, 30.0),
    dtw=METRICS_V2_DTW,
    metrics_version=2,
    max_speed=MAX_ANGULAR_SPEED,
    segmentation=SegmentationConfig(
        start_threshold=25.0, end_threshold=20.0, end_frames=5,
        min_segment_frames=6, min_range=12.0, min_active_frames=5,
    ),
    plane_check=PlaneCheckConfig(expected_plane=90.0, tolerance=30.0),
    adaptive_parameters=(
        _target_reps(),
        AdaptiveParameter(
            name="target_range", label="Target shoulder elevation",
            unit="°", default=80.0, minimum=50.0, maximum=150.0, step=5.0,
        ),
    ),
    rep_threshold_parameter="target_range",
    camera_view="frontal",
    reference_view="front",
    instructions=(
        "Raise the affected arm out to the SIDE and upward, keeping the "
        "trunk upright, then lower it slowly."
    ),
    notes=(
        "Same sideways raise as the original reference animation, now "
        "measured with MediaPipe 3D world landmarks."
    ),
)


# ---------------------------------------------------------
# Elbow flexion
# ---------------------------------------------------------

def _elbow_reference(side, parameters):
    return synthetic_reference(
        pose_fn=lambda s, v: arm_pose(s, elbow_flexion=v),
        angle_fn=world_elbow_flexion,
        side=side,
        peak=parameters["target_range"],
        view="side",
    )


ELBOW_FLEXION = ExerciseConfig(
    exercise_id="elbow_flexion",
    display_name="Elbow Flexion",
    joint="Elbow",
    body_region="Upper Limb",
    supported_impairments=IMPAIRMENTS,
    angle_label="Elbow flexion",
    angle_unit="°",
    required_landmarks=lambda side: arm_landmarks(side),
    angle_fn=world_elbow_flexion,
    reference_fn=_elbow_reference,
    rep=_rep_config(90.0, 30.0),
    dtw=METRICS_V2_DTW,
    metrics_version=2,
    max_speed=MAX_ANGULAR_SPEED,
    segmentation=SegmentationConfig(
        start_threshold=25.0, end_threshold=20.0, end_frames=5,
        min_segment_frames=6, min_range=12.0, min_active_frames=5,
    ),
    adaptive_parameters=(
        _target_reps(),
        AdaptiveParameter(
            name="target_range", label="Target elbow flexion",
            unit="°", default=90.0, minimum=60.0, maximum=140.0, step=5.0,
        ),
    ),
    rep_threshold_parameter="target_range",
    camera_view="frontal or side",
    reference_view="side",
    instructions=(
        "Start with the affected arm straight by your side. Bend the "
        "elbow to bring the hand towards the shoulder, then straighten."
    ),
    notes=(
        "3D shoulder-elbow-wrist angle. Starting from a straight arm is "
        "required: a forearm resting on a table starts near 90°."
    ),
)


# ---------------------------------------------------------
# Forward reach
# ---------------------------------------------------------

def _reach_pose(side, reach_percent):
    # A straight arm raised forward by angle a reaches sin(a) of its
    # length in front of the shoulder.
    fraction = max(0.0, min(0.99, reach_percent / 100.0))
    return arm_pose(side, flexion=math.degrees(math.asin(fraction)))


def _reach_reference(side, parameters):
    return synthetic_reference(
        pose_fn=_reach_pose,
        angle_fn=world_forward_reach,
        side=side,
        peak=parameters["target_range"],
        view="side",
    )


FORWARD_REACH = ExerciseConfig(
    exercise_id="forward_reach",
    display_name="Forward Reach",
    joint="Shoulder",
    body_region="Upper Limb",
    supported_impairments=IMPAIRMENTS,
    angle_label="Reach",
    angle_unit="% arm length",
    required_landmarks=_arm_and_shoulders,
    angle_fn=world_forward_reach,
    reference_fn=_reach_reference,
    rep=_rep_config(60.0, 20.0),
    dtw=METRICS_V2_DTW,
    metrics_version=2,
    max_speed=MAX_REACH_SPEED,
    segmentation=SegmentationConfig(
        start_threshold=15.0, end_threshold=10.0, end_frames=5,
        min_segment_frames=6, min_range=10.0, min_active_frames=5,
    ),
    adaptive_parameters=(
        _target_reps(),
        AdaptiveParameter(
            name="target_range", label="Target reach",
            unit="% arm length", default=60.0, minimum=30.0, maximum=90.0,
            step=5.0,
        ),
    ),
    rep_threshold_parameter="target_range",
    camera_view="side recommended",
    reference_view="side",
    instructions=(
        "Reach the affected hand straight forward at shoulder height as "
        "far as comfortable WITHOUT leaning the trunk, then return."
    ),
    notes=(
        "Depends almost entirely on MediaPipe's depth estimate; the "
        "highest-risk exercise for webcam accuracy. Trunk lean also "
        "increases reach and is only flagged by the experimental posture "
        "measurements."
    ),
)


# ---------------------------------------------------------
# Catalogue access
# ---------------------------------------------------------

# Ordered by implementation priority.
CATALOGUE = (
    SHOULDER_FLEXION,
    SHOULDER_ABDUCTION,
    ELBOW_FLEXION,
    FORWARD_REACH,
)

_BY_ID = {config.exercise_id: config for config in CATALOGUE}
_BY_NAME = {config.display_name.lower(): config for config in CATALOGUE}

# Historical names written by earlier versions of the application.
_LEGACY_NAMES = {
    "elbow flexion-extension": ELBOW_FLEXION,
}


def implemented_exercises():
    return [config for config in CATALOGUE if config.implemented]


def get_exercise_config(identifier):
    """
    Look up by exercise_id or display name (case-insensitive).
    Returns None when the exercise is not in the catalogue.
    """

    if not identifier:
        return None

    key = str(identifier).strip()
    return (
        _BY_ID.get(key)
        or _BY_NAME.get(key.lower())
        or _LEGACY_NAMES.get(key.lower())
    )
