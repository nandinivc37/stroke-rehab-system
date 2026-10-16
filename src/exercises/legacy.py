"""
The ORIGINAL shoulder-raise analysis, expressed as an ExerciseConfig.

Uses the 2D image-plane angle, the original 2D keyframe reference
(src/assessment/exercise_library.py) and the original thresholds.
It is not offered to patients; it exists so that regression tests can
prove the generic analyzer reproduces the original behaviour, and so
historical sessions can be interpreted.
"""

from src.assessment.exercise_library import create_shoulder_flexion_sequence
from src.exercises.angles import (
    arm_landmarks,
    image_plane_arm_elevation,
    image_plane_curve,
)
from src.exercises.config import (
    ExerciseConfig,
    ReferenceMovement,
    RepCountingConfig,
    SegmentationConfig,
)


def _legacy_reference(side, _parameters):
    frames = create_shoulder_flexion_sequence(affected_side=side)
    return ReferenceMovement(
        frames=frames,
        angle_curve=image_plane_curve(frames, side),
        synthetic=True,
    )


LEGACY_SHOULDER_RAISE_2D = ExerciseConfig(
    exercise_id="legacy_shoulder_raise_2d",
    display_name="Shoulder Raise (original 2D measure)",
    joint="Shoulder",
    body_region="Upper Limb",
    supported_impairments=(),
    angle_label="Arm angle",
    angle_unit="°",
    required_landmarks=arm_landmarks,
    angle_fn=image_plane_arm_elevation,
    reference_fn=_legacy_reference,
    rep=RepCountingConfig(up_threshold=80.0, down_threshold=45.0,
                          debounce_frames=5),
    segmentation=SegmentationConfig(
        start_threshold=12.0,
        end_threshold=10.0,
        end_frames=5,
        min_segment_frames=6,
        min_range=12.0,
        min_active_frames=1,
    ),
    implemented=False,
    notes="Original behaviour; regression and historical reference only.",
)
