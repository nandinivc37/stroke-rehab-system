"""
Declarative exercise configuration.

Each exercise is described by one ExerciseConfig. The generic
ExerciseAnalyzer reads it to decide which joints to require, how to
measure the movement angle, how to count repetitions and segment
attempts, which reference trajectory to compare against with DTW, and
which parameters the adaptive policy is allowed to change.

Threshold values in configs are engineering defaults chosen for
webcam-based tracking. They are NOT clinically validated.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence


@dataclass(frozen=True)
class AngleReading:
    """One frame's movement measurement."""

    value: float
    # Plane of elevation in degrees (0 = forward / sagittal,
    # 90 = sideways / frontal). None when not applicable or not
    # measurable (e.g. arm hanging down).
    plane: Optional[float] = None
    # "hips": trunk axis from hips to shoulders; "camera": hips not
    # visible, camera vertical used instead (assumes an upright camera).
    reference_frame: Optional[str] = None
    # Why the plane is (not) known, for measurements that report it
    # (see angles.PLANE_*). None when the measurement does not say.
    plane_status: Optional[str] = None


@dataclass(frozen=True)
class RepCountingConfig:
    up_threshold: float
    down_threshold: float
    debounce_frames: int = 5
    # Timestamp-based debounce (metrics v2). When set and frames carry
    # timestamps, a stage change needs the angle beyond the threshold
    # for at least this long (and at least min_debounce_frames frames)
    # instead of a fixed number of frames, so counting does not depend
    # on the frame rate. 0.13 s equals the original 5-frame criterion
    # at 30 fps (4 frame intervals). None keeps the frame-count rule.
    min_dwell_s: Optional[float] = None
    min_debounce_frames: int = 2
    # A gap longer than this between measured frames (tracking lost)
    # abandons a repetition in progress instead of crediting a movement
    # that was not observed. None disables the check.
    max_gap_s: Optional[float] = None


@dataclass(frozen=True)
class SegmentationConfig:
    """
    A movement attempt starts when the angle reaches start_threshold and
    ends after end_frames consecutive frames at or below end_threshold.
    """

    start_threshold: float
    end_threshold: float
    end_frames: int = 5
    # Segments shorter than this are discarded (original rule: > 5).
    min_segment_frames: int = 6
    # Segments whose angle range is smaller than this are not scored.
    min_range: float = 12.0
    # Frames at or above start_threshold required for a segment to be
    # kept. The original implementation effectively used 1, which let
    # single-frame landmark glitches become scored "attempts".
    min_active_frames: int = 1


@dataclass(frozen=True)
class DtwConfig:
    # Movement score = (1 - range_weight) * DTW similarity
    #                  + range_weight * range completion
    range_weight: float = 0.5
    # How the accumulated DTW cost is turned into a mean angular error:
    #   "n_plus_m": cost / (n + m), the original rule. n + m is an upper
    #       bound on the warping-path length, so the error is understated
    #       (by up to ~2x) and the similarity overstated.
    #   "path": cost / number of steps on the optimal warping path, i.e.
    #       the actual mean absolute difference between matched samples.
    normalization: str = "n_plus_m"


@dataclass(frozen=True)
class PlaneCheckConfig:
    """Movement-plane check for shoulder elevation exercises."""

    expected_plane: float
    tolerance: float
    # Plane is only meaningful once the arm is clearly elevated.
    min_elevation: float = 30.0
    # When True, a repetition or segment whose plane could not be
    # verified (AngleReading.plane_status in angles.PLANE_UNVERIFIED) is
    # not counted or scored, instead of being treated as in plane.
    require_known_plane: bool = False


@dataclass(frozen=True)
class AdaptiveParameter:
    name: str
    label: str
    unit: str
    default: float
    minimum: float
    maximum: float
    step: float
    integer: bool = False

    def clamp(self, value):
        value = max(self.minimum, min(self.maximum, value))
        return int(round(value)) if self.integer else float(value)


@dataclass(frozen=True)
class ReferenceMovement:
    """
    frames: (n, 33, 2) normalized 2D points for display.
    angle_curve: reference angle per frame, produced by the same
        angle function used for live analysis where possible.
    synthetic: True when generated from a model rather than recorded
        from a clinician demonstration.
    """

    frames: object
    angle_curve: Sequence[float]
    synthetic: bool = True


@dataclass(frozen=True)
class ExerciseConfig:
    exercise_id: str
    display_name: str
    joint: str
    body_region: str
    supported_impairments: Sequence[str]

    angle_label: str
    angle_unit: str

    # (side) -> landmark indices that must be visible.
    required_landmarks: Callable[[str], Sequence[int]]
    # (results, side) -> AngleReading or None.
    angle_fn: Callable[[object, str], Optional[AngleReading]]
    # (side, parameters) -> ReferenceMovement.
    reference_fn: Callable[[str, dict], ReferenceMovement]

    rep: RepCountingConfig
    segmentation: SegmentationConfig
    dtw: DtwConfig = field(default_factory=DtwConfig)
    plane_check: Optional[PlaneCheckConfig] = None

    adaptive_parameters: Sequence[AdaptiveParameter] = ()
    # Name of the adaptive parameter that replaces rep.up_threshold.
    rep_threshold_parameter: Optional[str] = None

    camera_view: str = "frontal"
    reference_view: str = "front"
    sides: Sequence[str] = ("LEFT", "RIGHT")
    visibility_threshold: float = 0.5

    # Measurement definitions (see ExerciseAnalyzer.summary):
    #   1 = original rules (raw single-frame peaks, every segment scored,
    #       smoothness on raw frames);
    #   2 = peak ROM from in-plane movements (3-frame median), off-plane
    #       segments not scored, smoothness on a 10 Hz time base.
    metrics_version: int = 1
    # Frames whose measured value changes faster than this (units of
    # angle_unit per second) since the previous measured frame are
    # rejected as tracking glitches (e.g. a left/right landmark swap).
    # None disables the check.
    max_speed: Optional[float] = None
    # () -> stateful (results, side) measurement. When set, each
    # analyzer session (and the reference) uses a fresh instance instead
    # of angle_fn, e.g. for per-session calibration.
    angle_fn_factory: Optional[Callable[[], Callable]] = None

    # Shown in the exercise catalogue / UI.
    instructions: str = ""
    implemented: bool = True
    webcam_validated: bool = False
    notes: str = ""

    def make_angle_fn(self):
        """The measurement to use for one session or reference."""
        if self.angle_fn_factory is not None:
            return self.angle_fn_factory()
        return self.angle_fn

    def parameter(self, name):
        for parameter in self.adaptive_parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(f"{self.exercise_id} has no parameter {name!r}")

    def default_parameters(self):
        return {p.name: p.default for p in self.adaptive_parameters}

    def resolve_parameters(self, parameters=None):
        """
        Merge stored parameters with defaults, clamping every value to
        its allowed range. Unknown keys are dropped.
        """

        resolved = self.default_parameters()

        for name, value in (parameters or {}).items():
            if name in resolved and value is not None:
                resolved[name] = self.parameter(name).clamp(float(value))

        return resolved
