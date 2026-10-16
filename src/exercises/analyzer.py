"""
Generic, Qt-free exercise analysis.

Consumes one MediaPipe Pose result per frame and maintains repetition
counting, movement segmentation, range of motion, timing/data-quality
counters, and the DTW-based comparison against the exercise reference.

The counting and segmentation state machine reproduces the original
shoulder-raise implementation exactly when given the legacy config
(see tests/test_characterization_shoulder_raise.py).
"""

import math
from dataclasses import dataclass
from typing import Optional

from src.exercises.angles import LANDMARK_NAMES
from src.quality_assessment.dtw import dtw_similarity

STATUS_NO_POSE = "no_pose"
STATUS_NOT_VISIBLE = "not_visible"
STATUS_UNMEASURABLE = "unmeasurable"
STATUS_OK = "ok"


@dataclass
class FrameAnalysis:
    status: str
    angle: Optional[float] = None
    plane: Optional[float] = None
    reps: int = 0
    movement_active: bool = False


def velocity_peak_count(signal):
    """
    Number of local maxima in absolute angular velocity after a 3-frame
    moving average. A smooth raise-and-lower has about 2; jerky movement
    has more. Logged as an exploratory smoothness indicator only.
    """

    if len(signal) < 5:
        return None

    smoothed = [
        sum(signal[max(0, i - 1):i + 2]) / len(signal[max(0, i - 1):i + 2])
        for i in range(len(signal))
    ]
    speed = [
        abs(smoothed[i + 1] - smoothed[i])
        for i in range(len(smoothed) - 1)
    ]

    floor = 0.1 * max(speed) if speed else 0.0

    return sum(
        1
        for i in range(1, len(speed) - 1)
        if speed[i] > speed[i - 1]
        and speed[i] >= speed[i + 1]
        and speed[i] > floor
    )


class ExerciseAnalyzer:

    def __init__(self, config, side, parameters=None):
        self.config = config
        self.side = side.upper()

        if self.side not in config.sides:
            raise ValueError(
                f"{config.display_name} does not support side {side!r}"
            )

        self.parameters = config.resolve_parameters(parameters)

        self.up_threshold = config.rep.up_threshold
        if config.rep_threshold_parameter:
            self.up_threshold = float(
                self.parameters[config.rep_threshold_parameter]
            )

        self.required = tuple(config.required_landmarks(self.side))

        self._reference = None
        self.reset()

    # -----------------------------------------------------
    # State
    # -----------------------------------------------------

    def reset(self):
        self.reps = 0
        self.off_plane_reps = 0
        self.stage = "down"
        self.up_frames = 0
        self.down_frames = 0
        self._up_plane_votes = []
        self._rep_in_plane = True

        self.trajectory = []
        self.movement_segments = []
        self.current_segment = []
        self._segment_planes = []
        self._current_planes = []
        self._current_active_frames = 0
        self.movement_active = False
        self.return_frames = 0

        self.current_angle = 0
        self.current_plane = None
        self.max_angle = 0
        self.min_angle = 180

        self.frames_total = 0
        self.frames_with_pose = 0
        self.frames_measured = 0
        self.frames_camera_vertical = 0
        self.frame_errors = 0
        # Tracking diagnostics: why frames were not measured.
        self.frames_not_visible = 0
        self.frames_unmeasurable = 0
        self.not_visible_by_landmark = {}
        self.last_status = None
        self.first_timestamp = None
        self.last_timestamp = None

    @property
    def reference(self):
        if self._reference is None:
            self._reference = self.config.reference_fn(
                self.side, self.parameters
            )
        return self._reference

    # -----------------------------------------------------
    # Per-frame update
    # -----------------------------------------------------

    def record_error(self):
        """Count a frame that could not be processed."""
        self.frames_total += 1
        self.frame_errors += 1
        self.last_status = None

    @property
    def mean_fps(self):
        """
        Processed frames per second since the first frame: the value the
        adaptive policy's frame-rate gate uses (session summary
        "mean_fps"). None until two timestamped frames exist.
        """
        if (
            self.first_timestamp is None
            or self.last_timestamp is None
            or self.last_timestamp <= self.first_timestamp
            or self.frames_total <= 1
        ):
            return None
        return (self.frames_total - 1) / (
            self.last_timestamp - self.first_timestamp
        )

    @property
    def measured_fraction(self):
        if not self.frames_total:
            return None
        return self.frames_measured / self.frames_total

    def tracking_summary(self):
        """
        Where the unmeasured frames went. Every processed frame is in
        exactly one of: measured, no pose, required landmark not
        visible, unmeasurable, processing error.
        """
        return {
            "frames_total": self.frames_total,
            "frames_measured": self.frames_measured,
            "frames_no_pose": self.frames_total - self.frames_with_pose
            - self.frame_errors,
            "frames_not_visible": self.frames_not_visible,
            "frames_unmeasurable": self.frames_unmeasurable,
            "frame_errors": self.frame_errors,
            "measured_fraction": self.measured_fraction,
            "visibility_threshold": self.config.visibility_threshold,
            "required_landmarks": [
                LANDMARK_NAMES.get(i, str(i)) for i in self.required
            ],
            # Frames in which each required landmark was at or below
            # the threshold (a frame can count for several landmarks).
            "not_visible_by_landmark": {
                LANDMARK_NAMES.get(i, str(i)): n
                for i, n in sorted(self.not_visible_by_landmark.items(),
                                   key=lambda item: -item[1])
            },
        }

    def update(self, results, timestamp=None):
        self.frames_total += 1

        if timestamp is not None:
            if self.first_timestamp is None:
                self.first_timestamp = timestamp
            self.last_timestamp = timestamp

        if results is None or results.pose_landmarks is None:
            return self._frame(STATUS_NO_POSE)

        self.frames_with_pose += 1

        landmarks = results.pose_landmarks.landmark
        hidden = [
            i for i in self.required
            if not landmarks[i].visibility > self.config.visibility_threshold
        ]

        if hidden:
            self.frames_not_visible += 1
            for i in hidden:
                self.not_visible_by_landmark[i] = (
                    self.not_visible_by_landmark.get(i, 0) + 1
                )
            return self._frame(STATUS_NOT_VISIBLE)

        reading = self.config.angle_fn(results, self.side)

        if reading is None or not math.isfinite(reading.value):
            self.frames_unmeasurable += 1
            return self._frame(STATUS_UNMEASURABLE)

        self.frames_measured += 1
        if reading.reference_frame == "camera":
            self.frames_camera_vertical += 1
        angle = float(reading.value)
        plane = reading.plane

        self._update_segmentation(angle, plane)

        self.current_angle = angle
        self.current_plane = plane
        self.max_angle = max(self.max_angle, angle)
        self.min_angle = min(self.min_angle, angle)

        self._update_reps(angle, plane)

        return self._frame(STATUS_OK, angle, plane)

    def _frame(self, status, angle=None, plane=None):
        self.last_status = status
        return FrameAnalysis(
            status=status,
            angle=angle,
            plane=plane,
            reps=self.reps,
            movement_active=self.movement_active,
        )

    def _update_segmentation(self, angle, plane):
        seg = self.config.segmentation

        if angle >= seg.start_threshold:
            self.movement_active = True

        if not self.movement_active:
            return

        self.current_segment.append(angle)
        self._current_planes.append(plane)
        self.trajectory.append(angle)

        if angle >= seg.start_threshold:
            self._current_active_frames += 1

        if angle <= seg.end_threshold:
            self.return_frames += 1

            if self.return_frames >= seg.end_frames:
                if self._segment_is_kept(
                    self.current_segment, self._current_active_frames
                ):
                    self.movement_segments.append(self.current_segment.copy())
                    self._segment_planes.append(list(self._current_planes))

                self.current_segment = []
                self._current_planes = []
                self._current_active_frames = 0
                self.movement_active = False
                self.return_frames = 0
        else:
            self.return_frames = 0

    def _segment_is_kept(self, segment, active_frames):
        seg = self.config.segmentation
        return (
            len(segment) >= seg.min_segment_frames
            and active_frames >= seg.min_active_frames
        )

    def _plane_ok(self, plane):
        check = self.config.plane_check
        if check is None or plane is None:
            return None
        return abs(plane - check.expected_plane) <= check.tolerance

    def _update_reps(self, angle, plane):
        rep = self.config.rep

        if angle >= self.up_threshold:
            self.up_frames += 1
            self.down_frames = 0
            self._up_plane_votes.append(self._plane_ok(plane))

            if (
                self.up_frames >= rep.debounce_frames
                and self.stage == "down"
            ):
                self.stage = "up"
                votes = [v for v in self._up_plane_votes if v is not None]
                # Unknown plane (no plane check, or not measurable)
                # does not block the repetition.
                self._rep_in_plane = (
                    not votes or sum(votes) * 2 >= len(votes)
                )

        elif angle <= rep.down_threshold:
            self.down_frames += 1
            self.up_frames = 0
            self._up_plane_votes = []

            if (
                self.down_frames >= rep.debounce_frames
                and self.stage == "up"
            ):
                self.stage = "down"
                if self._rep_in_plane:
                    self.reps += 1
                else:
                    self.off_plane_reps += 1

    # -----------------------------------------------------
    # Session summary
    # -----------------------------------------------------

    def _scored_segments(self):
        """Completed segments plus an unfinished one at session end."""

        segments = [
            (segment.copy(), planes)
            for segment, planes in zip(
                self.movement_segments, self._segment_planes
            )
        ]

        if self.movement_active and self._segment_is_kept(
            self.current_segment, self._current_active_frames
        ):
            segments.append(
                (self.current_segment.copy(), list(self._current_planes))
            )

        return segments

    def summary(self):
        reference_curve = list(self.reference.angle_curve)
        reference_peak = max(reference_curve, default=0.0)
        weight = self.config.dtw.range_weight
        seg_config = self.config.segmentation

        segment_details = []
        # Averaged unrounded, then rounded once (original behaviour).
        raw_scores = []

        for segment, planes in self._scored_segments():
            if len(segment) < seg_config.min_segment_frames:
                continue

            movement_range = max(segment) - min(segment)
            patient_peak = max(segment)

            if movement_range < seg_config.min_range:
                continue

            dtw_score = dtw_similarity(reference_curve, segment)

            range_completion = (
                min(1.0, patient_peak / reference_peak)
                if reference_peak > 0
                else 0.0
            )

            score = (
                (1.0 - weight) * dtw_score
                + weight * range_completion * 100.0
            )

            score = max(0.0, min(100.0, score))
            raw_scores.append(score)

            peak_index = segment.index(patient_peak)
            peak_plane = planes[peak_index] if planes else None

            segment_details.append({
                "frames": len(segment),
                "peak": round(patient_peak, 2),
                "range": round(movement_range, 2),
                "dtw_similarity": dtw_score,
                "range_completion": round(range_completion, 4),
                "score": round(score, 2),
                "plane_at_peak": (
                    None if peak_plane is None else round(peak_plane, 1)
                ),
                "velocity_peaks": velocity_peak_count(segment),
            })

        trajectory_similarity = (
            round(sum(raw_scores) / len(raw_scores), 2)
            if raw_scores else None
        )

        mean_fps = self.mean_fps
        duration = (
            self.last_timestamp - self.first_timestamp
            if mean_fps is not None else None
        )

        smoothness_values = [
            d["velocity_peaks"] for d in segment_details
            if d["velocity_peaks"] is not None
        ]

        return {
            "exercise_id": self.config.exercise_id,
            "side": self.side,
            "parameters": dict(self.parameters),
            "reps": self.reps,
            "off_plane_reps": self.off_plane_reps,
            "target_reps": self.parameters.get("target_reps"),
            "rep_threshold": self.up_threshold,
            "max_angle": self.max_angle if self.frames_measured else None,
            "min_angle": self.min_angle if self.frames_measured else None,
            "trajectory": list(self.trajectory),
            # None (not 0) when no scorable movement occurred.
            "trajectory_similarity": trajectory_similarity,
            "valid_segments": len(segment_details),
            "segments": segment_details,
            "mean_velocity_peaks": (
                round(sum(smoothness_values) / len(smoothness_values), 2)
                if smoothness_values else None
            ),
            "frames_total": self.frames_total,
            "frames_with_pose": self.frames_with_pose,
            "frames_measured": self.frames_measured,
            "frames_camera_vertical": self.frames_camera_vertical,
            "frame_errors": self.frame_errors,
            "duration_s": duration,
            "mean_fps": mean_fps,
            "reference_synthetic": self.reference.synthetic,
            "tracking": self.tracking_summary(),
        }
