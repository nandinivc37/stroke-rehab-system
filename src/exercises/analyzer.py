"""
Generic, Qt-free exercise analysis.

Consumes one MediaPipe Pose result per frame and maintains repetition
counting, movement segmentation, range of motion, timing/data-quality
counters, a per-frame diagnostic log, and the DTW-based comparison
against the exercise reference.

The counting and segmentation state machine reproduces the original
shoulder-raise implementation exactly when given the legacy config
(see tests/test_characterization_shoulder_raise.py).

Metrics version 2 (catalogue exercises, ExerciseConfig.metrics_version)
changes only what the legacy config does not use:
  * repetition debounce by time spent beyond a threshold
    (RepCountingConfig.min_dwell_s) instead of a frame count, and
    abandonment of a repetition interrupted by lost tracking;
  * a glitch filter on implausibly fast changes (ExerciseConfig.max_speed);
  * peak ROM from in-plane movements using a 3-frame median, instead of
    the raw maximum over the whole session;
  * off-plane segments are reported but not scored against the reference;
  * with PlaneCheckConfig.require_known_plane, repetitions and segments
    whose plane could not be verified are treated like off-plane ones
    (counted in off_plane_reps, broken out as plane_unverified_reps);
  * DTW error normalised by warping-path length (DtwConfig.normalization);
  * the smoothness indicator computed on a 10 Hz time base.
"""

import math
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Optional

from src.exercises.angles import LANDMARK_NAMES, PLANE_UNVERIFIED
from src.quality_assessment.dtw import dtw_distance, similarity_from_error

STATUS_NO_POSE = "no_pose"
STATUS_NOT_VISIBLE = "not_visible"
STATUS_UNMEASURABLE = "unmeasurable"
STATUS_IMPLAUSIBLE = "implausible"
STATUS_ERROR = "error"
STATUS_OK = "ok"

# Time base for the v2 smoothness indicator.
SMOOTHNESS_RESAMPLE_HZ = 10.0


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


def resample(values, times, rate_hz):
    """
    Linear interpolation of (times, values) onto a uniform time base.
    Returns the input unchanged when timestamps are missing or unusable.
    """

    if (
        len(values) < 2
        or len(times) != len(values)
        or any(t is None for t in times)
        or times[-1] <= times[0]
    ):
        return list(values)

    step = 1.0 / rate_hz
    count = int(math.floor((times[-1] - times[0]) / step + 1e-6)) + 1
    out = []
    k = 0
    for index in range(count):
        t = times[0] + index * step
        while k + 1 < len(times) - 1 and times[k + 1] < t:
            k += 1
        t0, t1 = times[k], times[k + 1]
        if t1 <= t0:
            out.append(values[k + 1])
        else:
            alpha = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
            out.append(values[k] * (1 - alpha) + values[k + 1] * alpha)
    return out


def rolling_median_peak(values, width=3):
    """Highest median of `width` consecutive values (single-frame
    spikes cannot set the peak). Falls back to the median of all values
    for shorter sequences."""

    if not values:
        return None
    if len(values) < width:
        return float(median(values))
    return float(max(
        median(values[i:i + width]) for i in range(len(values) - width + 1)
    ))


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
        self.v2 = config.metrics_version >= 2

        self._reference = None
        self.reset()

    # -----------------------------------------------------
    # State
    # -----------------------------------------------------

    def reset(self):
        # Fresh measurement per session (stateful ones calibrate).
        self.angle_fn = self.config.make_angle_fn()
        self.reps = 0
        self.off_plane_reps = 0
        self.plane_unverified_reps = 0
        self.interrupted_reps = 0
        self.stage = "down"
        self.up_frames = 0
        self.down_frames = 0
        self._up_dwell = 0.0
        self._down_dwell = 0.0
        self._prev_zone = None
        self._prev_zone_t = None
        self._up_plane_votes = []
        self._up_unverified = 0
        self._rep_in_plane = True
        self._rep_plane_verified = True
        self._excursion_peak = None
        self.rep_peaks = []          # [(robust peak, in plane)]
        self._recent = deque(maxlen=3)

        self.trajectory = []
        self.movement_segments = []
        self.current_segment = []
        self._segment_planes = []
        self._current_planes = []
        self._segment_unverified = []
        self._current_unverified = []
        self._segment_times = []
        self._current_times = []
        self._current_active_frames = 0
        self.movement_active = False
        self.return_frames = 0

        self.current_angle = 0
        self.current_plane = None
        self.current_plane_status = None
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
        self.frames_implausible = 0
        self.not_visible_by_landmark = {}
        self.last_status = None
        self.first_timestamp = None
        self.last_timestamp = None
        self._last_measured_t = None
        self._last_raw = None        # (timestamp, value) of last reading
        # Per-frame diagnostics:
        # (t_s, status, angle, plane, min_required_visibility, frame)
        self.frame_log = []

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

    def _relative_time(self, timestamp):
        if timestamp is None or self.first_timestamp is None:
            return None
        return timestamp - self.first_timestamp

    def _log(self, timestamp, status, angle=None, plane=None,
             visibility=None, reference_frame=None):
        self.frame_log.append((
            self._relative_time(timestamp), status, angle, plane,
            visibility, reference_frame,
        ))

    def record_error(self, timestamp=None):
        """Count a frame that could not be processed."""
        self.frames_total += 1
        self.frame_errors += 1
        self.last_status = None
        if timestamp is not None:
            if self.first_timestamp is None:
                self.first_timestamp = timestamp
            self.last_timestamp = timestamp
        self._log(timestamp, STATUS_ERROR)

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
        visible, unmeasurable, implausible, processing error.
        """
        return {
            "frames_total": self.frames_total,
            "frames_measured": self.frames_measured,
            "frames_no_pose": self.frames_total - self.frames_with_pose
            - self.frame_errors,
            "frames_not_visible": self.frames_not_visible,
            "frames_unmeasurable": self.frames_unmeasurable,
            "frames_implausible": self.frames_implausible,
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
            self._log(timestamp, STATUS_NO_POSE)
            return self._frame(STATUS_NO_POSE)

        self.frames_with_pose += 1

        landmarks = results.pose_landmarks.landmark
        visibility = min(
            (landmarks[i].visibility for i in self.required), default=None
        )
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
            self._log(timestamp, STATUS_NOT_VISIBLE, visibility=visibility)
            return self._frame(STATUS_NOT_VISIBLE)

        reading = self.angle_fn(results, self.side)

        if reading is None or not math.isfinite(reading.value):
            self.frames_unmeasurable += 1
            self._log(timestamp, STATUS_UNMEASURABLE, visibility=visibility)
            return self._frame(STATUS_UNMEASURABLE)

        angle = float(reading.value)
        plane = reading.plane
        unverified = (
            plane is None and reading.plane_status in PLANE_UNVERIFIED
        )

        if self._implausible(angle, timestamp):
            self.frames_implausible += 1
            self._log(timestamp, STATUS_IMPLAUSIBLE, angle, plane,
                      visibility, reading.reference_frame)
            return self._frame(STATUS_IMPLAUSIBLE)

        self._handle_gap(timestamp)

        self.frames_measured += 1
        if reading.reference_frame == "camera":
            self.frames_camera_vertical += 1

        self._recent.append(angle)
        self._log(timestamp, STATUS_OK, angle, plane, visibility,
                  reading.reference_frame)

        self._update_segmentation(angle, plane, timestamp, unverified)

        self.current_angle = angle
        self.current_plane = plane
        self.current_plane_status = reading.plane_status
        self.max_angle = max(self.max_angle, angle)
        self.min_angle = min(self.min_angle, angle)

        self._update_reps(angle, plane, timestamp, unverified)

        return self._frame(STATUS_OK, angle, plane)

    def _implausible(self, value, timestamp):
        """
        Glitch filter. Compares with the previous READING (accepted or
        not), so a lasting jump costs one frame rather than rejecting
        everything that follows.
        """

        previous = self._last_raw
        self._last_raw = (timestamp, value)

        limit = self.config.max_speed
        if limit is None or timestamp is None or previous is None:
            return False
        t0, v0 = previous
        if t0 is None or timestamp <= t0:
            return False
        return abs(value - v0) / (timestamp - t0) > limit

    def _handle_gap(self, timestamp):
        """Abandon a repetition when tracking was lost for too long."""

        max_gap = self.config.rep.max_gap_s
        last = self._last_measured_t
        self._last_measured_t = timestamp

        if max_gap is None or timestamp is None or last is None:
            return
        if timestamp - last <= max_gap:
            return

        if self.stage == "up":
            self.interrupted_reps += 1
            self.stage = "down"
        self.up_frames = 0
        self.down_frames = 0
        self._up_dwell = 0.0
        self._down_dwell = 0.0
        self._prev_zone = None
        self._up_plane_votes = []
        self._up_unverified = 0
        self._excursion_peak = None
        self._recent.clear()

    def _frame(self, status, angle=None, plane=None):
        self.last_status = status
        return FrameAnalysis(
            status=status,
            angle=angle,
            plane=plane,
            reps=self.reps,
            movement_active=self.movement_active,
        )

    def _update_segmentation(self, angle, plane, timestamp,
                             unverified=False):
        seg = self.config.segmentation

        if angle >= seg.start_threshold:
            self.movement_active = True

        if not self.movement_active:
            return

        self.current_segment.append(angle)
        self._current_planes.append(plane)
        self._current_unverified.append(unverified)
        self._current_times.append(self._relative_time(timestamp))
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
                    self._segment_unverified.append(
                        list(self._current_unverified)
                    )
                    self._segment_times.append(list(self._current_times))

                self.current_segment = []
                self._current_planes = []
                self._current_unverified = []
                self._current_times = []
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

    def _debounced(self, frames, dwell, timestamp):
        rep = self.config.rep
        if rep.min_dwell_s is None or timestamp is None:
            return frames >= rep.debounce_frames
        return (
            frames >= rep.min_debounce_frames
            and dwell >= rep.min_dwell_s - 1e-9
        )

    def _zone_dwell(self, zone, timestamp):
        """Time added to `zone` by this frame: only contiguous frames in
        the same zone accumulate (frames between the thresholds, or
        another zone, interrupt but do not reset the tally)."""
        added = 0.0
        if (
            timestamp is not None
            and self._prev_zone == zone
            and self._prev_zone_t is not None
        ):
            added = max(0.0, timestamp - self._prev_zone_t)
        self._prev_zone = zone
        self._prev_zone_t = timestamp
        return added

    def _rep_plane_decision(self):
        """(in plane, plane verified) for the repetition going up."""
        votes = [v for v in self._up_plane_votes if v is not None]
        check = self.config.plane_check
        if check is not None and check.require_known_plane:
            verified = bool(votes) and len(votes) >= self._up_unverified
            return verified and sum(votes) * 2 >= len(votes), verified
        # Unknown plane (no plane check, or not measurable) does not
        # block the repetition.
        return not votes or sum(votes) * 2 >= len(votes), True

    def _update_reps(self, angle, plane, timestamp, unverified=False):
        rep = self.config.rep
        robust = median(self._recent) if self._recent else angle

        if angle >= self.up_threshold:
            self.up_frames += 1
            self.down_frames = 0
            self._down_dwell = 0.0
            self._up_dwell += self._zone_dwell("up", timestamp)
            self._up_plane_votes.append(self._plane_ok(plane))
            self._up_unverified += unverified
            self._track_peak(robust)

            if (
                self._debounced(self.up_frames, self._up_dwell, timestamp)
                and self.stage == "down"
            ):
                self.stage = "up"
                self._rep_in_plane, self._rep_plane_verified = (
                    self._rep_plane_decision()
                )

        elif angle <= rep.down_threshold:
            self.down_frames += 1
            self.up_frames = 0
            self._up_dwell = 0.0
            self._down_dwell += self._zone_dwell("down", timestamp)
            self._up_plane_votes = []
            self._up_unverified = 0

            if (
                self._debounced(self.down_frames, self._down_dwell,
                                timestamp)
                and self.stage == "up"
            ):
                self.stage = "down"
                self.rep_peaks.append(
                    (self._excursion_peak, self._rep_in_plane)
                )
                if self._rep_in_plane:
                    self.reps += 1
                else:
                    self.off_plane_reps += 1
                    if not self._rep_plane_verified:
                        self.plane_unverified_reps += 1
                self._excursion_peak = None
            elif self.stage == "down":
                self._excursion_peak = None

        else:
            self._prev_zone = "mid"
            self._prev_zone_t = timestamp
            self._track_peak(robust)

    def _track_peak(self, robust):
        if self._excursion_peak is None or robust > self._excursion_peak:
            self._excursion_peak = robust

    # -----------------------------------------------------
    # Session summary
    # -----------------------------------------------------

    def _scored_segments(self):
        """Completed segments plus an unfinished one at session end."""

        segments = [
            (segment.copy(), planes, times, unverified)
            for segment, planes, times, unverified in zip(
                self.movement_segments, self._segment_planes,
                self._segment_times, self._segment_unverified,
            )
        ]

        if self.movement_active and self._segment_is_kept(
            self.current_segment, self._current_active_frames
        ):
            segments.append(
                (self.current_segment.copy(), list(self._current_planes),
                 list(self._current_times), list(self._current_unverified))
            )

        return segments

    def segment_in_plane(self, segment, planes, unverified=None):
        """
        True/False when the segment's elevated frames (angle at least
        plane_check.min_elevation, plane measurable) mostly lie within the
        plane tolerance; None when there is no plane check or no such
        frame, or when the plane is required but not verified (see
        segment_plane_verified). Same majority rule as repetition
        counting.
        """

        check = self.config.plane_check
        if check is None:
            return None
        if not self.segment_plane_verified(segment, planes, unverified):
            return None
        votes = [
            abs(plane - check.expected_plane) <= check.tolerance
            for angle, plane in zip(segment, planes)
            if plane is not None and angle >= check.min_elevation
        ]
        if not votes:
            return None
        return sum(votes) * 2 >= len(votes)

    def segment_plane_verified(self, segment, planes, unverified=None):
        """
        False when the exercise requires a known plane and fewer of the
        segment's elevated frames have a known plane than an unverified
        one; True otherwise (including segments never elevated enough
        for a plane, which are treated as before).
        """

        check = self.config.plane_check
        if check is None or not check.require_known_plane:
            return True
        flags = unverified or [False] * len(segment)
        known = unknown = 0
        for angle, plane, flag in zip(segment, planes, flags):
            if angle < check.min_elevation:
                continue
            if plane is not None:
                known += 1
            elif flag:
                unknown += 1
        return known >= unknown

    def peak_rom(self, segment_details):
        """
        v2 peak range of motion: the highest 3-frame-median value among
        counted in-plane repetitions and scored segments that were not
        off-plane. None when no such movement exists (not 0).
        """

        candidates = [
            peak for peak, in_plane in self.rep_peaks
            if in_plane and peak is not None
        ]
        candidates += [
            d["robust_peak"] for d in segment_details
            if d.get("in_plane") is not False
            and d.get("plane_verified", True)
            and d["robust_peak"] is not None
        ]
        return round(max(candidates), 2) if candidates else None

    def summary(self):
        reference_curve = list(self.reference.angle_curve)
        reference_peak = max(reference_curve, default=0.0)
        weight = self.config.dtw.range_weight
        normalization = self.config.dtw.normalization
        seg_config = self.config.segmentation

        segment_details = []
        # Averaged unrounded, then rounded once (original behaviour).
        raw_scores = []
        dtw_scores = []
        dtw_errors = []
        off_plane_segments = 0
        plane_unverified_segments = 0

        for segment, planes, times, unverified in self._scored_segments():
            if len(segment) < seg_config.min_segment_frames:
                continue

            movement_range = max(segment) - min(segment)
            patient_peak = max(segment)

            if movement_range < seg_config.min_range:
                continue

            in_plane = (
                self.segment_in_plane(segment, planes, unverified)
                if self.v2 else None
            )
            plane_verified = (
                self.segment_plane_verified(segment, planes, unverified)
                if self.v2 else True
            )
            robust_peak = rolling_median_peak(segment) if self.v2 \
                else patient_peak

            peak_index = segment.index(patient_peak)
            peak_plane = planes[peak_index] if planes else None

            detail = {
                "frames": len(segment),
                "peak": round(patient_peak, 2),
                "range": round(movement_range, 2),
                "plane_at_peak": (
                    None if peak_plane is None else round(peak_plane, 1)
                ),
            }

            if self.v2:
                detail["robust_peak"] = round(robust_peak, 2)
                detail["in_plane"] = in_plane
                detail["plane_verified"] = plane_verified
                known = [t for t in times if t is not None]
                detail["duration_s"] = (
                    round(known[-1] - known[0], 3) if len(known) > 1
                    else None
                )
                smooth_input = resample(
                    segment, times, SMOOTHNESS_RESAMPLE_HZ
                )
            else:
                smooth_input = segment

            detail["velocity_peaks"] = velocity_peak_count(smooth_input)

            if in_plane is False or not plane_verified:
                # Different (or unverifiable) movement plane: not
                # compared with the reference and not part of the
                # movement score.
                off_plane_segments += 1
                plane_unverified_segments += not plane_verified
                detail.update({
                    "dtw_similarity": None,
                    "dtw_mean_error_deg": None,
                    "range_completion": None,
                    "score": None,
                })
                segment_details.append(detail)
                continue

            error = dtw_distance(reference_curve, segment, normalization)
            dtw_score = similarity_from_error(error)

            range_completion = (
                min(1.0, robust_peak / reference_peak)
                if reference_peak > 0
                else 0.0
            )

            score = (
                (1.0 - weight) * dtw_score
                + weight * range_completion * 100.0
            )

            score = max(0.0, min(100.0, score))
            raw_scores.append(score)
            dtw_scores.append(dtw_score)
            if math.isfinite(error):
                dtw_errors.append(error)

            detail.update({
                "dtw_similarity": dtw_score,
                "range_completion": round(range_completion, 4),
                "score": round(score, 2),
            })
            if self.v2:
                detail["dtw_mean_error_deg"] = (
                    round(error, 2) if math.isfinite(error) else None
                )
            segment_details.append(detail)

        # Movement score Q: mean over scored segments of
        # (1 - w) * DTW similarity + w * 100 * range completion.
        # Stored in the "trajectory_similarity" column for historical
        # reasons; it is NOT a pure similarity.
        movement_score = (
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
            if d["velocity_peaks"] is not None and d.get("score") is not None
        ]

        summary = {
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
            "trajectory_similarity": movement_score,
            "movement_score": movement_score,
            "valid_segments": len(raw_scores) if self.v2
            else len(segment_details),
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
            "metrics_version": self.config.metrics_version,
        }

        if self.v2:
            summary.update({
                "peak_rom": self.peak_rom(segment_details),
                "dtw_similarity": (
                    round(sum(dtw_scores) / len(dtw_scores), 2)
                    if dtw_scores else None
                ),
                "dtw_mean_error_deg": (
                    round(sum(dtw_errors) / len(dtw_errors), 2)
                    if dtw_errors else None
                ),
                "dtw_normalization": normalization,
                "off_plane_segments": off_plane_segments,
                "plane_unverified_segments": plane_unverified_segments,
                "plane_unverified_reps": self.plane_unverified_reps,
                "interrupted_reps": self.interrupted_reps,
                "rep_in_progress_at_end": self.stage == "up",
                "frame_log": list(self.frame_log),
            })

        return summary
