"""
Rule-based compensation detection from MediaPipe Pose world landmarks
(method "geometric_rules_v1").

PROVISIONAL. The thresholds below are engineering defaults chosen to be
well above typical landmark noise for a seated adult. They have NOT been
validated against labelled data or reviewed by a physiotherapist, and
results must not be presented as clinical findings. Adjust them in
DetectorConfig after therapist review; every session stores the
configuration it used.

Two references per pattern, both from the patient's own resting posture
(median of a stable resting window with the arm measured and relaxed):
  * current resting baseline: refreshed at every stable rest between
    attempts. The live verdict (detected / not_detected, episodes,
    detected_fraction) is relative to it, so shifting in the chair is
    not flagged as compensation during the next attempt;
  * session-start baseline: the first stable resting window of the
    session, never replaced. Deviations from it are reported separately
    (summary "session_start", "rest_drift") so that a patient leaning
    gradually further between attempts cannot make that change disappear
    from the session record. It is NOT an ideal or clinically validated
    posture, only where this session started.

Patterns:

  trunk_lean_image_plane
      Tilt of the hip-midpoint -> shoulder-midpoint line within the
      camera image plane (x/y), in degrees. With a frontal camera this is
      SIDEWAYS (lateral) trunk lean; with a side camera it is forward or
      backward lean. Does not use depth.
  trunk_lean_depth
      Tilt of the same line towards/away from the camera (y/z), in
      degrees. With a frontal camera this is forward lean. Uses
      MediaPipe's depth estimate, which is noisy: higher threshold.
  shoulder_hiking
      Rise of the affected shoulder relative to the other shoulder,
      measured along the trunk axis (so trunk lean does not count as
      hiking), in shoulder widths. Raising the arm moves the shoulder
      girdle physiologically, so shoulder exercises use a higher
      threshold than elbow exercises.

Not implemented: trunk rotation (needs reliable depth for both shoulder
and hip lines; frontal MediaPipe depth is too noisy to separate rotation
from tracking error).

Assessment states per pattern and frame:
  detected / not_detected  - assessed during an active movement attempt
  unable_to_assess         - landmarks missing or below visibility, no
                             stable resting baseline yet, or degenerate
                             geometry. Never treated as "not detected".
Frames at rest (no movement attempt) are not assessed: resting posture
is not compensation.

Implausible pose estimates are unable_to_assess, never detected: a trunk
more than max_trunk_inclination degrees from vertical, or a deviation
beyond the pattern's plausible_limit, is not a seated posture but a
MediaPipe world-landmark failure (recorded sessions, 2026-11-01, had
trunk "lean" of 150-179° and deviations up to 193° scored as detected).

Restricted range of motion and slow movement are not inputs: only trunk
and shoulder-girdle displacement from the resting posture is measured.
"""

import math
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from src.exercises.angles import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    world_array,
)

METHOD = "geometric_rules_v1"

DETECTED = "detected"
NOT_DETECTED = "not_detected"
UNABLE = "unable_to_assess"

# Stable categorical reasons (summaries count frames per reason); the
# measured values are in diagnostics(), never in the reason text.
REASON_IMPLAUSIBLE_POSE = (
    "implausible pose estimate (trunk beyond {limit:g}° from vertical)")
REASON_IMPLAUSIBLE_DEVIATION = (
    "implausible measurement (deviation beyond plausible limit)")
REASON_NO_POSE = "no pose detected"
REASON_NOT_VISIBLE = "shoulders/hips not visible"
# Ordinary visibility loss: the only reasons that may hold an ongoing
# warning for up to max_gap_frames. A geometrically invalid frame
# (implausible, undefined geometry) ends the warning immediately.
HOLDABLE_REASONS = (REASON_NO_POSE, REASON_NOT_VISIBLE)

TRUNK_IMAGE = "trunk_lean_image_plane"
TRUNK_DEPTH = "trunk_lean_depth"
SHOULDER_HIKING = "shoulder_hiking"

SHOULDERS = (LEFT_SHOULDER, RIGHT_SHOULDER)
HIPS = (LEFT_HIP, RIGHT_HIP)


@dataclass(frozen=True)
class PatternConfig:
    name: str
    label: str
    unit: str
    # Smoothed |deviation| must reach enter_threshold for onset_frames
    # consecutive frames to start an episode, and stay below
    # exit_threshold for release_frames frames to end it (hysteresis).
    enter_threshold: float
    exit_threshold: float
    landmarks: tuple
    cue: str
    depth_dependent: bool = False
    # |deviation from baseline| beyond this is a pose-estimation failure,
    # not posture (engineering bound, far above enter_threshold).
    plausible_limit: float = None


@dataclass(frozen=True)
class DetectorConfig:
    baseline_frames: int = 15
    # The resting baseline is accepted only if its samples vary by less
    # than this fraction of each pattern's enter threshold (median
    # absolute deviation); otherwise the oldest samples are dropped and
    # collection continues.
    baseline_stability: float = 0.25
    min_visibility: float = 0.5
    # Hip-midpoint -> shoulder-midpoint line further than this from
    # vertical (3D) is not a seated posture: pose estimate rejected.
    max_trunk_inclination: float = 60.0
    smoothing_frames: int = 5
    onset_frames: int = 5
    release_frames: int = 5
    # Missing frames tolerated inside an episode before the pattern
    # becomes unable_to_assess and its episode state is reset.
    max_gap_frames: int = 3
    patterns: tuple = (
        PatternConfig(
            TRUNK_IMAGE, "Trunk lean (sideways in a front view)", "degrees",
            enter_threshold=10.0, exit_threshold=7.0,
            landmarks=SHOULDERS + HIPS,
            cue="Keep your trunk upright and still",
            plausible_limit=45.0,
        ),
        PatternConfig(
            TRUNK_DEPTH, "Trunk lean towards/away from camera (depth-based)",
            "degrees", enter_threshold=15.0, exit_threshold=10.0,
            landmarks=SHOULDERS + HIPS,
            cue="Keep your trunk upright and still",
            depth_dependent=True,
            plausible_limit=45.0,
        ),
        PatternConfig(
            SHOULDER_HIKING, "Shoulder hiking (affected side)",
            "shoulder widths", enter_threshold=0.10, exit_threshold=0.07,
            landmarks=SHOULDERS + HIPS,
            cue="Keep your shoulder relaxed and down",
            plausible_limit=0.5,
        ),
    )
    # Per-exercise threshold overrides {exercise_id: {pattern: (enter,
    # exit)}}. Raising the arm overhead lifts the shoulder girdle
    # physiologically, so hiking needs more margin in shoulder exercises.
    exercise_overrides: dict = field(default_factory=lambda: {
        "shoulder_flexion": {SHOULDER_HIKING: (0.15, 0.11)},
        "shoulder_abduction": {SHOULDER_HIKING: (0.15, 0.11)},
        "forward_reach": {SHOULDER_HIKING: (0.15, 0.11)},
    })

    def for_exercise(self, exercise_id):
        overrides = self.exercise_overrides.get(exercise_id, {})
        patterns = tuple(
            PatternConfig(**dict(
                asdict(p),
                enter_threshold=overrides[p.name][0],
                exit_threshold=overrides[p.name][1],
            )) if p.name in overrides else p
            for p in self.patterns
        )
        return DetectorConfig(**dict(
            {k: getattr(self, k) for k in (
                "baseline_frames", "baseline_stability", "min_visibility",
                "max_trunk_inclination", "smoothing_frames", "onset_frames",
                "release_frames", "max_gap_frames")},
            patterns=patterns, exercise_overrides={},
        ))


# ---------------------------------------------------------
# Geometry
# ---------------------------------------------------------

def measure(points, side):
    """
    Raw measurements from (33, 3) world points (x right, y down, z away
    from the camera). Returns {pattern: value} with None for any value
    that is geometrically undefined.
    """

    left_shoulder, right_shoulder = points[LEFT_SHOULDER], points[RIGHT_SHOULDER]
    trunk = (
        (left_shoulder + right_shoulder) / 2
        - (points[LEFT_HIP] + points[RIGHT_HIP]) / 2
    )
    trunk_length = float(np.linalg.norm(trunk))
    shoulder_line = (
        left_shoulder - right_shoulder if side.upper() == "LEFT"
        else right_shoulder - left_shoulder
    )  # other shoulder -> affected shoulder
    shoulder_width = float(np.linalg.norm(shoulder_line))

    values = {TRUNK_IMAGE: None, TRUNK_DEPTH: None, SHOULDER_HIKING: None}
    if trunk_length < 1e-6:
        return values

    # Up is -y. Angles from vertical within each plane.
    values[TRUNK_IMAGE] = math.degrees(math.atan2(trunk[0], -trunk[1]))
    values[TRUNK_DEPTH] = math.degrees(math.atan2(-trunk[2], -trunk[1]))

    if shoulder_width > 1e-6:
        up = trunk / trunk_length
        values[SHOULDER_HIKING] = float(shoulder_line @ up) / shoulder_width
    return values


def trunk_inclination(points):
    """Angle (degrees) of the hip-midpoint -> shoulder-midpoint line from
    the camera vertical, in 3D; None if degenerate."""

    trunk = (
        (points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]) / 2
        - (points[LEFT_HIP] + points[RIGHT_HIP]) / 2
    )
    length = float(np.linalg.norm(trunk))
    if length < 1e-6:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, -trunk[1] / length))))


# ---------------------------------------------------------
# Detector
# ---------------------------------------------------------

@dataclass
class PatternResult:
    state: str
    deviation: float = None        # smoothed deviation from baseline
    threshold: float = None        # enter threshold in force
    unit: str = ""
    reason: str = ""
    min_visibility: float = None
    baseline_samples: int = 0
    value: float = None            # raw measurement this frame
    baseline: float = None         # current resting baseline in force
    session_baseline: float = None  # session-start baseline (immutable)
    session_deviation: float = None  # value - session_baseline (raw)


class _PatternState:
    def __init__(self, config):
        self.config = config
        self.baseline_samples = deque(maxlen=None)
        self.baseline = None
        self.recent = deque(maxlen=None)
        self.active = False          # inside a detected episode
        self.above = 0
        self.below = 0
        self.gap = 0
        # Session statistics (active-movement frames only).
        self.counts = {DETECTED: 0, NOT_DETECTED: 0, UNABLE: 0}
        # Frames whose warning was held through brief visibility loss:
        # shown live, but not measured, so never counted as detected.
        self.held = 0
        self.unable_reasons = {}
        self.episodes = 0
        self.deviations = []
        self.baseline_updates = 0
        # Session-start reference (set once, never replaced).
        self.session_baseline = None
        self.session_deviations = []    # assessed frames only
        self.session_beyond = 0         # ... beyond enter_threshold
        self.max_abs_rest_drift = 0.0

    def reset_episode(self):
        self.recent.clear()
        self.active = False
        self.above = 0
        self.below = 0


class GeometricCompensationDetector:
    """
    Per-frame compensation assessment from the same MediaPipe results the
    exercise analyzer uses. Qt-free; call update() once per frame.
    """

    def __init__(self, side, exercise_id=None, config=None):
        self.side = side.upper()
        self.exercise_id = exercise_id
        self.config = (config or DetectorConfig()).for_exercise(exercise_id)
        self.reset()

    def reset(self):
        self.patterns = {
            p.name: _PatternState(p) for p in self.config.patterns
        }
        self.frames_resting = 0
        self.frames_active = 0
        self.last = None
        self.last_inclination = None
        self._was_active = False

    # -----------------------------------------------------

    def _visibility(self, results, indices):
        if results is None or getattr(results, "pose_landmarks", None) is None:
            return None
        landmarks = results.pose_landmarks.landmark
        return min(landmarks[i].visibility for i in indices)

    def update(self, results, movement_active, arm_at_rest=None):
        """
        Returns {pattern: PatternResult}. While no movement attempt is
        active the frame is not assessed (state unable_to_assess, reason
        "resting"); it extends the resting baseline only when arm_at_rest
        (default: not movement_active), i.e. the exercise arm was measured
        and is not in an attempt. The baseline is re-established at every
        stable rest, and its samples restart when an attempt begins.
        """

        cfg = self.config
        if arm_at_rest is None:
            arm_at_rest = not movement_active
        points = None
        if results is not None and getattr(results, "pose_landmarks", None):
            points = world_array(results)
        values = measure(points, self.side) if points is not None else None
        self.last_inclination = (
            trunk_inclination(points) if points is not None else None)
        implausible = (
            self.last_inclination is not None
            and self.last_inclination > cfg.max_trunk_inclination
        )

        if movement_active:
            self.frames_active += 1
            if not self._was_active:
                for state in self.patterns.values():
                    state.baseline_samples.clear()
        else:
            self.frames_resting += 1
        self._was_active = movement_active

        out = {}
        for name, state in self.patterns.items():
            pattern = state.config
            visibility = self._visibility(results, pattern.landmarks)
            value = None if values is None else values.get(name)
            reason = ""
            if results is None or getattr(results, "pose_landmarks", None) is None:
                reason = REASON_NO_POSE
            elif visibility is None or visibility < cfg.min_visibility:
                reason = REASON_NOT_VISIBLE
            elif value is None or not math.isfinite(value):
                reason = "geometry undefined"
            elif implausible:
                reason = REASON_IMPLAUSIBLE_POSE.format(
                    limit=cfg.max_trunk_inclination)

            if not movement_active:
                if not reason and arm_at_rest:
                    self._add_baseline_sample(state, value)
                state.reset_episode()
                out[name] = PatternResult(
                    UNABLE, unit=pattern.unit,
                    reason=reason or ("calibrating baseline"
                                      if state.baseline is None
                                      else "resting"),
                    threshold=pattern.enter_threshold,
                    min_visibility=visibility,
                    baseline_samples=len(state.baseline_samples),
                    value=value, baseline=state.baseline,
                    session_baseline=state.session_baseline,
                )
                continue

            if not reason and state.baseline is None:
                reason = "no stable resting baseline"
            if (
                not reason and pattern.plausible_limit is not None
                and abs(value - state.baseline) > pattern.plausible_limit
            ):
                reason = REASON_IMPLAUSIBLE_DEVIATION

            if reason:
                state.gap += 1
                if (
                    state.active and state.gap <= cfg.max_gap_frames
                    and reason in HOLDABLE_REASONS
                ):
                    # Brief visibility loss inside an episode: keep the
                    # warning on screen, but nothing was measured, so the
                    # frame is not confirmed compensation.
                    result = PatternResult(
                        DETECTED, deviation=state.deviations[-1]
                        if state.deviations else None,
                        threshold=pattern.enter_threshold, unit=pattern.unit,
                        reason="held through brief dropout",
                        min_visibility=visibility,
                        baseline_samples=len(state.baseline_samples),
                        value=value, baseline=state.baseline,
                        session_baseline=state.session_baseline,
                    )
                    state.held += 1
                else:
                    state.reset_episode()
                    result = PatternResult(
                        UNABLE, threshold=pattern.enter_threshold,
                        unit=pattern.unit, reason=reason,
                        min_visibility=visibility,
                        baseline_samples=len(state.baseline_samples),
                        value=value, baseline=state.baseline,
                        session_baseline=state.session_baseline,
                    )
                    state.counts[UNABLE] += 1
                    state.unable_reasons[reason] = (
                        state.unable_reasons.get(reason, 0) + 1)
                out[name] = result
                continue

            state.gap = 0
            state.recent.append(value - state.baseline)
            while len(state.recent) > cfg.smoothing_frames:
                state.recent.popleft()
            deviation = float(np.median(state.recent))
            magnitude = self._magnitude(name, deviation)

            if magnitude >= pattern.enter_threshold:
                state.above += 1
                state.below = 0
            elif magnitude < pattern.exit_threshold:
                state.below += 1
                state.above = 0
            else:
                state.above = 0
                state.below = 0

            if not state.active and state.above >= cfg.onset_frames:
                state.active = True
                state.episodes += 1
            elif state.active and state.below >= cfg.release_frames:
                state.active = False

            verdict = DETECTED if state.active else NOT_DETECTED
            state.counts[verdict] += 1
            state.deviations.append(deviation)
            session_deviation = value - state.session_baseline
            state.session_deviations.append(session_deviation)
            if self._magnitude(name, session_deviation) >= (
                    pattern.enter_threshold):
                state.session_beyond += 1
            out[name] = PatternResult(
                verdict, deviation=deviation,
                threshold=pattern.enter_threshold, unit=pattern.unit,
                min_visibility=visibility,
                baseline_samples=len(state.baseline_samples),
                value=value, baseline=state.baseline,
                session_baseline=state.session_baseline,
                session_deviation=session_deviation,
            )

        self.last = out
        return out

    def _add_baseline_sample(self, state, value):
        cfg = self.config
        state.baseline_samples.append(value)
        while len(state.baseline_samples) > cfg.baseline_frames:
            state.baseline_samples.popleft()
        if len(state.baseline_samples) < cfg.baseline_frames:
            return
        samples = np.asarray(state.baseline_samples)
        median = float(np.median(samples))
        spread = float(np.median(np.abs(samples - median)))
        if spread <= cfg.baseline_stability * state.config.enter_threshold:
            state.baseline = median
            state.baseline_updates += 1
            if state.session_baseline is None:
                state.session_baseline = median
            state.max_abs_rest_drift = max(
                state.max_abs_rest_drift,
                abs(median - state.session_baseline))
        # Otherwise keep sliding: the patient was still moving.

    @staticmethod
    def _magnitude(name, deviation):
        """Depth lean counts only towards the camera (forward) and hiking
        only upwards; image-plane lean counts either way."""
        return (deviation if name in (TRUNK_DEPTH, SHOULDER_HIKING)
                else abs(deviation))

    # -----------------------------------------------------

    def baseline_established(self):
        return any(s.baseline is not None for s in self.patterns.values())

    def summary(self):
        """
        Session summary. Patterns that were never assessed report
        assessed=False and detected_fraction None, never 0.
        """

        patterns = {}
        for name, state in self.patterns.items():
            pattern = state.config
            assessed = state.counts[DETECTED] + state.counts[NOT_DETECTED]
            values = np.abs(np.asarray(state.deviations, dtype=float))
            session_values = np.abs(
                np.asarray(state.session_deviations, dtype=float))
            patterns[name] = {
                "label": pattern.label,
                "unit": pattern.unit,
                "enter_threshold": pattern.enter_threshold,
                "exit_threshold": pattern.exit_threshold,
                "depth_dependent": pattern.depth_dependent,
                "baseline": (None if state.baseline is None
                             else round(state.baseline, 4)),
                "baseline_updates": state.baseline_updates,
                # Session-start reference (not an ideal posture).
                "session_start_baseline": (
                    None if state.session_baseline is None
                    else round(state.session_baseline, 4)),
                "rest_drift": (
                    None if state.session_baseline is None
                    else round(state.baseline - state.session_baseline, 4)),
                "max_abs_rest_drift": (
                    None if state.session_baseline is None
                    else round(state.max_abs_rest_drift, 4)),
                "session_start": {
                    "max_abs_deviation": (
                        round(float(session_values.max()), 4)
                        if session_values.size else None),
                    "p95_abs_deviation": (
                        round(float(np.percentile(session_values, 95)), 4)
                        if session_values.size else None),
                    "beyond_threshold_fraction": (
                        round(state.session_beyond / assessed, 4)
                        if assessed else None),
                },
                "assessed": assessed > 0,
                "frames": dict(state.counts),
                # Not part of "frames" (layout kept for older sessions)
                # nor of detected_fraction.
                "frames_held_through_dropout": state.held,
                "detected_fraction": (
                    round(state.counts[DETECTED] / assessed, 4)
                    if assessed else None),
                "episodes": state.episodes,
                "max_abs_deviation": (round(float(values.max()), 4)
                                      if values.size else None),
                "p95_abs_deviation": (
                    round(float(np.percentile(values, 95)), 4)
                    if values.size else None),
                "unable_reasons": dict(state.unable_reasons),
            }
        return {
            "method": METHOD,
            "validated": False,
            "provisional_thresholds": True,
            "used_for_adaptation": False,
            "exercise_id": self.exercise_id,
            "side": self.side,
            "frames_active": self.frames_active,
            "frames_resting": self.frames_resting,
            "baseline_established": self.baseline_established(),
            "any_detected": any(p["episodes"] for p in patterns.values()),
            "patterns": patterns,
            "config": {
                k: getattr(self.config, k) for k in (
                    "baseline_frames", "baseline_stability",
                    "min_visibility", "max_trunk_inclination",
                    "smoothing_frames", "onset_frames", "release_frames",
                    "max_gap_frames")
            },
        }

    def diagnostics(self):
        """Latest frame's decision inputs, for the overlay and recordings
        (JSON-serialisable)."""

        def number(value, digits=3):
            return None if value is None else round(float(value), digits)

        return {
            "baseline_established": self.baseline_established(),
            "trunk_inclination": number(self.last_inclination, 1),
            "patterns": {
                name: {
                    "state": result.state,
                    "value": number(result.value),
                    "baseline": number(result.baseline),
                    "deviation": number(result.deviation),
                    "current_deviation": number(
                        None if result.value is None or result.baseline is None
                        else result.value - result.baseline),
                    "session_baseline": number(result.session_baseline),
                    "session_deviation": number(
                        None if result.value is None
                        or result.session_baseline is None
                        else result.value - result.session_baseline),
                    "threshold": result.threshold,
                    "reason": result.reason,
                    "baseline_samples": result.baseline_samples,
                    "min_visibility": number(result.min_visibility, 2),
                    "held_through_dropout": (
                        result.reason == "held through brief dropout"),
                    "frames_held_through_dropout": (
                        self.patterns[name].held),
                }
                for name, result in (self.last or {}).items()
            },
        }


# ---------------------------------------------------------
# Live feedback text
# ---------------------------------------------------------

def feedback(results_by_pattern, config, movement_active, baseline_ready):
    """
    (text, level) for the live posture line. level is "alert", "ok" or
    "info". Never says "no compensation" for a pattern that could not be
    assessed.
    """

    prefix = "Posture check (experimental): "
    if results_by_pattern is None:
        return prefix + "waiting for camera", "info"
    if not baseline_ready:
        return (prefix + "calibrating — sit upright with your arms relaxed",
                "info")
    if not movement_active:
        return prefix + "ready", "info"

    labels = {p.name: p for p in config.patterns}
    detected = [n for n, r in results_by_pattern.items()
                if r.state == DETECTED]
    if detected:
        cues = list(dict.fromkeys(labels[n].cue for n in detected))
        names = ", ".join(labels[n].label.split(" (")[0].lower()
                          for n in detected)
        return prefix + f"{names} — " + "; ".join(cues), "alert"

    assessed = [n for n, r in results_by_pattern.items()
                if r.state == NOT_DETECTED]
    unable = [r.reason for r in results_by_pattern.values()
              if r.state == UNABLE]
    if not assessed:
        reason = unable[0] if unable else "no data"
        return prefix + f"unable to assess ({reason})", "info"
    text = prefix + "no compensation detected"
    if unable:
        text += f" in {len(assessed)} of {len(results_by_pattern)} checks"
    return text, "ok"
