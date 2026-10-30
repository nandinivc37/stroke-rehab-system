"""
Rule-based compensation detection from MediaPipe Pose world landmarks
(method "geometric_rules_v1").

PROVISIONAL. The thresholds below are engineering defaults chosen to be
well above typical landmark noise for a seated adult. They have NOT been
validated against labelled data or reviewed by a physiotherapist, and
results must not be presented as clinical findings. Adjust them in
DetectorConfig after therapist review; every session stores the
configuration it used.

Patterns (each a deviation from the patient's own resting posture,
measured at the start of the session with the arms relaxed):

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


@dataclass(frozen=True)
class DetectorConfig:
    baseline_frames: int = 15
    # The resting baseline is accepted only if its samples vary by less
    # than this fraction of each pattern's enter threshold (median
    # absolute deviation); otherwise the oldest samples are dropped and
    # collection continues.
    baseline_stability: float = 0.25
    min_visibility: float = 0.5
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
        ),
        PatternConfig(
            TRUNK_DEPTH, "Trunk lean towards/away from camera (depth-based)",
            "degrees", enter_threshold=15.0, exit_threshold=10.0,
            landmarks=SHOULDERS + HIPS,
            cue="Keep your trunk upright and still",
            depth_dependent=True,
        ),
        PatternConfig(
            SHOULDER_HIKING, "Shoulder hiking (affected side)",
            "shoulder widths", enter_threshold=0.10, exit_threshold=0.07,
            landmarks=SHOULDERS + HIPS,
            cue="Keep your shoulder relaxed and down",
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
                "smoothing_frames", "onset_frames", "release_frames",
                "max_gap_frames")},
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
        self.unable_reasons = {}
        self.episodes = 0
        self.deviations = []

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

    # -----------------------------------------------------

    def _visibility(self, results, indices):
        if results is None or getattr(results, "pose_landmarks", None) is None:
            return None
        landmarks = results.pose_landmarks.landmark
        return min(landmarks[i].visibility for i in indices)

    def update(self, results, movement_active):
        """
        Returns {pattern: PatternResult}. While no movement attempt is
        active the frame may extend the resting baseline but is not
        assessed (state unable_to_assess, reason "resting").
        """

        cfg = self.config
        points = None
        if results is not None and getattr(results, "pose_landmarks", None):
            points = world_array(results)
        values = measure(points, self.side) if points is not None else None

        if movement_active:
            self.frames_active += 1
        else:
            self.frames_resting += 1

        out = {}
        for name, state in self.patterns.items():
            pattern = state.config
            visibility = self._visibility(results, pattern.landmarks)
            value = None if values is None else values.get(name)
            reason = ""
            if results is None or getattr(results, "pose_landmarks", None) is None:
                reason = "no pose detected"
            elif visibility is None or visibility < cfg.min_visibility:
                reason = "shoulders/hips not visible"
            elif value is None or not math.isfinite(value):
                reason = "geometry undefined"

            if not movement_active:
                if not reason and state.baseline is None:
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
                )
                continue

            if not reason and state.baseline is None:
                reason = "no stable resting baseline"

            if reason:
                state.gap += 1
                if state.active and state.gap <= cfg.max_gap_frames:
                    # Brief dropout inside an episode: keep reporting it.
                    result = PatternResult(
                        DETECTED, deviation=state.deviations[-1]
                        if state.deviations else None,
                        threshold=pattern.enter_threshold, unit=pattern.unit,
                        reason="held through brief dropout",
                        min_visibility=visibility,
                        baseline_samples=len(state.baseline_samples),
                    )
                    state.counts[DETECTED] += 1
                else:
                    state.reset_episode()
                    result = PatternResult(
                        UNABLE, threshold=pattern.enter_threshold,
                        unit=pattern.unit, reason=reason,
                        min_visibility=visibility,
                        baseline_samples=len(state.baseline_samples),
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
            magnitude = (
                deviation if name in (TRUNK_DEPTH, SHOULDER_HIKING)
                else abs(deviation)
            )
            # Depth lean counts only towards the camera (forward) and
            # hiking only upwards; image-plane lean counts either way.

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
            out[name] = PatternResult(
                verdict, deviation=deviation,
                threshold=pattern.enter_threshold, unit=pattern.unit,
                min_visibility=visibility,
                baseline_samples=len(state.baseline_samples),
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
        # Otherwise keep sliding: the patient was still moving.

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
            patterns[name] = {
                "label": pattern.label,
                "unit": pattern.unit,
                "enter_threshold": pattern.enter_threshold,
                "exit_threshold": pattern.exit_threshold,
                "depth_dependent": pattern.depth_dependent,
                "baseline": (None if state.baseline is None
                             else round(state.baseline, 4)),
                "assessed": assessed > 0,
                "frames": dict(state.counts),
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
                    "min_visibility", "smoothing_frames", "onset_frames",
                    "release_frames", "max_gap_frames")
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
