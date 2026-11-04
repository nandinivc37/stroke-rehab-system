"""
Shoulder flexion repetition counting from a front-facing camera: the
image-based movement plane, explicit rejection reasons, the diagnostic
overlay and the landmark recorder.

Live sessions 56-58 (2026-10-31, target 60°) completed 12-14 repetitions
each, but almost all were rejected as off-plane: the plane computed from
MediaPipe WORLD landmarks sat at 60-80° (0 = forward, 90 = sideways)
throughout the raise. Two image-based planes were then tried for the
DECISION (orthographic, then a pinhole camera model). On a 26-minute
recording (2026-11-01) the pinhole version rejected 12 repetitions that
the world plane of commit 7ee51f7 accepted, so the decision was restored
to the world plane; the image plane is computed for diagnostics only.
Close-range front-facing flexion remains unresolved. The scenarios below
are SYNTHETIC (pinhole camera); they are not evidence of webcam accuracy.
"""

import numpy as np
import pytest

import src.exercises.angles as angles
from src.exercises.angles import (
    PLANE_DEPTH_AMBIGUOUS,
    CalibratedHumeralElevation,
)
from src.exercises.analyzer import (
    REP_OFF_PLANE,
    REP_PLANE_UNVERIFIED,
    ExerciseAnalyzer,
)
from src.exercises.catalogue import get_exercise_config
from src.simulation.session_simulator import repetition_values, timestamps
from src.simulation.synthetic_pose import (
    ArmPose,
    BodyPose,
    make_results,
    project_perspective,
    world_points,
)
from src.validation.flexion_trace import trace
from src.validation.landmark_recorder import LandmarkRecorder, load_sessions

FLEXION = get_exercise_config("shoulder_flexion")
TARGET_60 = {"target_range": 60.0, "target_reps": 5}
WIDTH, HEIGHT = 1280, 720


def pose(side, flexion=0.0, abduction=0.0):
    body = BodyPose()
    arm = ArmPose(flexion=flexion, abduction=abduction)
    setattr(body, "left_arm" if side == "LEFT" else "right_arm", arm)
    return body


def frame(value, world_abduction=0.0, image_abduction=0.0, side="LEFT",
          aspect=1.0):
    """
    One frame of a raise to `value` degrees. The world landmarks and the
    image landmarks can disagree about how far sideways the arm is.
    aspect != 1 normalises image x by a wider frame, as MediaPipe does.
    """
    world_pose = pose(side, value, world_abduction if value else 0.0)
    image_pose = pose(side, value, image_abduction if value else 0.0)
    image = project_perspective(world_points(image_pose), aspect=aspect)
    return make_results(world_pose, image_points=image)


def session(peaks, **kwargs):
    return [frame(v, **kwargs)
            for peak in peaks for v in repetition_values(peak)]


def analyse(frames, params=TARGET_60, side="LEFT", image_size=None):
    analyzer = ExerciseAnalyzer(FLEXION, side, params)
    if image_size:
        analyzer.set_image_size(*image_size)
    for result, t in zip(frames, timestamps(len(frames))):
        analyzer.update(result, t)
    return analyzer, analyzer.summary()


# World abduction giving a world-landmark plane near the 67° median seen
# live, for a raise the image shows as straight forward.
WORLD_SKEW = 70.0


# ---------------------------------------------------------
# The world plane decides; the image plane is diagnostic only
# ---------------------------------------------------------

@pytest.mark.parametrize("side", ["LEFT", "RIGHT"])
def test_world_plane_decides_when_image_disagrees(side):
    """As in commit 7ee51f7: the world-landmark plane decides, even when
    the image shows a forward raise (the image plane is only reported)."""
    frames = session([90] * 3, world_abduction=WORLD_SKEW, side=side)
    analyzer, summary = analyse(frames, side=side)
    assert summary["reps"] == 0
    assert summary["off_plane_reps"] == 3
    assert all(e["outcome"] == REP_OFF_PLANE and 55 < e["plane"] < 85
               for e in summary["rep_events"])


def test_image_landmarks_never_change_the_decision(monkeypatch):
    """Same counts, planes and outcomes with the image plane disabled."""
    scenarios = [
        session([90] * 2, world_abduction=WORLD_SKEW),
        session([90] * 2, image_abduction=WORLD_SKEW),
        session([65, 55]),
        live_session("LEFT", [90, 110]),
        live_session("LEFT", [90], movement="abduction", depth_bias=-0.03),
    ]
    with_image = [analyse(frames, image_size=(WIDTH, HEIGHT))[1]
                  for frames in scenarios]
    monkeypatch.setattr(angles, "_image_plane_components",
                        lambda *_args: None)
    without = [analyse(frames, image_size=(WIDTH, HEIGHT))[1]
               for frames in scenarios]
    for a, b in zip(with_image, without):
        for key in ("reps", "off_plane_reps", "plane_unverified_reps",
                    "interrupted_reps", "peak_rom", "dtw_similarity"):
            assert a[key] == b[key], key
        assert a["rep_events"] == b["rep_events"]


def test_debug_state_exposes_the_disagreement():
    analyzer = ExerciseAnalyzer(FLEXION, "LEFT", TARGET_60)
    for result in session([90], world_abduction=WORLD_SKEW)[:40]:
        analyzer.update(result, None)
    state = analyzer.debug_state()
    assert state["plane_source"] == "world"
    assert state["plane"] == pytest.approx(state["world_plane"])
    assert abs(state["image_plane"]) < 2
    assert state["world_plane"] > 55
    assert state["plane_ok"] is False


def test_really_diagonal_raise_is_still_rejected():
    """No forced counting: if the image also shows the arm far out to
    the side, the repetition is rejected with the measured plane."""
    analyzer, summary = analyse(session(
        [90] * 3, world_abduction=WORLD_SKEW, image_abduction=WORLD_SKEW))
    assert summary["reps"] == 0
    assert summary["off_plane_reps"] == 3
    for event in summary["rep_events"]:
        assert event["outcome"] == REP_OFF_PLANE
        assert 55 < event["plane"] < 85
        assert "expected 0±30°" in event["reason"]


def test_sideways_raise_is_rejected():
    analyzer, summary = analyse([make_results(pose("LEFT", 0, v))
                                 for v in repetition_values(100) * 2])
    assert summary["reps"] == 0
    assert summary["off_plane_reps"] == 2


def test_aspect_ratio_is_needed_for_the_image_plane():
    """MediaPipe normalises image x by the width and y by the height;
    without the frame size the diagnostic image plane of a diagonal
    raise is wrong (here at 60° elevation, close to the camera). The
    decision does not depend on it."""

    def reading(image_size):
        measure = CalibratedHumeralElevation()
        if image_size:
            measure.set_image_size(*image_size)
        for _ in range(measure.rest_frames):
            measure(live_frame("LEFT"), "LEFT")
        result = measure(live_frame("LEFT", flexion=60, abduction=30),
                         "LEFT")
        return result, measure.last_details

    (with_size, details), (without, details_without) = (
        reading((WIDTH, HEIGHT)), reading(None))
    assert details["image_plane"] == pytest.approx(
        details["world_plane"], abs=1.0)
    assert abs(details_without["image_plane"]
               - details_without["world_plane"]) > 5
    assert with_size.plane == without.plane


def test_image_size_survives_reset():
    analyzer = ExerciseAnalyzer(FLEXION, "LEFT")
    analyzer.set_image_size(WIDTH, HEIGHT)
    analyzer.reset()
    assert analyzer.angle_fn.image_aspect == pytest.approx(WIDTH / HEIGHT)


# ---------------------------------------------------------
# Target, phases and reasons
# ---------------------------------------------------------

def test_target_of_60_is_applied():
    analyzer, summary = analyse(session([65, 55]))
    assert summary["rep_threshold"] == 60.0
    assert summary["reps"] == 1
    # The 55° attempt is recorded and scored as movement, not counted.
    assert summary["valid_segments"] == 2


def test_pending_reasons_follow_the_repetition():
    analyzer = ExerciseAnalyzer(FLEXION, "LEFT", TARGET_60)
    reasons = []
    for result, t in zip(session([90]), timestamps(60)):
        analyzer.update(result, t)
        reasons.append(analyzer.pending_reason())
    assert reasons[0] == "ready"
    assert any(r.startswith("raise to the target") for r in reasons)
    assert any(r.startswith("lower below 30°") for r in reasons)
    assert reasons[-1] == "ready"
    assert analyzer.reps == 1


def test_off_plane_repetition_is_announced_while_raised():
    analyzer = ExerciseAnalyzer(FLEXION, "LEFT", TARGET_60)
    frames = session([90], world_abduction=WORLD_SKEW,
                     image_abduction=WORLD_SKEW)
    for result, t in zip(frames[:35], timestamps(35)):
        analyzer.update(result, t)
    assert analyzer.stage == "up"
    assert analyzer.pending_reason().startswith("will NOT count: plane")


def test_unverified_repetition_reason():
    frames = [frame(v) for v in
              list(np.linspace(40, 100, 15)) + [100] * 10
              + list(np.linspace(100, 0, 20)) + [0.0] * 10]
    _, summary = analyse(frames)
    assert summary["rep_events"][0]["outcome"] == REP_PLANE_UNVERIFIED
    assert "could not be verified" in summary["rep_events"][0]["reason"]


def test_unmeasured_frames_are_reported():
    analyzer = ExerciseAnalyzer(FLEXION, "LEFT", TARGET_60)
    from src.simulation.synthetic_pose import empty_results
    analyzer.update(empty_results(), 0.0)
    assert analyzer.pending_reason() == "arm not measured (no_pose)"
    assert analyzer.debug_state()["angle"] is None


# ---------------------------------------------------------
# Diagnostics: overlay, live text, recorder
# ---------------------------------------------------------

def run_widget(frames, monkeypatch, overlay=True, record=None):
    import src.ui.live_pose_widget as live
    from src.simulation.synthetic_pose import FakeCamera, ScriptedPose

    if overlay:
        monkeypatch.setenv(live.DEBUG_OVERLAY_ENV, "1")
    widget = live.LivePoseWidget(FLEXION, "LEFT", parameters=TARGET_60,
                                 threaded=False)
    # Square frames: the synthetic landmarks are projected for aspect 1.
    widget.camera = FakeCamera(len(frames), width=240, height=240, fps=30)
    widget.pose = ScriptedPose(frames)
    if record:
        monkeypatch.setenv(live.RECORD_LANDMARKS_ENV, str(record))
        widget._start_recorder()
    texts = []
    for _ in frames:
        widget.update_frame()
        texts.append((widget.status_label.text(), widget.debug_label.text()))
    stats = widget.get_stats()
    widget.stop()
    return stats, texts


def test_overlay_shows_every_decision_input(qapp, monkeypatch):
    frames = session([90] * 2, world_abduction=WORLD_SKEW,
                     image_abduction=WORLD_SKEW)
    stats, texts = run_widget(frames, monkeypatch)
    debug = " ".join(d for _, d in texts)
    for fragment in ("elevation", "target 60°", "lower below 30°",
                     "image", "world", "expected 0±30°", "OUT OF RANGE",
                     "segment ACTIVE", "rep stage up", "will NOT count",
                     "last rep: not counted"):
        assert fragment in debug, fragment
    assert stats["reps"] == 0
    assert "Last repetition not counted: movement plane" in texts[-1][0]


def test_overlay_is_off_by_default(qapp, monkeypatch):
    import src.ui.live_pose_widget as live
    monkeypatch.delenv(live.DEBUG_OVERLAY_ENV, raising=False)
    stats, texts = run_widget(session([90]), monkeypatch, overlay=False)
    assert all(d == "" for _, d in texts)
    assert stats["reps"] == 1


def test_recording_replays_to_the_same_result(qapp, monkeypatch, tmp_path):
    path = tmp_path / "session.jsonl"
    frames = session([90] * 2, world_abduction=WORLD_SKEW)
    stats, _ = run_widget(frames, monkeypatch, overlay=False, record=path)

    [(header, recorded)] = load_sessions(path)
    assert header["exercise_id"] == "shoulder_flexion"
    assert len(recorded) == len(frames)
    report = trace(header, recorded)
    assert report["reps"] == stats["reps"] == 0
    assert report["off_plane_reps"] == stats["off_plane_reps"] == 2
    assert report["target"] == 60.0
    # The tool shows which landmarks moved the plane.
    assert abs(report["image_plane_quartiles"][1]) < 2
    assert report["world_plane_quartiles"][1] > 55


def test_recorder_failure_does_not_stop_analysis(qapp, monkeypatch,
                                                 tmp_path):
    frames = session([90])

    def broken(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(LandmarkRecorder, "record", broken)
    stats, _ = run_widget(frames, monkeypatch, overlay=False,
                          record=tmp_path / "x.jsonl")
    assert stats["reps"] == 1


# ---------------------------------------------------------
# Perspective: close-range frontal camera (live 2026-11-01)
# ---------------------------------------------------------
#
# The recording had the shoulders about 0.6 m from a 1280x720 webcam,
# hips out of frame, wrists leaving the frame, and the image shoulder
# width growing 31% during the raise (patient leaning in). Here: camera
# at shoulder height, centred, 0.65 m from the shoulders, the patient
# leaning 0.12 m closer by 90°.

LIVE_DISTANCE, LIVE_LEAN = 0.65, 0.12


def live_frame(side, flexion=0.0, abduction=0.0, depth_scale=1.0,
               depth_bias=0.0, distance=LIVE_DISTANCE):
    """
    Frontal close-range frame. depth_scale compresses MediaPipe's world
    depth of the moving arm (as observed live); depth_bias adds a
    spurious forward offset (negative z = towards the camera).
    """
    world_pose = pose(side, flexion, abduction)
    world = world_points(world_pose)
    distance -= LIVE_LEAN * min(max(flexion, abduction), 90.0) / 90.0
    image = project_perspective(world, distance=distance,
                                camera=(0.0, world[11, 1]),
                                aspect=WIDTH / HEIGHT)
    result = make_results(world_pose, image_points=image,
                          visibility_overrides={23: 0.01, 24: 0.01})
    shoulder, elbow = (11, 13) if side == "LEFT" else (12, 14)
    landmarks = result.pose_world_landmarks.landmark
    for index in (elbow, elbow + 2):
        landmarks[index].z = (
            landmarks[shoulder].z
            + (landmarks[index].z - landmarks[shoulder].z) * depth_scale
            + depth_bias
        )
    return result


def live_session(side, peaks, **kwargs):
    movement = kwargs.pop("movement", "flexion")
    return [live_frame(side, **{movement: v}, **kwargs)
            for peak in peaks for v in repetition_values(peak)]


def analyse_live(frames, side, horizontal_fov="default"):
    analyzer = ExerciseAnalyzer(FLEXION, side, TARGET_60)
    analyzer.set_image_size(WIDTH, HEIGHT)
    if horizontal_fov != "default":
        analyzer.angle_fn.horizontal_fov = horizontal_fov
    for result, t in zip(frames, timestamps(len(frames))):
        analyzer.update(result, t)
    return analyzer, analyzer.summary()


@pytest.mark.parametrize("side", ["LEFT", "RIGHT"])
def test_orthographic_image_model_reads_close_forward_raise_as_sideways(side):
    """Diagnostic image plane: an orthographic reading of a close-range
    image puts a pure forward raise far from 0°, the pinhole model does
    not. Neither changes the count (world plane decides)."""

    def run(horizontal_fov):
        analyzer = ExerciseAnalyzer(FLEXION, side, TARGET_60)
        analyzer.set_image_size(WIDTH, HEIGHT)
        if horizontal_fov != "default":
            analyzer.angle_fn.horizontal_fov = horizontal_fov
        planes = []
        for result in live_session(side, [90] * 3):
            analyzer.update(result, None)
            state = analyzer.debug_state()
            if state["angle"] and state["angle"] > 80:
                planes.append(state["image_plane"])
        return analyzer.reps, planes

    reps_ortho, ortho = run(None)
    reps_pinhole, pinhole = run("default")
    assert all(p > 45 for p in ortho)
    assert all(abs(p) < 10 for p in pinhole)
    assert reps_ortho == reps_pinhole == 3


@pytest.mark.parametrize("side", ["LEFT", "RIGHT"])
@pytest.mark.parametrize("depth_scale", [1.0, 0.4])
def test_close_range_forward_raise_counts(side, depth_scale):
    frames = live_session(side, [75, 90, 110], depth_scale=depth_scale)
    analyzer, summary = analyse_live(frames, side)
    assert summary["reps"] == 3
    assert summary["off_plane_reps"] == 0
    assert all(abs(e["plane"]) < 10 for e in summary["rep_events"])
    assert summary["peak_rom"] == pytest.approx(110.0, abs=2.0)
    assert summary["dtw_similarity"] is not None
    # Leaning in is followed: the shoulder came 0.12 m closer.
    assert analyzer.debug_state()["camera_distance"] == pytest.approx(
        LIVE_DISTANCE, abs=0.02)


@pytest.mark.parametrize("side", ["LEFT", "RIGHT"])
def test_close_range_sideways_abduction_is_rejected(side):
    frames = live_session(side, [90, 110], movement="abduction")
    _, summary = analyse_live(frames, side)
    assert summary["reps"] == 0
    assert summary["off_plane_reps"] == 2
    assert summary["plane_unverified_reps"] == 0
    assert all(e["plane"] == pytest.approx(90.0, abs=2.0)
               for e in summary["rep_events"])


@pytest.mark.parametrize("side", ["LEFT", "RIGHT"])
@pytest.mark.parametrize("bias", [-0.03, -0.06])
def test_close_range_abduction_with_spurious_forward_depth_never_counts(
        side, bias):
    """Close to the camera, the image of a sideways arm also fits an arm
    in front of the body. If MediaPipe's depth wrongly suggests forward,
    the repetition is not counted (direction unverified)."""
    frames = live_session(side, [90, 110], movement="abduction",
                          depth_bias=bias)
    _, summary = analyse_live(frames, side)
    assert summary["reps"] == 0
    assert summary["off_plane_reps"] == 2


@pytest.mark.parametrize("side", ["LEFT", "RIGHT"])
def test_close_range_diagonal_raise_is_rejected(side):
    frames = [live_frame(side, flexion=v, abduction=v * 0.75)
              for v in repetition_values(100) * 2]
    _, summary = analyse_live(frames, side)
    assert summary["reps"] == 0
    assert summary["off_plane_reps"] == 2
    assert all(e["plane"] > 30 for e in summary["rep_events"]
               if e["plane"] is not None)


def test_sideways_arm_with_ambiguous_image_is_flagged():
    """The diagnostic image plane flags the close-range ambiguity; the
    decision (world plane) still reads the arm as sideways."""
    measure = CalibratedHumeralElevation()
    measure.set_image_size(WIDTH, HEIGHT)
    for _ in range(measure.rest_frames):
        measure(live_frame("LEFT"), "LEFT")
    reading = measure(live_frame("LEFT", abduction=90, depth_bias=-0.03),
                      "LEFT")
    assert measure.last_details["image_plane_status"] == PLANE_DEPTH_AMBIGUOUS
    assert reading.plane > 60


@pytest.mark.parametrize("value, expected", [
    ("70", 70.0), ("", angles.DEFAULT_CAMERA_HORIZONTAL_FOV),
    ("wide", angles.DEFAULT_CAMERA_HORIZONTAL_FOV),
    ("5", angles.DEFAULT_CAMERA_HORIZONTAL_FOV),
])
def test_camera_field_of_view_setting(monkeypatch, value, expected):
    monkeypatch.setenv(angles.CAMERA_HFOV_ENV, value)
    assert angles._camera_horizontal_fov() == expected


def test_overlay_shows_the_camera_distance(qapp, monkeypatch):
    from src.ui.live_pose_widget import format_debug_state
    analyzer, _ = analyse_live(live_session("LEFT", [90])[:30], "LEFT")
    assert "camera ~0.5 m" in format_debug_state(analyzer.debug_state())
