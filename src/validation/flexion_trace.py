"""
Replay a recorded live session (see landmark_recorder.py) through the
real ExerciseAnalyzer and explain every repetition.

It also replays the geometric compensation detector with the same
wiring as the live widget. For shoulder elevation exercises it compares, over the elevated
frames, the movement plane computed from MediaPipe WORLD landmarks with
the one computed from the 2D IMAGE landmarks. If they disagree, the 3D
reconstruction (not the movement) is what moves the plane; if both put
a "forward" raise far from 0°, the arm really moved diagonally relative
to the camera, or the patient was not facing it.

This explains what the software did with real MediaPipe output. It does
not measure accuracy: that needs a reference measurement (goniometer or
inclinometer, docs/validation_protocol.md).

Usage:
    python -m src.validation.flexion_trace recording.jsonl
"""

import argparse
import json
from statistics import median

from src.compensation_detection.geometric_detector import (
    GeometricCompensationDetector,
)
from src.exercises.analyzer import STATUS_OK, ExerciseAnalyzer
from src.exercises.catalogue import get_exercise_config
from src.validation.landmark_recorder import load_sessions


def _quartiles(values):
    values = sorted(values)
    if not values:
        return None
    n = len(values)
    return (round(values[n // 4], 1), round(median(values), 1),
            round(values[(3 * n) // 4], 1))


def trace(header, frames):
    """Replay one session; returns a JSON-serialisable report."""

    config = get_exercise_config(header["exercise_id"])
    analyzer = ExerciseAnalyzer(config, header["side"],
                                header.get("parameters"))
    # Same wiring as the live widget: attempt signal from segmentation,
    # baseline only while the exercise arm is measured and at rest.
    detector = GeometricCompensationDetector(header["side"],
                                             config.exercise_id)

    world, image, distance = [], [], []
    for frame in frames:
        if frame["frame_size"]:
            analyzer.set_image_size(*frame["frame_size"])
        analysis = analyzer.update(frame["results"], frame["t"])
        detector.update(
            frame["results"], analysis.movement_active,
            arm_at_rest=(analysis.status == STATUS_OK
                         and not analysis.movement_active),
        )
        state = analyzer.debug_state()
        if (
            analysis.angle is not None
            and config.plane_check is not None
            and analysis.angle >= config.plane_check.min_elevation
        ):
            if state["world_plane"] is not None:
                world.append(state["world_plane"])
            if state["image_plane"] is not None:
                image.append(state["image_plane"])
            if state["camera_distance"] is not None:
                distance.append(state["camera_distance"])

    summary = analyzer.summary()
    return {
        "exercise_id": config.exercise_id,
        "side": analyzer.side,
        "target": analyzer.up_threshold,
        "frames": len(frames),
        "reps": summary["reps"],
        "off_plane_reps": summary["off_plane_reps"],
        "plane_unverified_reps": summary.get("plane_unverified_reps"),
        "repetitions": [
            {k: e[k] for k in ("t_s", "outcome", "peak", "plane", "reason")}
            for e in summary.get("rep_events", [])
        ],
        # (lower quartile, median, upper quartile) over elevated frames.
        "world_plane_quartiles": _quartiles(world),
        "image_plane_quartiles": _quartiles(image),
        # Shoulder-to-camera distance (MediaPipe world units, roughly
        # metres) under the assumed camera field of view.
        "camera_distance_quartiles": _quartiles(distance),
        "tracking": summary["tracking"],
        "interrupted_reps": summary.get("interrupted_reps"),
        "bridged_gaps": summary.get("bridged_gaps"),
        "abandoned_segments": summary.get("abandoned_segments"),
        "resting_angle": summary.get("resting_angle"),
        "compensation": _compensation_report(detector.summary()),
    }


def _compensation_report(summary):
    return {
        "baseline_established": summary["baseline_established"],
        "frames_active": summary["frames_active"],
        "frames_resting": summary["frames_resting"],
        "patterns": {
            name: {k: p[k] for k in (
                "assessed", "frames", "detected_fraction", "episodes",
                "baseline", "baseline_updates", "max_abs_deviation",
                "session_start_baseline", "max_abs_rest_drift",
                "session_start", "frames_held_through_dropout",
                "unable_reasons")}
            for name, p in summary["patterns"].items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("recording")
    args = parser.parse_args()
    for header, frames in load_sessions(args.recording):
        print(json.dumps(trace(header, frames), indent=2, default=str))


if __name__ == "__main__":
    main()
