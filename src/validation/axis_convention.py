"""
Empirical check of the MediaPipe world-landmark axis convention that
src/exercises/angles.py assumes (x towards image right, y down, z away
from the camera) and of the trunk frame's forward axis.

Runs real MediaPipe Pose on a video file (or the webcam) and, for every
sampled frame with a clearly visible person, checks:

  facing the camera (subject's left shoulder on the image right):
    x      world x of left shoulder > right shoulder (follows image x)
    y      shoulders above hips has smaller world y (y points down)
    nose_z nose nearer the camera than the shoulders (z points away)
    fwd    trunk_frame() forward axis points towards the camera (-z)
  facing away:
    back_fwd forward axis points away from the camera (+z)

This verifies coordinate CONVENTIONS on real MediaPipe output. It does
not measure angle ACCURACY; that needs a reference measurement (see
docs/validation_protocol.md).

Usage:
    python -m src.validation.axis_convention --video path/to/video.mp4
    python -m src.validation.axis_convention --camera 0 --frames 150
"""

import argparse
import json

VISIBILITY = 0.8
FACING_MARGIN = 0.02


def check_frame(results):
    """Per-frame outcome dict, or None when the frame is not usable."""

    from src.exercises.angles import trunk_frame, world_array

    if results is None or results.pose_landmarks is None:
        return None
    image = results.pose_landmarks.landmark
    points = world_array(results)
    if points is None:
        return None
    if min(image[k].visibility for k in (0, 11, 12, 23, 24)) < VISIBILITY:
        return None
    frame = trunk_frame(points)
    if frame is None:
        return None
    _, _, forward = frame

    if image[11].x > image[12].x + FACING_MARGIN:
        shoulder_y = (points[11, 1] + points[12, 1]) / 2
        hip_y = (points[23, 1] + points[24, 1]) / 2
        return {
            "facing": True,
            "x": bool(points[11, 0] > points[12, 0]),
            "y": bool(shoulder_y < hip_y),
            "nose_z": bool(points[0, 2] < (points[11, 2] + points[12, 2]) / 2),
            "fwd": bool(forward[2] < 0),
        }
    if image[12].x > image[11].x + FACING_MARGIN:
        return {"facing": False, "back_fwd": bool(forward[2] > 0)}
    return None


def summarise(outcomes):
    facing = [o for o in outcomes if o["facing"]]
    away = [o for o in outcomes if not o["facing"]]
    summary = {"frames_facing": len(facing), "frames_away": len(away)}
    for key in ("x", "y", "nose_z", "fwd"):
        summary[key] = sum(o[key] for o in facing)
    summary["back_fwd"] = sum(o["back_fwd"] for o in away)
    summary["all_consistent"] = bool(facing) and all(
        summary[k] == len(facing) for k in ("x", "y", "nose_z", "fwd")
    ) and summary["back_fwd"] == len(away)
    return summary


def main():
    import cv2
    import mediapipe as mp

    from src.pose_estimation.config import model_complexity

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video")
    source.add_argument("--camera", type=int)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--every", type=int, default=5)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    pose = mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=model_complexity(),
        min_detection_confidence=0.5,
    )
    outcomes = []
    try:
        for index in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                break
            if index % args.every:
                continue
            outcome = check_frame(
                pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            )
            if outcome is not None:
                outcomes.append(outcome)
    finally:
        pose.close()
        cap.release()

    print(json.dumps(summarise(outcomes), indent=2))


if __name__ == "__main__":
    main()
