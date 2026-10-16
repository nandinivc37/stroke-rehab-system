"""
Step B landmark recorder (see docs/step_b_protocol.md).

Records MediaPipe Pose output for protocol-defined trials, with the
operator marking the intended/observed compensation live via number
keys. By default only landmarks are stored, never video.

Per frame it stores: monotonic timestamp, pose-detected flag, world
landmarks (33x3, metres), image landmarks (33x3, normalized), visibility
(33) and the operator label in effect (0 = unlabelled, 1-4 = class).

Usage (needs a webcam; run from the project root):
    python -m src.data_collection.recorder --participant S01 \
        --task forward_reach --side LEFT --out data/step_b

Keys during recording:
    space  start / pause recording
    0      operator label: unlabelled (e.g. setting up, resting)
    1      no compensation        2  shoulder elevation
    3      trunk rotation         4  lean forward
    q/Esc  stop and save
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROTOCOL_VERSION = "step_b_v1"

LABEL_NAMES = {
    0: "Unlabelled",
    1: "No Compensation",
    2: "Shoulder Elevation",
    3: "Trunk Rotation",
    4: "Lean Forward",
}

TASKS = (
    "forward_reach",       # Toronto-like task
    "side_to_side_reach",  # Toronto-like task
    "shoulder_flexion",
    "shoulder_abduction",
)


class LandmarkRecorder:
    """Accumulates frames in memory; Qt/OpenCV-free for testing."""

    def __init__(self):
        self.timestamps = []
        self.detected = []
        self.world = []
        self.image = []
        self.visibility = []
        self.operator_label = []

    def __len__(self):
        return len(self.timestamps)

    def add_frame(self, results, timestamp, label):
        if label not in LABEL_NAMES:
            raise ValueError(f"Unknown label {label!r}")

        detected = (
            results is not None
            and results.pose_landmarks is not None
            and results.pose_world_landmarks is not None
        )

        if detected:
            image = np.array(
                [[p.x, p.y, p.z] for p in results.pose_landmarks.landmark]
            )
            world = np.array(
                [[p.x, p.y, p.z] for p in results.pose_world_landmarks.landmark]
            )
            visibility = np.array(
                [p.visibility for p in results.pose_landmarks.landmark]
            )
        else:
            image = np.full((33, 3), np.nan)
            world = np.full((33, 3), np.nan)
            visibility = np.zeros(33)

        self.timestamps.append(float(timestamp))
        self.detected.append(bool(detected))
        self.world.append(world)
        self.image.append(image)
        self.visibility.append(visibility)
        self.operator_label.append(int(label))

    def save(self, directory, metadata):
        """Write <recording_id>.npz and <recording_id>.json."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        recording_id = metadata["recording_id"]
        npz_path = directory / f"{recording_id}.npz"
        json_path = directory / f"{recording_id}.json"

        if npz_path.exists() or json_path.exists():
            raise FileExistsError(f"{recording_id} already exists")

        np.savez_compressed(
            npz_path,
            timestamps=np.asarray(self.timestamps),
            detected=np.asarray(self.detected),
            world=np.asarray(self.world, dtype=np.float32),
            image=np.asarray(self.image, dtype=np.float32),
            visibility=np.asarray(self.visibility, dtype=np.float32),
            operator_label=np.asarray(self.operator_label, dtype=np.int8),
        )

        metadata = dict(metadata)
        metadata.update({
            "protocol_version": PROTOCOL_VERSION,
            "frames": len(self),
            "frames_detected": int(sum(self.detected)),
            "label_names": LABEL_NAMES,
            "video_saved": False,
        })
        if len(self.timestamps) > 1:
            duration = self.timestamps[-1] - self.timestamps[0]
            metadata["duration_s"] = duration
            metadata["mean_fps"] = (len(self) - 1) / duration if duration else None

        json_path.write_text(json.dumps(metadata, indent=2))
        return npz_path, json_path


def load_recording(npz_path):
    npz_path = Path(npz_path)
    data = dict(np.load(npz_path))
    data["metadata"] = json.loads(npz_path.with_suffix(".json").read_text())
    return data


def make_recording_id(participant, task, side):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{participant}_{task}_{side}_{stamp}"


def main():  # pragma: no cover - requires a webcam
    import cv2
    import mediapipe as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", required=True,
                        help="Pseudonymous code only (e.g. S01). No names.")
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--side", required=True, choices=["LEFT", "RIGHT"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--operator", required=True,
                        help="Operator code (for annotation provenance).")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not camera.isOpened():
        raise SystemExit("Could not open camera")

    # Same settings as the application's live widget.
    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=2, smooth_landmarks=True,
        min_detection_confidence=0.7, min_tracking_confidence=0.7,
    )

    recorder = LandmarkRecorder()
    recording = False
    label = 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera read failed; stopping.")
                break

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if recording:
                recorder.add_frame(results, time.monotonic(), label)

            display = frame.copy()
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    display, results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS)
            display = cv2.flip(display, 1)
            state = "REC" if recording else "PAUSED"
            cv2.putText(display, f"{state}  label: {LABEL_NAMES[label]}  "
                        f"frames: {len(recorder)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Step B recorder", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                recording = not recording
            if ord("0") <= key <= ord("4"):
                label = key - ord("0")
    finally:
        camera.release()
        pose.close()
        cv2.destroyAllWindows()

    if len(recorder) == 0:
        print("Nothing recorded.")
        return

    recording_id = make_recording_id(args.participant, args.task, args.side)
    npz, meta = recorder.save(args.out, {
        "recording_id": recording_id,
        "participant": args.participant,
        "task": args.task,
        "side": args.side,
        "operator": args.operator,
        "notes": args.notes,
        "camera_resolution": [1280, 720],
        "mediapipe_settings": {"model_complexity": 2,
                               "smooth_landmarks": True},
    })
    print(f"Saved {npz} and {meta}")


if __name__ == "__main__":
    main()
