"""
Temporary diagnostic recorder for live sessions.

Enabled by setting REHAB_RECORD_LANDMARKS to a file path before starting
the application. Every processed frame is appended to that JSON Lines
file: timestamp, frame size, MediaPipe image and world landmarks (x, y,
z, visibility) and the analyzer's debug state. The recording can be
replayed offline through the real analyzer with
`python -m src.validation.flexion_trace <file>`.

Recordings contain body landmarks only (no images), but they still
describe a person's movement: store them as patient data.
"""

import json

from mediapipe.framework.formats import landmark_pb2

from src.simulation.synthetic_pose import SyntheticResults


def _landmarks(landmark_list):
    if landmark_list is None:
        return None
    return [
        [round(lm.x, 6), round(lm.y, 6), round(lm.z, 6),
         round(lm.visibility, 4)]
        for lm in landmark_list.landmark
    ]


class LandmarkRecorder:

    def __init__(self, path, meta):
        self.path = path
        self._file = open(path, "a", encoding="utf-8")
        self._write({"type": "header", **meta})

    def _write(self, record):
        self._file.write(json.dumps(record, default=str) + "\n")

    def record(self, timestamp, frame_size, results, state):
        self._write({
            "type": "frame",
            "t": timestamp,
            "frame_size": frame_size,
            "image": _landmarks(getattr(results, "pose_landmarks", None)),
            "world": _landmarks(
                getattr(results, "pose_world_landmarks", None)),
            "state": state,
        })

    def close(self):
        if not self._file.closed:
            self._file.close()


def _landmark_list(values, cls):
    if values is None:
        return None
    out = cls()
    for x, y, z, visibility in values:
        out.landmark.add(x=x, y=y, z=z, visibility=visibility)
    return out


def load_recording(path):
    """
    (header, frames) where frames is a list of dicts with keys t,
    frame_size, results (MediaPipe-like object) and state. Uses the last
    header in the file (one file may hold several sessions; see
    load_sessions).
    """
    sessions = load_sessions(path)
    if not sessions:
        raise ValueError(f"{path}: no recorded session")
    return sessions[-1]


def load_sessions(path):
    """All (header, frames) sessions in a recording file, in order."""
    sessions = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["type"] == "header":
                sessions.append((record, []))
            elif sessions:
                sessions[-1][1].append({
                    "t": record["t"],
                    "frame_size": record["frame_size"],
                    "results": SyntheticResults(
                        _landmark_list(record["image"],
                                       landmark_pb2.NormalizedLandmarkList),
                        _landmark_list(record["world"],
                                       landmark_pb2.LandmarkList),
                    ),
                    "state": record.get("state"),
                })
    return sessions
