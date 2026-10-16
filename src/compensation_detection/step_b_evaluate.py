"""
Step B: external validation of FROZEN Step A models on labelled
MediaPipe webcam recordings (see docs/step_b_protocol.md).

Pipeline per recording:
  * keep frames with a detected pose and all 8 shared joints visible;
  * split into contiguous segments at gaps longer than MAX_GAP_S;
  * resample each segment to 30 Hz (the Toronto training rate);
  * shared_v1 features (identical code to training);
  * 20-frame windows whose frames all carry the same label 1-4
    (label 0 = unlabelled frames are excluded).

Labels come from the operator labels stored in the recording, or from a
reviewer annotation CSV (recording_id,start_s,end_s,label,annotator)
that replaces them when given.

Outputs pooled and per-participant macro-F1 with a participant-level
bootstrap 95% CI, against a majority-class baseline. No model is
trained or tuned here.

Usage:
    python -m src.compensation_detection.step_b_evaluate \
        --recordings data/step_b [--annotations annotations.csv]
"""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score

from src.compensation_detection.shared_features import (
    FEATURE_NAMES,
    FEATURE_SET,
    MEDIAPIPE,
    TARGET_RATE_HZ,
    mediapipe_to_shared,
    resample,
    sequence_features,
)
from src.compensation_detection.step_a_compare import (
    BiLSTMClassifier,
    LABELS,
    OUTPUT_DIR as STEP_A_DIR,
    WINDOW,
    predict_bilstm,
    window_summary,
)
from src.data_collection.recorder import load_recording

MAX_GAP_S = 0.2
VISIBILITY = 0.5
MAJORITY_LABEL = 1  # most frequent class in the Toronto training data

SHARED_INDICES = sorted(
    index for side in MEDIAPIPE.values() for index in side.values()
)


def read_annotations(path):
    """{recording_id: [(start_s, end_s, label)]} from a reviewer CSV."""
    annotations = {}
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            label = int(row["label"])
            if label not in LABELS:
                raise ValueError(f"Invalid label {label} in {path}")
            annotations.setdefault(row["recording_id"], []).append(
                (float(row["start_s"]), float(row["end_s"]), label)
            )
    return annotations


def frame_labels(recording, annotations):
    """Per-frame labels: reviewer intervals if available, else operator."""

    recording_id = recording["metadata"]["recording_id"]
    if annotations is None or recording_id not in annotations:
        return recording["operator_label"].astype(int)

    relative = recording["timestamps"] - recording["timestamps"][0]
    labels = np.zeros(len(relative), dtype=int)
    for start, end, label in annotations[recording_id]:
        labels[(relative >= start) & (relative < end)] = label
    return labels


def valid_segments(recording):
    """Index arrays of contiguous usable frames."""

    usable = recording["detected"].astype(bool) & np.all(
        recording["visibility"][:, SHARED_INDICES] >= VISIBILITY, axis=1
    ) & np.all(np.isfinite(recording["world"]), axis=(1, 2))

    times = recording["timestamps"]
    segments, current = [], []
    for i in np.flatnonzero(usable):
        if current and (i != current[-1] + 1
                        or times[i] - times[current[-1]] > MAX_GAP_S):
            segments.append(np.array(current))
            current = []
        current.append(i)
    if current:
        segments.append(np.array(current))
    return [s for s in segments if len(s) >= 2]


def recording_windows(recording, annotations=None):
    """(X windows, y labels) for one recording."""

    side = recording["metadata"]["side"]
    labels = frame_labels(recording, annotations)
    X, y = [], []

    for segment in valid_segments(recording):
        times = recording["timestamps"][segment]
        new_times, world = resample(times, recording["world"][segment],
                                    TARGET_RATE_HZ)
        # Label of the most recent original frame (no interpolation).
        source = np.searchsorted(times, new_times, side="right") - 1
        segment_labels = labels[segment][np.clip(source, 0, None)]

        features = sequence_features(
            mediapipe_to_shared(world, side), side).astype(np.float32)

        for start in range(0, len(features) - WINDOW + 1):
            window_labels = segment_labels[start:start + WINDOW]
            if window_labels[0] == 0 or not np.all(
                window_labels == window_labels[0]
            ):
                continue
            X.append(features[start:start + WINDOW])
            y.append(int(window_labels[0]))

    if not X:
        return (np.empty((0, WINDOW, len(FEATURE_NAMES)), dtype=np.float32),
                np.empty(0, dtype=int))

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int)


def bootstrap_ci(y, predictions, groups, rng, iterations=2000):
    participants = np.unique(groups)
    scores = []
    for _ in range(iterations):
        sample = rng.choice(participants, len(participants), replace=True)
        mask = np.concatenate([np.flatnonzero(groups == p) for p in sample])
        scores.append(f1_score(y[mask], predictions[mask], labels=LABELS,
                               average="macro", zero_division=0))
    return [float(np.percentile(scores, 2.5)),
            float(np.percentile(scores, 97.5))]


def evaluate(X, y, groups, predictions_by_model, seed=0):
    rng = np.random.default_rng(seed)
    report = {}
    for name, predictions in predictions_by_model.items():
        report[name] = {
            "pooled_macro_f1": float(f1_score(
                y, predictions, labels=LABELS, average="macro",
                zero_division=0)),
            "macro_f1_95ci_participant_bootstrap": (
                bootstrap_ci(y, predictions, groups, rng)
                if len(np.unique(groups)) > 1 else None),
            "per_class_f1": [float(v) for v in f1_score(
                y, predictions, labels=LABELS, average=None,
                zero_division=0)],
            "per_participant_macro_f1": {
                p: float(f1_score(
                    y[groups == p], predictions[groups == p],
                    labels=sorted(set(y[groups == p])), average="macro",
                    zero_division=0))
                for p in np.unique(groups)
            },
        }
    return report


def load_frozen_models(model_dir):
    rf_bundle = joblib.load(model_dir / "rf_shared_v1.joblib")
    checkpoint = torch.load(model_dir / "bilstm_shared_v1.pt",
                            map_location="cpu", weights_only=False)
    for bundle in (rf_bundle, checkpoint):
        if bundle["feature_set"] != FEATURE_SET:
            raise ValueError("Model feature set does not match shared_v1")
    model = BiLSTMClassifier(checkpoint["input_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return rf_bundle["model"], (model, checkpoint["mean"], checkpoint["std"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--models", type=Path,
                        default=STEP_A_DIR / "final_models_not_deployed")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    annotations = (read_annotations(args.annotations)
                   if args.annotations else None)

    X_all, y_all, groups = [], [], []
    for npz in sorted(args.recordings.glob("*.npz")):
        recording = load_recording(npz)
        X, y = recording_windows(recording, annotations)
        if len(y):
            X_all.append(X)
            y_all.append(y)
            groups += [recording["metadata"]["participant"]] * len(y)

    if not X_all:
        raise SystemExit("No labelled windows found.")

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    groups = np.asarray(groups)

    rf, (bilstm, mean, std) = load_frozen_models(args.models)
    device = torch.device("cpu")
    predictions = {
        "majority_class": np.full_like(y, MAJORITY_LABEL),
        "rf_shared_v1": rf.predict(window_summary(X)),
        "bilstm_shared_v1": predict_bilstm(bilstm.to(device), X, mean, std,
                                           device),
    }

    report = {
        "windows": int(len(y)),
        "participants": sorted(set(groups.tolist())),
        "label_counts": {str(k): int((y == k).sum()) for k in LABELS},
        "label_source": "reviewer annotations" if annotations
        else "operator labels",
        "results": evaluate(X, y, groups, predictions),
    }

    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
