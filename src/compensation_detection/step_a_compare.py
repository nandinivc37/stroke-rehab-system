"""
Step A: compare compensation classifiers on the Toronto Rehab Stroke
Pose dataset using a MediaPipe-compatible feature set.

Models (all evaluated with leave-one-participant-out, LOPO):
  1. RF  on the ORIGINAL 21 Kinect features   (reference: signal ceiling
     of the old, MediaPipe-incompatible pipeline)
  2. RF  on shared_v1 features                (lightweight baseline)
  3. BiLSTM on shared_v1 features             (same architecture as the
     existing model, retrained; existing weights are NOT modified)
plus majority-class and label-frequency baselines.

The BiLSTM uses an inner validation split of training participants for
early stopping; the held-out participant is never used for any choice.
RF hyperparameters are fixed in advance (no tuning).

IMPORTANT: every number produced here is Kinect-to-Kinect performance.
It says nothing about accuracy on MediaPipe webcam data (Step B) and
nothing about clinical validity. Most compensation examples in this
dataset are performed deliberately by healthy participants.

Per-fold results are checkpointed atomically; rerunning the same
command resumes from the completed folds (see run_lopo / check_manifest).

Usage (from the project root):
    python -m src.compensation_detection.step_a_compare            # all
    python -m src.compensation_detection.step_a_compare --models bilstm
    # smoke test (scratch directory, never writes metrics):
    python -m src.compensation_detection.step_a_compare --models bilstm \
        --only-folds H01 --run-dir /tmp/step_a_smoke
"""

import argparse
import hashlib
import inspect
import json
import os
try:
    import resource  # POSIX only: peak-memory logging
except ImportError:  # Windows
    resource = None
import tempfile
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.compensation_detection.preprocess_toronto import find_trials, load_trial
from src.compensation_detection.shared_features import (
    FEATURE_NAMES,
    FEATURE_SET,
    active_side_from_trial,
    kinect_to_shared,
    sequence_features,
)
from src.paths import DATA_DIR, OUTPUTS_DIR

TORONTO_DIR = DATA_DIR / "toronto" / "data_new"
ORIGINAL_WINDOWS = DATA_DIR / "processed_toronto" / "compensation_temporal_windows.npz"
OUTPUT_DIR = OUTPUTS_DIR / "compensation_step_a"

WINDOW = 20
CLASS_NAMES = ["No Compensation", "Shoulder Elevation", "Trunk Rotation",
               "Lean Forward"]
LABELS = [1, 2, 3, 4]
SEED = 42


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

def build_shared_windows():
    """20-frame, label-homogeneous windows of shared_v1 features."""

    X, y, groups, trials = [], [], [], []

    for trial_path in find_trials(TORONTO_DIR):
        trial = load_trial(trial_path)
        side = active_side_from_trial(trial["trial"])
        features = sequence_features(
            kinect_to_shared(trial["positions"], side), side
        ).astype(np.float32)
        labels = trial["labels"]

        for start in range(0, len(labels) - WINDOW + 1):
            window_labels = labels[start:start + WINDOW]
            if not np.all(window_labels == window_labels[0]):
                continue
            X.append(features[start:start + WINDOW])
            y.append(int(window_labels[0]))
            groups.append(trial["participant"])
            trials.append(trial["trial"])

    return (np.asarray(X, dtype=np.float32), np.asarray(y),
            np.asarray(groups), np.asarray(trials))


def load_original_windows():
    data = np.load(ORIGINAL_WINDOWS)
    return (data["X"].astype(np.float32), data["y"], data["participants"],
            data["trials"])


def window_summary(X):
    """Per-feature mean, std, min, max and last value of each window."""
    return np.concatenate(
        [X.mean(1), X.std(1), X.min(1), X.max(1), X[:, -1, :]], axis=1
    )


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

def make_rf():
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=4,
        random_state=SEED,
    )


class BiLSTMClassifier(nn.Module):
    """Same architecture as the existing compensation BiLSTM."""

    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 num_classes=4, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.classifier(self.dropout(output[:, -1, :]))


def standardizer(X):
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(0)
    std = flat.std(0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def train_bilstm(X_train, y_train, X_val, y_val, device, max_epochs,
                 patience=5, fixed_epochs=None, history=None):
    """
    Train with class-weighted cross-entropy. With validation data,
    keep the epoch with the best validation macro-F1 (early stopping).
    With fixed_epochs, train exactly that many epochs (final model).

    history: optional list; one dict per epoch (seconds, validation
    macro-F1) is appended. Instrumentation only.
    """

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    mean, std = standardizer(X_train)
    norm = lambda a: (a - mean) / std  # noqa: E731

    model = BiLSTMClassifier(X_train.shape[-1]).to(device)
    counts = np.bincount(y_train - 1, minlength=4).astype(np.float64)
    weights = len(y_train) / (4 * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=1e-4)

    Xt = torch.tensor(norm(X_train), dtype=torch.float32)
    yt = torch.tensor(y_train - 1, dtype=torch.long)

    best = (-1.0, None, 0)
    stale = 0
    epochs = fixed_epochs or max_epochs

    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        model.train()
        order = torch.randperm(len(Xt))
        for i in range(0, len(order), 128):
            idx = order[i:i + 128]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if fixed_epochs:
            if history is not None:
                history.append({"epoch": epoch,
                                "seconds": time.time() - epoch_started})
            continue

        predictions = predict_bilstm(model, X_val, mean, std, device)
        score = f1_score(y_val, predictions, labels=LABELS, average="macro",
                         zero_division=0)
        if history is not None:
            history.append({"epoch": epoch,
                            "seconds": time.time() - epoch_started,
                            "val_macro_f1": float(score)})
        if score > best[0]:
            best = (score, {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}, epoch)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if not fixed_epochs:
        model.load_state_dict(best[1])

    return model, mean, std, (best[2] if not fixed_epochs else epochs)


def predict_bilstm(model, X, mean, std, device):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.tensor((X[i:i + 512] - mean) / std,
                              dtype=torch.float32, device=device)
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out) + 1


def inner_validation_participants(train_participants, rng):
    """
    Up to two healthy + one patient participant for early stopping,
    always leaving at least one participant of each group to train on.
    """
    healthy = sorted(p for p in train_participants if p.startswith("H"))
    patients = sorted(p for p in train_participants if p.startswith("P"))
    chosen = list(rng.choice(healthy, min(2, max(0, len(healthy) - 1)),
                             replace=False))
    if len(patients) > 1:
        chosen.append(rng.choice(patients))
    if not chosen:
        chosen = [rng.choice(sorted(train_participants))]
    return set(chosen)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def summarize(name, y_true, y_pred, groups):
    present = sorted(set(LABELS))
    per_participant = {}
    for participant in sorted(set(groups)):
        mask = groups == participant
        labels_here = sorted(set(y_true[mask]))
        per_participant[participant] = {
            "windows": int(mask.sum()),
            "classes_present": [int(v) for v in labels_here],
            "macro_f1_present_classes": float(f1_score(
                y_true[mask], y_pred[mask], labels=labels_here,
                average="macro", zero_division=0)),
            "accuracy": float((y_true[mask] == y_pred[mask]).mean()),
        }

    def subset(prefix):
        mask = np.char.startswith(groups.astype(str), prefix)
        return float(f1_score(y_true[mask], y_pred[mask], labels=present,
                              average="macro", zero_division=0))

    return {
        "model": name,
        "pooled_macro_f1": float(f1_score(y_true, y_pred, labels=present,
                                          average="macro", zero_division=0)),
        "pooled_accuracy": float((y_true == y_pred).mean()),
        "per_class_f1": dict(zip(CLASS_NAMES, map(float, f1_score(
            y_true, y_pred, labels=present, average=None, zero_division=0)))),
        "healthy_macro_f1": subset("H"),
        "patient_macro_f1": subset("P"),
        "confusion_matrix": confusion_matrix(y_true, y_pred,
                                             labels=present).tolist(),
        "report": classification_report(y_true, y_pred, labels=present,
                                        target_names=CLASS_NAMES, digits=4,
                                        zero_division=0),
        "per_participant": per_participant,
    }




# ---------------------------------------------------------
# Fold checkpointing and resume
# ---------------------------------------------------------
#
# Each completed participant fold is saved as
#   <run_dir>/<model>/fold_<participant>.npz
# written atomically (temporary file, fsync, rename), so an interruption
# leaves either a complete fold or none. A run manifest stores one
# fingerprint per model covering the data arrays, labels, participant
# split, configuration and the source code of every function that
# affects training or prediction. Resuming with anything different is
# refused rather than silently mixed.

MODELS = ("rf_original", "rf_shared", "bilstm")
DEFAULT_RUN_DIR = OUTPUT_DIR / "lopo_run"
PATIENCE = 5  # early-stopping patience (train_bilstm default)


class CheckpointMismatch(RuntimeError):
    pass


def _log(message):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _sha256_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        if array.dtype.kind in "US":
            data = "\n".join(map(str, array.tolist())).encode()
        else:
            data = array.tobytes()
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(data)
    return digest.hexdigest()


def _sources(*functions):
    return "\n".join(inspect.getsource(f) for f in functions)


def model_fingerprint(model, X, y, groups, max_epochs):
    """Hash of everything that determines a model's fold results."""

    shared = _sources(run_lopo, fit_predict_fold, inner_validation_participants)
    if model == "bilstm":
        code = _sources(train_bilstm, predict_bilstm, BiLSTMClassifier,
                        standardizer)
        config = {"max_epochs": max_epochs, "patience": PATIENCE,
                  "feature_set": FEATURE_SET,
                  "feature_names": list(FEATURE_NAMES)}
    else:
        code = _sources(make_rf, window_summary)
        config = {"feature_set": (FEATURE_SET if model == "rf_shared"
                                  else "original_21_kinect")}

    config.update({"model": model, "window": WINDOW, "seed": SEED,
                   "labels": LABELS})

    digest = hashlib.sha256()
    digest.update(_sha256_arrays(X, y, groups).encode())
    digest.update(json.dumps(config, sort_keys=True).encode())
    digest.update(shared.encode())
    digest.update(code.encode())
    return digest.hexdigest()


def _fsync_dir(directory):
    # Windows cannot open a directory as a file; os.replace is atomic on
    # NTFS and there is no separate directory-entry flush to request.
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path, write):
    """Write via a temporary file in the same directory, then rename."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as file:
            write(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path, data):
    atomic_write_bytes(
        path, lambda f: f.write(json.dumps(data, indent=2).encode()))


def fold_path(run_dir, model, participant):
    return Path(run_dir) / model / f"fold_{participant}.npz"


def save_fold(path, fingerprint, participant, test_indices, y_true,
              predictions, meta):
    meta = dict(meta, fingerprint=fingerprint, participant=participant)
    atomic_write_bytes(path, lambda f: np.savez(
        f,
        test_indices=test_indices.astype(np.int64),
        y_true=y_true.astype(np.int64),
        predictions=predictions.astype(np.int64),
        meta=np.array(json.dumps(meta)),
    ))


def load_fold(path, fingerprint, participant, test_indices, y_true,
              validation_participants=None):
    """Load a fold checkpoint, refusing anything that does not match."""

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        saved_indices = data["test_indices"]
        saved_y = data["y_true"]
        predictions = data["predictions"]

    problems = []
    if meta.get("fingerprint") != fingerprint:
        problems.append("configuration/data/code fingerprint differs")
    if meta.get("participant") != participant:
        problems.append("participant differs")
    if not np.array_equal(saved_indices, test_indices):
        problems.append("test window indices differ")
    if not np.array_equal(saved_y, y_true):
        problems.append("labels differ")
    if len(predictions) != len(test_indices):
        problems.append("prediction count differs")
    if not set(np.unique(predictions)) <= set(LABELS):
        problems.append("predictions contain invalid labels")
    if (validation_participants is not None and
            meta.get("validation_participants")
            != sorted(validation_participants)):
        problems.append("validation participants differ")

    if problems:
        raise CheckpointMismatch(f"{path}: " + "; ".join(problems))

    return predictions, meta


def check_manifest(run_dir, fingerprints, smoke):
    """
    Create or verify <run_dir>/manifest.json. A model's fingerprint may
    be added once; an existing one must match exactly. A smoke-test run
    directory can never be reused for a full run and vice versa.
    """

    path = Path(run_dir) / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get("smoke_test") != smoke:
            raise CheckpointMismatch(
                f"{path}: run directory was created with smoke_test="
                f"{manifest.get('smoke_test')}; use another --run-dir")
        for model, fingerprint in fingerprints.items():
            known = manifest["fingerprints"].get(model)
            if known is not None and known != fingerprint:
                raise CheckpointMismatch(
                    f"{path}: {model} fingerprint differs from the existing "
                    "run (data, configuration or code changed); use a new "
                    "--run-dir")
        if all(model in manifest["fingerprints"] for model in fingerprints):
            return manifest
    else:
        manifest = {"created": datetime.now().isoformat(timespec="seconds"),
                    "smoke_test": smoke, "fingerprints": {}}

    manifest["fingerprints"].update(fingerprints)
    atomic_write_json(path, manifest)
    return manifest


def fit_predict_fold(model, X, y, groups, test, val_participants, device,
                     max_epochs):
    """Train on one fold exactly as the original LOPO loops did."""

    info = {}
    if model == "bilstm":
        val = np.isin(groups, list(val_participants)) & ~test
        train = ~test & ~val
        history = []
        net, mean, std, epoch = train_bilstm(
            X[train], y[train], X[val], y[val], device, max_epochs,
            history=history)
        predictions = predict_bilstm(net, X[test], mean, std, device)
        info.update(best_epoch=int(epoch), epochs=history,
                    validation_participants=sorted(val_participants))
    else:
        features = window_summary(X)
        rf = make_rf().fit(features[~test], y[~test])
        predictions = rf.predict(features[test])
    return predictions, info


def run_lopo(model, X, y, groups, device, run_dir, fingerprint, max_epochs,
             only_folds=None):
    """
    Leave-one-participant-out with per-fold checkpoints.

    The inner-validation generator is advanced for EVERY fold in sorted
    order, including folds that are skipped or resumed, so each fold's
    validation participants are identical to an uninterrupted run.

    Returns (predictions, fold_meta, done_mask); predictions are 0 for
    windows whose fold has not been completed.
    """

    predictions = np.zeros_like(y)
    done = np.zeros(len(y), dtype=bool)
    fold_meta = {}
    rng = np.random.default_rng(SEED)

    for participant in sorted(set(groups)):
        test = groups == participant
        test_indices = np.flatnonzero(test)

        val_participants = None
        if model == "bilstm":
            val_participants = inner_validation_participants(
                set(groups[~test]), rng)

        if only_folds and participant not in only_folds:
            continue

        path = fold_path(run_dir, model, participant)

        if path.exists():
            fold_pred, meta = load_fold(path, fingerprint, participant,
                                        test_indices, y[test],
                                        val_participants)
            _log(f"[{model}] fold {participant}: resumed from checkpoint "
                 f"(acc {np.mean(fold_pred == y[test]):.3f})")
        else:
            started = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            fold_pred, info = fit_predict_fold(
                model, X, y, groups, test, val_participants, device,
                max_epochs)

            meta = dict(info, seconds=round(time.time() - started, 1),
                        device=(torch.cuda.get_device_name(device)
                                if device.type == "cuda" else "cpu"),
                        peak_rss_mb=(round(resource.getrusage(
                            resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
                            if resource is not None else None),
                        finished=datetime.now().isoformat(timespec="seconds"))
            if device.type == "cuda" and model == "bilstm":
                meta["peak_vram_allocated_mb"] = round(
                    torch.cuda.max_memory_allocated(device) / 2**20, 1)
                meta["peak_vram_reserved_mb"] = round(
                    torch.cuda.max_memory_reserved(device) / 2**20, 1)

            save_fold(path, fingerprint, participant, test_indices, y[test],
                      fold_pred, meta)

            epochs = meta.get("epochs") or []
            epoch_text = ""
            if epochs:
                epoch_text = (
                    f", best epoch {meta['best_epoch']} of {len(epochs)} run, "
                    f"{np.mean([e['seconds'] for e in epochs]):.1f} s/epoch")
            _log(f"[{model}] fold {participant}: "
                 f"acc {np.mean(fold_pred == y[test]):.3f}{epoch_text}, "
                 f"{meta['seconds']} s, saved {path.name}")

        predictions[test] = fold_pred
        done[test] = True
        fold_meta[participant] = meta

    return predictions, fold_meta, done


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

MODEL_NAMES = {
    "rf_original": "RF, original Kinect features",
    "rf_shared": "RF, shared_v1 features",
    "bilstm": "BiLSTM, shared_v1 features",
}


def main():
    parser = argparse.ArgumentParser(
        description="Step A leave-one-participant-out comparison with "
                    "per-fold checkpoints (rerun the same command to resume).")
    parser.add_argument("--models", default=",".join(MODELS),
                        help="comma-separated subset of " + ", ".join(MODELS))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--only-folds", default="",
                        help="SMOKE TEST ONLY: comma-separated participants; "
                             "requires a --run-dir other than the default "
                             "and never writes metrics")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--cpu-threads", type=int, default=2,
                        help="limit CPU threads (does not change results)")
    parser.add_argument("--final-models", action="store_true",
                        help="after all folds of all models are complete, "
                             "train the Step B models on all participants")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(models) - set(MODELS)
    if unknown or not models:
        parser.error(f"unknown models: {sorted(unknown)}")

    only_folds = {p.strip() for p in args.only_folds.split(",") if p.strip()}
    smoke = bool(only_folds)
    run_dir = args.run_dir.resolve()
    if smoke and run_dir == DEFAULT_RUN_DIR.resolve():
        parser.error("--only-folds requires a separate scratch --run-dir")
    if smoke and args.final_models:
        parser.error("--final-models cannot be used with --only-folds")

    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()

    _log(f"device: {device}"
         + (f" ({torch.cuda.get_device_name(device)}, "
            f"{torch.cuda.get_device_properties(device).total_memory / 2**30:.1f} GiB)"
            if device.type == "cuda" else ""))
    _log(f"models: {models}; run dir: {run_dir}"
         + ("; SMOKE TEST (no metrics will be written)" if smoke else ""))

    _log("Building shared_v1 windows from raw Toronto data...")
    X_shared, y, groups, _ = build_shared_windows()
    X_orig, y_orig, groups_orig, _ = load_original_windows()
    _log(f"shared_v1: {X_shared.shape}, original: {X_orig.shape}")

    # Same trials and labelling rule, so label counts must agree.
    assert np.array_equal(np.bincount(y), np.bincount(y_orig)), (
        "window label counts differ between feature sets")

    data = {
        "rf_original": (X_orig, y_orig, groups_orig),
        "rf_shared": (X_shared, y, groups),
        "bilstm": (X_shared, y, groups),
    }

    unknown_folds = only_folds - set(groups.tolist())
    if unknown_folds:
        parser.error(f"unknown participants: {sorted(unknown_folds)}")

    fingerprints = {
        model: model_fingerprint(model, *data[model], args.max_epochs)
        for model in models
    }
    check_manifest(run_dir, fingerprints, smoke)

    complete = {}
    for model in models:
        X, y_m, groups_m = data[model]
        _log(f"{MODEL_NAMES[model]} (LOPO)...")
        predictions, fold_meta, done = run_lopo(
            model, X, y_m, groups_m, device, run_dir, fingerprints[model],
            args.max_epochs, only_folds or None)

        n_done = len(fold_meta)
        n_total = len(set(groups_m.tolist()))
        if smoke or not done.all():
            _log(f"[{model}] {n_done}/{n_total} folds available; "
                 "no metrics written for this model")
            continue

        result = summarize(MODEL_NAMES[model], y_m, predictions, groups_m)
        result["folds"] = fold_meta
        result["fingerprint"] = fingerprints[model]
        atomic_write_json(run_dir / f"metrics_{model}.json", result)
        complete[model] = result
        _log(f"[{model}] all {n_total} folds complete: macro-F1 "
             f"{result['pooled_macro_f1']:.4f}, accuracy "
             f"{result['pooled_accuracy']:.4f} -> metrics_{model}.json")

    if smoke:
        _log(f"SMOKE TEST finished in {time.time() - started:.0f} s. "
             "Checkpoints in the scratch run directory are not results.")
        return

    # Combined report only when every model has all folds.
    finished = {
        model: json.loads((run_dir / f"metrics_{model}.json").read_text())
        for model in MODELS
        if (run_dir / f"metrics_{model}.json").exists()
    }
    if len(finished) == len(MODELS):
        results = {
            "majority_class": summarize(
                "Majority class", y, np.full_like(y, np.bincount(y).argmax()),
                groups),
        }
        rng = np.random.default_rng(SEED)
        freq = np.bincount(y, minlength=5)[1:] / len(y)
        results["label_frequency_random"] = summarize(
            "Random by label frequency", y,
            rng.choice(LABELS, size=len(y), p=freq), groups)
        results.update(finished)
        atomic_write_json(run_dir / "metrics.json", {
            "meta": {
                "feature_set": FEATURE_SET,
                "feature_names": list(FEATURE_NAMES),
                "window": WINDOW,
                "windows": int(len(y)),
                "participants": sorted(set(groups.tolist())),
                "evaluation": "leave-one-participant-out; pooled predictions",
                "fingerprints": {m: finished[m]["fingerprint"]
                                 for m in MODELS},
                "caveat": (
                    "Kinect-to-Kinect evaluation on the Toronto dataset only. "
                    "Not evidence of MediaPipe webcam accuracy or clinical "
                    "validity."),
            },
            "results": results,
        })
        _log(f"All models complete: {run_dir / 'metrics.json'}")
    else:
        missing = [m for m in MODELS if m not in finished]
        _log(f"Combined metrics.json NOT written; incomplete models: "
             f"{missing}")

    if args.final_models:
        if len(finished) != len(MODELS):
            _log("--final-models skipped: evaluation is not complete")
            return
        final_dir = OUTPUT_DIR / "final_models_not_deployed"
        final_dir.mkdir(exist_ok=True)
        rf_path = final_dir / "rf_shared_v1.joblib"
        nn_path = final_dir / "bilstm_shared_v1.pt"
        if rf_path.exists() or nn_path.exists():
            raise SystemExit(f"{final_dir} already contains models; "
                             "not overwriting")

        # Final models for Step B external validation only.
        # NOT deployed in the application.
        rf = make_rf().fit(window_summary(X_shared), y)
        joblib.dump({"model": rf, "feature_set": FEATURE_SET,
                     "feature_names": FEATURE_NAMES, "window": WINDOW,
                     "summary": "mean,std,min,max,last"}, rf_path)

        epochs = int(np.median([f["best_epoch"] for f in
                                finished["bilstm"]["folds"].values()]))
        net, mean, std, _ = train_bilstm(
            X_shared, y, None, None, device, epochs, fixed_epochs=epochs)
        torch.save({"model_state_dict": net.state_dict(),
                    "input_size": X_shared.shape[-1], "mean": mean,
                    "std": std, "epochs": epochs,
                    "feature_set": FEATURE_SET, "window": WINDOW}, nn_path)
        _log(f"Final models saved to {final_dir}")

    _log(f"Finished in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
