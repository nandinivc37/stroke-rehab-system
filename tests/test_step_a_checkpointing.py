"""
Step A: fold checkpoint integrity, resume and validation-split replay.

Synthetic arrays only; no Toronto data and no real-length training.
"""

import json

import numpy as np
import pytest
import torch

import src.compensation_detection.step_a_compare as sa

PARTICIPANTS = ["H01", "H02", "H03", "H04", "H05", "P01", "P02", "P03"]


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    groups = np.repeat(PARTICIPANTS, 15)
    X = rng.normal(size=(len(groups), sa.WINDOW, 6)).astype(np.float32)
    y = rng.integers(1, 5, size=len(groups))
    return X, y, groups


class FakeFit:
    """Records calls; returns deterministic predictions per fold."""

    def __init__(self, fail_after=None):
        self.calls = []
        self.fail_after = fail_after

    def __call__(self, model, X, y, groups, test, val_participants, device,
                 max_epochs):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise KeyboardInterrupt("simulated power cut")
        participant = groups[test][0]
        self.calls.append((participant, sorted(val_participants or [])))
        predictions = (np.arange(test.sum()) % 4) + 1
        info = {}
        if model == "bilstm":
            info = {"best_epoch": 1,
                    "validation_participants": sorted(val_participants)}
        return predictions, info


CPU = torch.device("cpu")


def run(data, tmp_path, fake, monkeypatch, only=None, fp="fp", model="bilstm"):
    monkeypatch.setattr(sa, "fit_predict_fold", fake)
    X, y, groups = data
    return sa.run_lopo(model, X, y, groups, CPU, tmp_path, fp, 30, only)


# ---------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------

def test_failed_write_leaves_no_file_or_temp(tmp_path):
    target = tmp_path / "x.npz"

    def broken(file):
        file.write(b"partial")
        raise RuntimeError("power cut while writing")

    with pytest.raises(RuntimeError):
        sa.atomic_write_bytes(target, broken)
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_replaces_whole_file(tmp_path):
    target = tmp_path / "m.json"
    sa.atomic_write_json(target, {"a": 1})
    sa.atomic_write_json(target, {"b": 2})
    assert json.loads(target.read_text()) == {"b": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]


# ---------------------------------------------------------
# Fold checkpoint validation
# ---------------------------------------------------------

def saved_fold(tmp_path):
    path = tmp_path / "fold_H01.npz"
    idx = np.arange(10, 20)
    y_true = np.array([1, 2, 3, 4, 1, 1, 2, 2, 3, 4])
    sa.save_fold(path, "fp", "H01", idx, y_true, y_true.copy(),
                 {"best_epoch": 3, "validation_participants": ["H02", "P01"]})
    return path, idx, y_true


def test_fold_round_trip(tmp_path):
    path, idx, y_true = saved_fold(tmp_path)
    predictions, meta = sa.load_fold(path, "fp", "H01", idx, y_true,
                                     {"P01", "H02"})
    np.testing.assert_array_equal(predictions, y_true)
    assert meta["best_epoch"] == 3


@pytest.mark.parametrize("change, message", [
    (dict(fingerprint="other"), "fingerprint"),
    (dict(participant="H02"), "participant"),
    (dict(shift=1), "indices"),
    (dict(flip=True), "labels"),
    (dict(val={"H03", "P01"}), "validation participants"),
])
def test_mismatched_checkpoint_is_refused(tmp_path, change, message):
    path, idx, y_true = saved_fold(tmp_path)
    with pytest.raises(sa.CheckpointMismatch, match=message):
        sa.load_fold(
            path,
            change.get("fingerprint", "fp"),
            change.get("participant", "H01"),
            idx + change.get("shift", 0),
            (5 - y_true) if change.get("flip") else y_true,
            change.get("val", {"H02", "P01"}),
        )


def test_corrupt_checkpoint_raises(tmp_path):
    path, idx, y_true = saved_fold(tmp_path)
    path.write_bytes(b"not an npz")
    with pytest.raises(Exception):
        sa.load_fold(path, "fp", "H01", idx, y_true)


# ---------------------------------------------------------
# Manifest
# ---------------------------------------------------------

def test_manifest_accepts_same_and_new_models(tmp_path):
    sa.check_manifest(tmp_path, {"bilstm": "a"}, smoke=False)
    sa.check_manifest(tmp_path, {"bilstm": "a"}, smoke=False)
    manifest = sa.check_manifest(tmp_path, {"rf_shared": "b"}, smoke=False)
    assert manifest["fingerprints"] == {"bilstm": "a", "rf_shared": "b"}


def test_manifest_refuses_changed_fingerprint(tmp_path):
    sa.check_manifest(tmp_path, {"bilstm": "a"}, smoke=False)
    with pytest.raises(sa.CheckpointMismatch, match="fingerprint"):
        sa.check_manifest(tmp_path, {"bilstm": "changed"}, smoke=False)


def test_manifest_keeps_smoke_and_full_runs_apart(tmp_path):
    sa.check_manifest(tmp_path, {"bilstm": "a"}, smoke=True)
    with pytest.raises(sa.CheckpointMismatch, match="smoke"):
        sa.check_manifest(tmp_path, {"bilstm": "a"}, smoke=False)


# ---------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------

def test_fingerprint_detects_data_label_split_and_config_changes(data):
    X, y, groups = data
    base = sa.model_fingerprint("bilstm", X, y, groups, 30)
    assert base == sa.model_fingerprint("bilstm", X.copy(), y.copy(),
                                        groups.copy(), 30)

    X2 = X.copy()
    X2[0, 0, 0] += 1e-3
    y2 = y.copy()
    y2[0] = 1 + (y2[0] % 4)
    g2 = groups.copy()
    g2[0] = "H02"

    for changed in (
        sa.model_fingerprint("bilstm", X2, y, groups, 30),
        sa.model_fingerprint("bilstm", X, y2, groups, 30),
        sa.model_fingerprint("bilstm", X, y, g2, 30),
        sa.model_fingerprint("bilstm", X, y, groups, 29),
        sa.model_fingerprint("rf_shared", X, y, groups, 30),
    ):
        assert changed != base


# ---------------------------------------------------------
# Resume and validation-generator replay
# ---------------------------------------------------------

def test_interrupted_run_resumes_without_retraining(data, tmp_path,
                                                    monkeypatch):
    reference = FakeFit()
    full, _, done = run(data, tmp_path / "reference", reference, monkeypatch)
    assert done.all()

    interrupted = FakeFit(fail_after=3)
    with pytest.raises(KeyboardInterrupt):
        run(data, tmp_path / "run", interrupted, monkeypatch)
    assert len(list((tmp_path / "run" / "bilstm").glob("fold_*.npz"))) == 3

    resumed = FakeFit()
    predictions, meta, done = run(data, tmp_path / "run", resumed,
                                  monkeypatch)

    assert done.all()
    # Only the 5 unfinished folds were trained.
    assert [c[0] for c in resumed.calls] == PARTICIPANTS[3:]
    # Identical outcome to an uninterrupted run, including the
    # validation participants of every fold.
    np.testing.assert_array_equal(predictions, full)
    assert resumed.calls == reference.calls[3:]


def test_single_fold_gets_same_validation_split_as_full_run(
    data, tmp_path, monkeypatch
):
    full = FakeFit()
    run(data, tmp_path / "full", full, monkeypatch)

    single = FakeFit()
    _, meta, done = run(data, tmp_path / "single", single, monkeypatch,
                        only={"H04"})

    assert [c[0] for c in single.calls] == ["H04"]
    assert single.calls[0] == dict((c[0], c) for c in full.calls)["H04"]
    assert list(meta) == ["H04"]
    assert done.sum() == (data[2] == "H04").sum()


def test_resume_refuses_checkpoint_from_other_configuration(
    data, tmp_path, monkeypatch
):
    run(data, tmp_path, FakeFit(), monkeypatch, only={"H01"}, fp="old")
    with pytest.raises(sa.CheckpointMismatch):
        run(data, tmp_path, FakeFit(), monkeypatch, only={"H01"}, fp="new")


def test_rf_folds_checkpoint_without_validation_split(data, tmp_path,
                                                      monkeypatch):
    fake = FakeFit()
    run(data, tmp_path, fake, monkeypatch, model="rf_shared")
    resumed = FakeFit()
    _, _, done = run(data, tmp_path, resumed, monkeypatch, model="rf_shared")
    assert done.all() and resumed.calls == []


def test_real_bilstm_fold_trains_checkpoints_and_resumes(data, tmp_path):
    """One genuine (tiny, CPU, 1-epoch) fold through the real code path."""
    X, y, groups = data
    predictions, meta, _ = sa.run_lopo("bilstm", X, y, groups, CPU,
                                       tmp_path, "fp", 1, {"P02"})
    fold = meta["P02"]
    assert fold["best_epoch"] == 1 and len(fold["epochs"]) == 1
    assert "seconds" in fold["epochs"][0]
    again, meta2, _ = sa.run_lopo("bilstm", X, y, groups, CPU, tmp_path,
                                  "fp", 1, {"P02"})
    np.testing.assert_array_equal(again, predictions)
    assert meta2["P02"]["finished"] == fold["finished"]


# ---------------------------------------------------------
# Command-line guards (checked before any data is loaded)
# ---------------------------------------------------------

@pytest.mark.parametrize("argv", [
    ["--models", "bilstm", "--only-folds", "H01"],          # default dir
    ["--models", "bilstm", "--only-folds", "H01", "--run-dir", "/tmp/x",
     "--final-models"],
    ["--models", "lstm"],
])
def test_cli_rejects_unsafe_or_invalid_options(argv, monkeypatch):
    monkeypatch.setattr("sys.argv", ["step_a_compare"] + argv)
    with pytest.raises(SystemExit):
        sa.main()
