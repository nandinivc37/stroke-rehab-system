"""
Step A derived analysis: reads fold checkpoints, never trains.

Synthetic checkpoints only; no Toronto data.
"""

import numpy as np
import pytest

import src.compensation_detection.step_a_analysis as an
import src.compensation_detection.step_a_compare as sa

PARTICIPANTS = ["H01", "H02", "P01"]


def write_run(run_dir, models=an.MODELS):
    rng = np.random.default_rng(0)
    start = 0
    for participant in PARTICIPANTS:
        y = rng.integers(1, 5, size=40)
        idx = np.arange(start, start + len(y))
        start += len(y)
        for model in models:
            predictions = y.copy() if model == "rf_shared" else np.ones_like(y)
            path = sa.fold_path(run_dir, model, participant)
            path.parent.mkdir(parents=True, exist_ok=True)
            sa.save_fold(path, "fp", participant, idx, y, predictions, {})


def test_loads_every_participant_and_model(tmp_path):
    write_run(tmp_path)
    folds = an.load_predictions(tmp_path)
    assert sorted(folds) == PARTICIPANTS
    y, predictions = an.pooled(folds, "rf_shared", PARTICIPANTS)
    assert an.macro_f1(y, predictions) == 1.0


def test_refuses_incomplete_run(tmp_path):
    write_run(tmp_path, models=["rf_original", "bilstm"])
    with pytest.raises(SystemExit, match="incomplete"):
        an.load_predictions(tmp_path)


def test_report_is_labelled_as_derived(tmp_path, monkeypatch, capsys):
    write_run(tmp_path)
    monkeypatch.setattr(an, "BOOTSTRAP_RESAMPLES", 20)
    monkeypatch.setattr("sys.argv", ["step_a_analysis", "--run-dir",
                                     str(tmp_path)])
    an.main()
    out = capsys.readouterr().out
    assert out.startswith("DERIVED, POST-HOC ANALYSIS")
    assert "rf_shared - rf_original" in out
    assert "rf_shared > rf_original in 3/3 participants" in out
