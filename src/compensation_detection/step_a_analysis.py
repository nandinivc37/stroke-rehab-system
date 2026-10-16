"""
Step A: derived analysis of the saved fold predictions. Never trains.

Reads the per-fold prediction checkpoints of a completed Step A run and
prints analyses that are NOT part of step_a_compare.py's metrics files:

  * per-participant accuracy, false-alarm rate (true "No Compensation"
    windows predicted as any compensation) and any-compensation recall,
  * binary "any compensation vs none" precision/recall/F1, pooled and
    for healthy participants and patients,
  * participant-level bootstrap 95% CIs of pooled macro-F1 and of the
    pairwise differences between models,
  * per-participant macro-F1 win counts.

These analyses were defined after the Step A metrics had been seen
(post hoc). They describe the existing predictions; they were not used
to change any model, hyperparameter or the evaluation protocol.

Usage (from the project root):
    python -m src.compensation_detection.step_a_analysis \
        [--run-dir outputs/compensation_step_a/lopo_run_colab]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.paths import OUTPUTS_DIR

MODELS = ["rf_original", "rf_shared", "bilstm"]
LABELS = [1, 2, 3, 4]
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
DEFAULT_RUN_DIR = OUTPUTS_DIR / "compensation_step_a" / "lopo_run_colab"


def load_predictions(run_dir):
    """{participant: {model: (y_true, predictions)}} for complete runs."""
    folds = {}
    for model in MODELS:
        for path in sorted((Path(run_dir) / model).glob("fold_*.npz")):
            with np.load(path, allow_pickle=False) as data:
                participant = json.loads(str(data["meta"]))["participant"]
                folds.setdefault(participant, {})[model] = (
                    data["y_true"].copy(), data["predictions"].copy())
    incomplete = [p for p, m in folds.items() if len(m) != len(MODELS)]
    if not folds or incomplete:
        raise SystemExit(f"incomplete run in {run_dir}: {incomplete}")
    for per_model in folds.values():
        labels = [per_model[m][0] for m in MODELS]
        if not all(np.array_equal(labels[0], y) for y in labels[1:]):
            raise SystemExit("models were evaluated on different labels")
    return folds


def pooled(folds, model, participants):
    y = np.concatenate([folds[p][model][0] for p in participants])
    predictions = np.concatenate([folds[p][model][1] for p in participants])
    return y, predictions


def macro_f1(y, predictions):
    return f1_score(y, predictions, labels=LABELS, average="macro",
                    zero_division=0)


def per_participant(folds):
    print("== Per participant (derived)")
    print("label counts = windows of No Compensation / Shoulder Elevation "
          "/ Trunk Rotation / Lean Forward")
    print("cells = accuracy | false-alarm rate on true-None windows | "
          "any-compensation recall (nan: no such windows)")
    print(f"{'part':5} {'counts':>24}  "
          + "  ".join(f"{m:>22}" for m in MODELS))
    for participant in sorted(folds):
        y = folds[participant][MODELS[0]][0]
        counts = [int((y == k).sum()) for k in LABELS]
        cells = []
        for model in MODELS:
            y_true, predictions = folds[participant][model]
            accuracy = np.mean(y_true == predictions)
            none = y_true == 1
            false_alarm = (np.mean(predictions[none] != 1) if none.any()
                           else float("nan"))
            compensation = ~none
            recall = (np.mean(predictions[compensation] != 1)
                      if compensation.any() else float("nan"))
            cells.append(f"{accuracy:.3f}|{false_alarm:.3f}|{recall:.3f}")
        print(f"{participant:5} {str(counts):>24}  "
              + "  ".join(f"{c:>22}" for c in cells))


def binary_detection(folds):
    print("\n== Binary: any compensation vs none, pooled (derived)")
    participants = sorted(folds)
    for group, selected in [
        ("all", participants),
        ("healthy", [p for p in participants if p.startswith("H")]),
        ("patients", [p for p in participants if p.startswith("P")]),
    ]:
        for model in MODELS:
            y, predictions = pooled(folds, model, selected)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y != 1, predictions != 1, average="binary", zero_division=0)
            false_alarm = np.mean(predictions[y == 1] != 1)
            print(f"  {group:8} {model:12} precision {precision:.3f} "
                  f"recall {recall:.3f} F1 {f1:.3f} "
                  f"false-alarm {false_alarm:.3f}  (compensation windows "
                  f"{int((y != 1).sum())}, none {int((y == 1).sum())})")


def bootstrap(folds):
    print(f"\n== Participant-level bootstrap of pooled macro-F1 "
          f"({BOOTSTRAP_RESAMPLES} resamples of the 19 participants with "
          f"replacement, seed {BOOTSTRAP_SEED}) (derived)")
    participants = sorted(folds)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    scores = {m: np.empty(BOOTSTRAP_RESAMPLES) for m in MODELS}
    for b in range(BOOTSTRAP_RESAMPLES):
        selected = rng.choice(participants, size=len(participants),
                              replace=True)
        for model in MODELS:
            scores[model][b] = macro_f1(*pooled(folds, model, selected))
    point = {m: macro_f1(*pooled(folds, m, participants)) for m in MODELS}
    for model in MODELS:
        low, high = np.percentile(scores[model], [2.5, 97.5])
        print(f"  {model:12} point {point[model]:.4f}  "
              f"95% CI [{low:.3f}, {high:.3f}]")
    for a, b in [("rf_shared", "rf_original"), ("bilstm", "rf_shared"),
                 ("bilstm", "rf_original")]:
        difference = scores[a] - scores[b]
        low, high = np.percentile(difference, [2.5, 97.5])
        print(f"  {a} - {b}: {point[a] - point[b]:+.4f}  "
              f"95% CI [{low:+.3f}, {high:+.3f}]  "
              f"share of resamples > 0: {np.mean(difference > 0):.3f}")


def win_counts(folds):
    print("\n== Per-participant macro-F1 wins, over the classes present "
          "in each participant's data (derived)")
    for a, b in [("bilstm", "rf_shared"), ("bilstm", "rf_original"),
                 ("rf_shared", "rf_original")]:
        wins = 0
        for participant in sorted(folds):
            y_true = folds[participant][a][0]
            present = sorted(set(y_true))
            score = {m: f1_score(y_true, folds[participant][m][1],
                                 labels=present, average="macro",
                                 zero_division=0) for m in (a, b)}
            wins += score[a] > score[b]
        print(f"  {a} > {b} in {wins}/{len(folds)} participants")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()

    folds = load_predictions(args.run_dir)
    print("DERIVED, POST-HOC ANALYSIS of saved Step A fold predictions.")
    print("Verified metrics are in metrics.json / metrics_<model>.json; "
          "nothing here was used to change models or protocol.")
    print(f"run dir: {args.run_dir}; participants: {len(folds)}\n")
    per_participant(folds)
    binary_detection(folds)
    bootstrap(folds)
    win_counts(folds)


if __name__ == "__main__":
    main()
