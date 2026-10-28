"""
Evaluate recorded webcam sessions against independent ground truth.

Inputs (see docs/validation_protocol.md for how to collect them):

  --db           the application database (a COPY is recommended).
  --labels       CSV, one row per validation session:
                   session_id, manual_reps, movement_label
                 manual_reps: repetitions counted by an observer (or
                   from a video review) that met the instruction.
                 movement_label: correct | incomplete | jerky | off_plane
                   (the movement the participant was asked to perform).
  --goniometer   optional CSV of reference angle readings:
                   session_id, t_s, reference_value
                 t_s: seconds from the start of the session (the same
                   clock as session_frames.t_s).
  --criteria     acceptance criteria fixed BEFORE analysis
                   (default docs/validation_criteria.json).

Prints and optionally writes a JSON report. Every criterion is reported
as PASS, FAIL or INSUFFICIENT_DATA; nothing is estimated when data are
missing.

Usage:
    python -m src.validation.evaluate_sessions --db copy.db \
        --labels labels.csv [--goniometer gonio.csv] [--out report.json]
"""

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

DEFAULT_CRITERIA = (
    Path(__file__).resolve().parents[2] / "docs" / "validation_criteria.json"
)

PASS = "PASS"
FAIL = "FAIL"
INSUFFICIENT = "INSUFFICIENT_DATA"


def load_sessions(db_path, session_ids):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        sessions = {}
        for sid in session_ids:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (sid,)
            ).fetchone()
            if row is not None:
                sessions[sid] = dict(row)
        frames = defaultdict(list)
        for sid in sessions:
            for row in conn.execute(
                "SELECT t_s, status, angle FROM session_frames "
                "WHERE session_id = ? ORDER BY frame_index", (sid,)
            ):
                frames[sid].append(dict(row))
        return sessions, frames
    finally:
        conn.close()


def read_csv(path):
    with open(path, newline="", encoding="utf8") as file:
        return list(csv.DictReader(file))


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def rep_agreement(sessions, labels):
    """Per exercise: exact-match rate and mean absolute error."""

    by_exercise = defaultdict(list)
    for label in labels:
        session = sessions.get(int(label["session_id"]))
        if session is None or label.get("manual_reps") in (None, ""):
            continue
        by_exercise[session["exercise_id"]].append(
            (session["reps"] or 0, int(label["manual_reps"]))
        )
    return {
        exercise: {
            "sessions": len(pairs),
            "exact_match_rate": sum(s == m for s, m in pairs) / len(pairs),
            "mean_abs_error": mean(abs(s - m) for s, m in pairs),
            "pairs": pairs,
        }
        for exercise, pairs in by_exercise.items()
    }


def angle_agreement(sessions, frames, readings, window_s):
    """
    Each reference reading is matched with the median measured value of
    the session's measured frames within +/- window_s. Readings with no
    measured frame in the window are counted as unmatched (not scored).
    """

    by_exercise = defaultdict(lambda: {"errors": [], "unmatched": 0})
    for reading in readings:
        sid = int(reading["session_id"])
        session = sessions.get(sid)
        if session is None:
            continue
        t = float(reading["t_s"])
        reference = float(reading["reference_value"])
        near = [
            f["angle"] for f in frames.get(sid, [])
            if f["status"] == "ok" and f["angle"] is not None
            and f["t_s"] is not None and abs(f["t_s"] - t) <= window_s
        ]
        entry = by_exercise[session["exercise_id"]]
        if near:
            entry["errors"].append(median(near) - reference)
        else:
            entry["unmatched"] += 1

    return {
        exercise: {
            "readings": len(e["errors"]),
            "unmatched": e["unmatched"],
            "mean_abs_error": (
                mean(abs(x) for x in e["errors"]) if e["errors"] else None
            ),
            "bias": mean(e["errors"]) if e["errors"] else None,
            "max_abs_error": (
                max(abs(x) for x in e["errors"]) if e["errors"] else None
            ),
        }
        for exercise, e in by_exercise.items()
    }


def plane_accuracy(sessions, labels):
    """
    For sessions labelled correct or off_plane, the fraction of
    repetitions the system placed in the expected plane category:
    correct -> counted reps; off_plane -> off-plane reps.
    """

    right = total = 0
    for label in labels:
        session = sessions.get(int(label["session_id"]))
        kind = label.get("movement_label")
        if session is None or kind not in ("correct", "off_plane"):
            continue
        reps = session["reps"] or 0
        off = session["off_plane_reps"] or 0
        total += reps + off
        right += reps if kind == "correct" else off
    return {"reps": total, "accuracy": right / total if total else None}


def auc(positives, negatives):
    """Probability that a positive scores higher than a negative
    (ties count half): the Mann-Whitney AUC."""

    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p in positives for n in negatives
    )
    return wins / (len(positives) * len(negatives))


def score_discrimination(sessions, labels):
    """Movement score and DTW similarity by label; AUC correct vs rest.
    Sessions without a score are listed, not scored as 0."""

    by_label = defaultdict(lambda: {"score": [], "dtw": [], "no_score": 0})
    for label in labels:
        session = sessions.get(int(label["session_id"]))
        kind = label.get("movement_label")
        if session is None or not kind:
            continue
        entry = by_label[kind]
        if session.get("trajectory_similarity") is None:
            entry["no_score"] += 1
            continue
        entry["score"].append(session["trajectory_similarity"])
        if session.get("dtw_similarity") is not None:
            entry["dtw"].append(session["dtw_similarity"])

    other = [
        x for kind, e in by_label.items() if kind != "correct"
        for x in e["score"]
    ]
    other_dtw = [
        x for kind, e in by_label.items() if kind != "correct"
        for x in e["dtw"]
    ]
    correct = by_label.get("correct", {"score": [], "dtw": []})
    return {
        "by_label": {
            kind: {
                "sessions_scored": len(e["score"]),
                "sessions_without_score": e["no_score"],
                "mean_movement_score": mean(e["score"]) if e["score"] else None,
                "mean_dtw_similarity": mean(e["dtw"]) if e["dtw"] else None,
            }
            for kind, e in by_label.items()
        },
        "auc_movement_score_correct_vs_other": auc(correct["score"], other),
        "auc_dtw_correct_vs_other": auc(correct["dtw"], other_dtw),
    }


# ---------------------------------------------------------
# Criteria
# ---------------------------------------------------------

def judge(report, criteria):
    verdicts = {}

    fps = [s["mean_fps"] for s in report["sessions"] if s["mean_fps"]]
    verdicts["fps"] = (
        INSUFFICIENT if not fps
        else PASS if min(fps) >= criteria["fps"]["min_mean_fps"] else FAIL
    )

    rc = criteria["rep_count"]
    for exercise, r in report["rep_count"].items():
        verdicts[f"rep_count:{exercise}"] = (
            INSUFFICIENT if r["sessions"] < rc["min_sessions_per_exercise"]
            else PASS if (
                r["exact_match_rate"] >= rc["min_exact_match_rate"]
                and r["mean_abs_error"] <= rc["max_mean_abs_error_reps"]
            ) else FAIL
        )

    ac = criteria["angle"]
    for exercise, a in report["angle"].items():
        verdicts[f"angle:{exercise}"] = (
            INSUFFICIENT if a["readings"] < ac["min_readings_per_exercise"]
            else PASS if (
                a["mean_abs_error"] <= ac["max_mean_abs_error"]
                and abs(a["bias"]) <= ac["max_abs_bias"]
            ) else FAIL
        )

    pc = criteria["plane"]
    plane = report["plane"]
    verdicts["plane"] = (
        INSUFFICIENT if plane["reps"] < pc["min_reps"]
        else PASS if plane["accuracy"] >= pc["min_accuracy"] else FAIL
    )

    sc = criteria["score_discrimination"]
    disc = report["score_discrimination"]
    counts = [e["sessions_scored"] for e in disc["by_label"].values()]
    value = disc["auc_movement_score_correct_vs_other"]
    verdicts["score_discrimination"] = (
        INSUFFICIENT if (
            value is None or len(counts) < 2
            or min(counts) < sc["min_sessions_per_label"]
        )
        else PASS if value >= sc["min_auc_correct_vs_other"] else FAIL
    )
    return verdicts


def evaluate(db_path, labels, readings=(), criteria=None):
    criteria = criteria or json.loads(DEFAULT_CRITERIA.read_text("utf8"))
    ids = {int(row["session_id"]) for row in labels} | {
        int(row["session_id"]) for row in readings
    }
    sessions, frames = load_sessions(db_path, sorted(ids))

    report = {
        "criteria_status": criteria.get("status"),
        "sessions_requested": len(ids),
        "sessions_found": len(sessions),
        "sessions": [
            {
                "session_id": sid,
                "exercise_id": s["exercise_id"],
                "mean_fps": s["mean_fps"],
                "measured_fraction": (
                    s["frames_measured"] / s["frames_total"]
                    if s["frames_total"] else None
                ),
                "frames_logged": len(frames.get(sid, [])),
            }
            for sid, s in sessions.items()
        ],
        "rep_count": rep_agreement(sessions, labels),
        "angle": angle_agreement(
            sessions, frames, readings, criteria["angle"]["match_window_s"]
        ),
        "plane": plane_accuracy(sessions, labels),
        "score_discrimination": score_discrimination(sessions, labels),
    }
    report["verdicts"] = judge(report, criteria)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--goniometer", type=Path)
    parser.add_argument("--criteria", type=Path, default=DEFAULT_CRITERIA)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = evaluate(
        args.db,
        read_csv(args.labels),
        read_csv(args.goniometer) if args.goniometer else [],
        json.loads(args.criteria.read_text("utf8")),
    )
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf8")


if __name__ == "__main__":
    main()
