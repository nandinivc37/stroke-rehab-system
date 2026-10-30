# Validation protocol: measurement, repetition counting, movement score

Status: **PREPARED, NOT RUN.** No validation recordings exist yet. This
protocol produces the evidence that Objectives 1-3 need and that
synthetic tests cannot provide.

It is a technical validation with healthy volunteers (e.g. team
members). It says nothing about clinical validity. Follow your
institution's rules on recording people; the application stores
landmark-derived numbers only, never video.

## 0. Before anything is recorded

1. Fix the acceptance criteria in
   [`validation_criteria.json`](validation_criteria.json). The values in
   the file are **proposals**. Change `status` to record who confirmed
   them and when. Do not change them after looking at results.
2. Work on a copy of the database, or note the session ids you create.
3. Record: laptop model, power source (mains/battery) and power plan,
   camera model, `REHAB_POSE_COMPLEXITY` (default 1), date, testers.

## 1. Setup

* Camera upright on a tripod at about shoulder height, 2-2.5 m away.
  Shoulder flexion: frontal view (patient facing the camera, arm
  resting by the side at the start). Forward reach: side or 45° view.
  Shoulder abduction: frontal view. Elbow flexion: frontal or side.
* Upper body and both hips visible; even light; fitted clothing.
* Optional sanity check of the axis convention with your camera:
  `python -m src.validation.axis_convention --camera 0 --frames 150`
  (stand facing the camera, then turn away). All counts should equal
  the number of usable frames.

## 2. Sessions to record

For each exercise and each side, record separate sessions (one session =
Start Exercise … End Session). Note the session id shown on the
Progress page.

| Label | Instruction | Repetitions |
|---|---|---|
| `correct` | the exercise as instructed, above the range target, through the full movement | 5 |
| `incomplete` | the same movement, stopping clearly below the target (about half range) | 5 |
| `jerky` | the full movement in 3-4 stop-start steps up and down | 5 |
| `off_plane` | flexion and abduction only: the *other* raise (abduct when flexion is prescribed, and vice versa) | 5 |

Record at least 3 sessions per label per exercise (the criteria file
requires this for a verdict). Vary speed across sessions (about 1 s,
2 s and 4 s per raise).

An observer counts the repetitions that met the instruction and writes
`labels.csv`:

```
session_id,manual_reps,movement_label
57,5,correct
58,0,incomplete
```

`manual_reps` counts repetitions that should be credited by the
prescription (for `incomplete` and `off_plane` that is normally 0).

## 3. Reference angles (goniometer or inclinometer)

During `correct` sessions, the participant pauses for about 2 s at four
positions: rest, about 45°, about 90° and the peak. A second person
measures the joint angle with a goniometer (or a phone inclinometer on
the upper arm for shoulder elevation) and notes the time from the start
of the session (a stopwatch started with *Start Exercise*):

```
session_id,t_s,reference_value
57,4.0,2
57,8.5,47
```

For forward reach the reference is the wrist's forward distance as a
percentage of arm length (tape measure). The measured value used is the
median of the measured frames within ±0.25 s of `t_s`.

## 4. Analysis

```
python -m src.validation.evaluate_sessions --db data/rehab.db \
    --labels labels.csv --goniometer gonio.csv --out validation_report.json
```

It reports, per exercise:

* frame rate of each session (`fps` verdict: every session ≥ 10 fps);
* repetition count agreement (exact-match rate, mean absolute error);
* angle error against the reference (mean absolute error, bias, max);
* plane classification accuracy (correct reps counted, off-plane reps
  rejected);
* movement score and DTW similarity by label, and the AUC of "correct"
  against all other labels.

Each criterion is PASS, FAIL or INSUFFICIENT_DATA. Report all of them,
including FAILs. A FAIL is a finding, not something to tune away. If
you change thresholds afterwards, record new data to test them.

The per-frame log of every session (`session_frames`: time, status,
angle, plane, landmark visibility) and the pipeline timing
(`sessions.diagnostics_json`) are there to explain failures.

## 5. What this protocol cannot establish

* Accuracy for stroke survivors (different movement patterns, tremor,
  limited range).
* Clinical meaning of the movement score or of any threshold.
* Compensation detection (see `step_b_protocol.md`).
