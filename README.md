# Stroke Rehab — camera-based upper-limb exercise analysis

A PyQt6 desktop application that uses a webcam and MediaPipe Pose to
guide seated upper-limb exercises, count repetitions, compare each
movement with a reference using Dynamic Time Warping (DTW), and propose
conservative next-session difficulty changes for therapist approval.

> **Research prototype.** Nothing in this repository has been clinically
> validated. Measurement accuracy on webcam data has **not been
> validated**, and no live session has yet passed the assessment gate
> (see [Validation status](#validation-status)).

## Quick start

Requires Python 3.12. Tested with 3.12.13 on Windows 11 (`.venv`) and,
earlier, 3.12.3 on Ubuntu (`venv`).

Windows (PowerShell or cmd):

```
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m src.database.migrations   # verified backup + schema upgrade
.venv\Scripts\python -m src.ui.app                # or double-click run.bat
```

Linux/macOS:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements-dev.txt
venv/bin/python -m src.database.migrations
venv/bin/python -m src.ui.app
```

The application also upgrades the database on start-up (additive
migration, after a verified backup in `data/backups/`). All data paths
are resolved from the project root (`src/paths.py`).

Pose model: `REHAB_POSE_COMPLEXITY` (0 lite, 1 full, 2 heavy; default 1,
see [Frame rate](#frame-rate)) selects the MediaPipe model for the app
and the Step B recorder. The value used is stored with every session.

Run the tests (no webcam needed; Qt renders offscreen):

```bash
python -m pytest                     # full suite
python -m pytest tests/test_characterization_shoulder_raise.py  # original rep/DTW behaviour
python -m pyflakes src tests         # static check
```

`pytest.ini` disables third-party plugin autoloading because ROS
packages on some machines' `PYTHONPATH` register incompatible plugins.
Tests never touch `data/rehab.db`: they use temporary databases.

## Using the application

1. **Patients** — add a patient and the affected side/joint.
2. **Assessment** — record a clinical assessment. A rule-based
   recommender (`rules_v2`, see below) suggests an implemented exercise
   and starting targets derived from the baseline ROM and restriction.
   The therapist may choose another supported exercise and edit the
   targets within the catalogue bounds, then approves. Approval stores
   the exercise, side and the approved targets; any change from the
   suggestion is recorded in the rationale.
3. **Rehabilitation** — shows the approved exercise, targets and
   instructions. *Start Exercise* opens the webcam; the reference
   animation (synthetic, labelled as such) plays beside the live view.
   Capture and pose estimation run on a worker thread, so the
   interface stays responsive. *End Session* saves the session (one
   attempt, including a per-frame diagnostic log) and runs the adaptive
   controller. Closing the application during a session saves it as
   `INTERRUPTED_APP_CLOSED`; incomplete sessions are never used for
   adaptation. The controller shows feedback and what to do next
   (repeat, continue, or a change proposed to the therapist).
4. **Progress** — per-exercise history table and trends (missing values
   shown as "—", legacy sessions grouped separately), experimental
   posture measurements for therapist review, adaptation decision log,
   and Approve/Reject for proposed changes. An approved change is used
   by the next session.
5. **Dashboard** — counts from stored data and a shortcut to start a
   session.

## Exercises

Defined declaratively in `src/exercises/catalogue.py` and analysed by
one generic, Qt-free `ExerciseAnalyzer` (`src/exercises/analyzer.py`).
Each config sets the measured angle, required landmarks, rep thresholds,
segmentation, DTW reference, movement-plane check and adaptive
parameters (`target_reps`, `target_range`) with bounds and step sizes.

| Exercise | Measurement (MediaPipe 3D world landmarks) | Status |
|---|---|---|
| Shoulder Flexion | upper-arm elevation relative to trunk; reps only count in the forward plane (±30°) | implemented, offline-tested, **webcam NOT TESTED** |
| Shoulder Abduction | same angle; reps only count in the sideways plane (±30°). This is the original sideways-raise reference movement | implemented, offline-tested, **webcam NOT TESTED** |
| Elbow Flexion | 180° − shoulder–elbow–wrist angle | implemented, offline-tested, **webcam NOT TESTED** |
| Forward Reach | wrist displacement in front of shoulder, % of arm length (depends on MediaPipe depth) | implemented, offline-tested, **webcam NOT TESTED**; highest risk |

Thresholds are engineering defaults, not clinically validated. Reference
movements are synthetic animations generated from a rigid body model;
the DTW reference curve is computed with the same angle function as the
live measurement. When the hips are not visible, the trunk axis falls
back to the camera vertical (assumes an upright camera).

**Coordinate conventions.** Angles use MediaPipe *world* landmarks
(metres, origin between the hips), assumed to have x towards the image
right, y down and z away from the camera. A trunk frame (lateral: right
to left shoulder; up: hips to shoulders, orthogonalised; forward:
lateral × up) makes every angle independent of seating position, image
size and camera distance (scale-free). MediaPipe receives the
**unmirrored** frame; only the display is mirrored. Mirroring the input
would flip the frame's handedness and turn a forward raise into a
backward one (`test_mirrored_input_breaks_the_flexion_plane`).

The axis convention was **checked on real MediaPipe output** (2026-10-28,
`python -m src.validation.axis_convention`, one public video of standing
people, 40 usable frames). In frames of people facing the camera:
* world x followed image x in 34/34;
* y pointed down in 34/34;
* the nose was nearer the camera than the shoulders in 33/34;
* the trunk forward axis pointed at the camera in 34/34, and away from
  it in 6/6 frames of people facing away.

This checks the conventions, not angle accuracy.

### Measurement definitions (metrics version 2)

Sessions are tagged with `metrics_version`; the adaptive policy only
compares sessions of the current version (2).

* **Repetition:** the angle reaches the range target (`target_range`)
  and stays at or above it for ≥ 0.13 s (and ≥ 2 frames), then returns
  to the down threshold for ≥ 0.13 s. The time is measured from frame
  timestamps, so the count does not depend on the frame rate (the
  previous 5-frame rule needed 0.7 s above target at 7 fps). Frames
  between the thresholds interrupt but do not reset the timers. A gap
  of more than 1 s without a measured frame abandons the repetition in
  progress (`interrupted_reps`). A repetition still in progress at the
  end is not counted.
* **Movement plane** (flexion/abduction): a repetition counts only if
  most of its frames above the threshold lie within ±30° of the
  prescribed plane; otherwise it is an off-plane repetition.
* **Glitch filter:** a frame whose value changes faster than
  1000°/s (500 %/s for reach) since the previous reading is rejected
  (`implausible`).
* **Peak ROM** (`peak_rom`): the highest 3-frame median among counted
  repetitions and scored movements that were not off-plane. Single-frame
  spikes, setup movements in another plane and off-plane raises do not
  set it. `max_angle` keeps the raw session maximum for reference.
* **DTW** (`dtw_similarity`, `dtw_mean_error_deg`): each movement
  segment is aligned with the reference curve. The mean error is the
  accumulated |Δangle| divided by the **warping-path length** (the mean
  difference between matched samples). The original rule divided by
  n + m and understated the error by up to about 2×. Similarity =
  100 × (1 − error / 90°), clipped. The 90° scale is a convention, not a
  calibrated threshold. Off-plane segments are reported but not
  compared with the reference.
* **Movement score** (stored in the historical column
  `trajectory_similarity`, shown as "Movement score"):
  0.5 × DTW similarity + 0.5 × 100 × range completion, averaged over
  scored movements. It is **not** a pure similarity, and it has not
  been shown to discriminate movement quality on real data.
* **Smoothness** (velocity peaks): computed on a 10 Hz time base, which
  removes most of the frame-rate inflation of the raw count (17 → 6
  peaks for the same synthetic movement at 30 fps). It is still
  sensitive below 10 fps. It is exploratory and not used anywhere.
* **Per-frame log** (`session_frames`): time, status, angle, plane,
  lowest required-landmark visibility and reference frame for every
  processed frame. Numbers only, no images.

The original 2D image-plane analysis is kept as
`src/exercises/legacy.py` for regression tests and for interpreting
historical sessions.

## Adaptive rehabilitation loop

```
MediaPipe Pose (live_pose_widget.py)
  → ExerciseAnalyzer (src/exercises/analyzer.py): repetitions, ROM,
    movement-plane check, segmentation, DTW similarity to the reference
  → session saved (= one attempt)
  → adaptive controller (src/adaptation/controller.py):
      attempt gate → attempt check → policy (src/adaptation/policy.py)
      → safety guard
  → action + patient feedback shown; decision logged
  → if a change is proposed: PENDING prescription
  → therapist approves on the Progress page → APPROVED prescription
  → next attempt loads the active prescription → reassessed
```

Adaptive controller `attempt_controller_v1` (`decide()`):

* **Inputs:**
  * the `ExerciseConfig`;
  * the active prescription's parameters (the difficulty settings
    `target_reps` and `target_range`, with bounds and steps from the
    catalogue);
  * the attempt's session row: reps, peak ROM, movement score,
    off-plane reps, frames measured, frame errors, fps, session status;
  * the session history for the same exercise and side;
  * any compensation output, which is recorded but never used.
* **Outputs:**
  * `action`;
  * patient `feedback`;
  * a concise `reason`;
  * `next_attempt_parameters` (always the active prescription);
  * `proposed_parameters` (which differ only for SIMPLIFY and
    PROPOSE_PROGRESSION);
  * the attempt check and a compensation note.
* **Actions:**
  * `UNABLE_TO_ASSESS`: the attempt is stale or mismatched (another
    exercise, other settings, or not the latest session), the session is
    incomplete (app closed, camera lost, analysis halted), tracking is
    poor (the same gate as the policy, below), or recorded values are
    invalid. Nothing is scored and settings are unchanged.
  * `REPEAT`: this attempt missed a target: reps, peak ROM, movement
    score ≥ 70, or ≤ 20% off-plane reps. A missing score counts as
    missed.
  * `MAINTAIN`: this attempt met every target, but the policy does not
    (yet) justify a change.
  * `SIMPLIFY`: the policy proposes a decrease.
  * `PROPOSE_PROGRESSION`: the policy proposes an increase **and** this
    attempt met every target.
* **Safety guard:**
  * parameters stay within bounds;
  * at most one parameter changes, by one step;
  * increases only with `PROPOSE_PROGRESSION` after an acceptable
    attempt, and decreases only with `SIMPLIFY`.

  Any violation falls back to no change and is logged.
* **Approval:** every change of settings, easier or harder, requires
  therapist approval. A rejected proposal leaves the previous
  prescription active.

Policy `conservative_v2` (all thresholds configurable in `PolicyConfig`):

* Uses only sessions of the same exercise and side that meet all of:
  * metrics version 2 and the **current parameters**;
  * status `COMPLETE`;
  * arm tracked in ≥ 80% of frames;
  * ≤ 5% and < 30 frame errors;
  * a known frame rate ≥ 10 fps.

  Excluded sessions and the reason are logged.
* Fewer than 2 such sessions → no change (`INSUFFICIENT_DATA`).
* **Increase** only if both of the last 2 sessions met all of:
  * the rep target and the range target (peak ROM);
  * movement score ≥ 70;
  * ≤ 20% off-plane reps;
  * no score drop > 15 points.

  The range target rises one step if both peaks already reached the
  next level; otherwise reps rise one step.
* **Decrease** (both sessions below target):
  * **peak ROM below the range target in both → lower the range target
    one step.** Below the range target no repetition can be counted, so
    lowering the repetition target (what `conservative_v1` did) could
    not help;
  * otherwise, < 60% of target reps in both → lower reps one step.

  Sessions without any measured in-plane movement are not evidence for
  a decrease.
* Mixed or middling results → maintain. One parameter, one step, within
  bounds. Every decision stores its inputs, previous and proposed
  parameters and a plain-language rationale.
* **Not used:** movement smoothness, experimental posture measurements,
  any compensation classifier output — none is validated. A missing
  compensation output is recorded as "not assessed", never as "no
  compensation".

### Exercise recommendation (`rules_v2`, `src/assessment/recommendation.py`)

1. The diagnosis must be Stroke or Post-stroke hemiparesis.
2. The exercise's joint and region match the affected area, and it
   supports the side.
3. It addresses the recorded impairment. "Movement compensation" and
   "Other" have no matching exercise.
4. Exercises measured in degrees whose **lowest** range target exceeds
   the baseline ROM are not suggested: no repetition could be counted.
5. Starting targets depend on the restriction:
   * "Pain-limited movement" or "Therapist specified": every target at
     its lower bound;
   * otherwise: the defaults, with the range target never above the
     baseline ROM (rounded down to a step). This also covers "Limited
     range".
6. The first eligible exercise in catalogue order is suggested; the
   others are listed as alternatives.

The rules are deterministic and tested. They are not clinically
validated: the restriction handling in particular needs
physiotherapist review. The assessment has no severity field, so
severity is not used.

## Compensation detection status

**The original Kinect-trained BiLSTM is disabled** in the application
and in the standalone prototype `src/pose_estimation/inference.py`. Its
code and weights (`outputs/compensation_model_lstm/`) are unchanged. See
[`docs/compensation_model_card.md`](docs/compensation_model_card.md) for:
* the feature schema;
* the evidence that it predicts "No Compensation" for every
  MediaPipe-style window, including simulated compensations (a 2026-10-28
  diagnostic);
* the fact that its reported fixed-split metrics were selected on the
  test participants (the training script is corrected but not re-run:
  the Toronto data is not on this machine).

**This copy of the project does not contain `outputs/compensation_step_a/`
or `data/toronto/`**, so the Step A figures below could not be
re-verified here.

Why: it was trained on Toronto Rehab Stroke Pose Kinect features
(pre-normalized skeletons, non-causal Savitzky–Golay smoothing, spine
joints, Y-up axes). The live pipeline fed it MediaPipe image coordinates
(Y-down, mirrored X, anisotropic, different depth scale). Verified
effects: trunk-vertical angle off by ~135°, shoulder-line angle by
~156°, shoulder-height sign flipped. Its predictions on live input were
not meaningful.

**Step A models (RF and BiLSTM on `shared_v1`): evaluated, not
validated** for detecting compensation in stroke patients, and not
validated on webcam data at all. They are **not integrated** into the
application: none of them produces patient-facing feedback or feeds
adaptive decisions.

What exists instead:

* **Experimental posture measurements** (`geometric_v1`,
  `src/compensation_detection/geometric_measures.py`): shoulder elevation
  and trunk lean/rotation relative to the session's resting baseline.
  Stored per session and shown to therapists on the Progress page,
  labelled experimental. Not shown to patients; not used for adaptation.
  Session compensation-count columns stay NULL ("not measured").
* **Step A** (`src/compensation_detection/step_a_compare.py`): RF vs
  retrained BiLSTM on `shared_v1` — features computable identically from
  Kinect and MediaPipe (8 shared joints, vertical axis only, unit-free,
  causal). Leave-one-participant-out on Toronto. **Comparison
  complete; final models not trained.** Kinect-only — see the section
  below. Derived analysis: `src/compensation_detection/step_a_analysis.py`.
* **Step B** (prepared, **not started**): labelled webcam data collection
  and external validation of the frozen Step A models —
  [`docs/step_b_protocol.md`](docs/step_b_protocol.md),
  `src/data_collection/recorder.py`,
  `src/compensation_detection/step_b_evaluate.py`.

### Step A results

<!-- STEP_A_RESULTS -->
**Status: COMPLETE (leave-one-participant-out comparison).** All three
models were evaluated on all 19 participants. **No final models were
trained**, so the frozen models that Step B needs do not exist yet.
These results are Kinect-to-Kinect only. They say nothing about
performance on MediaPipe webcam data (Step B) or about clinical validity.

Run: `outputs/compensation_step_a/lopo_run_colab/` (see
`PROVENANCE.txt` there). The earlier, interrupted local run is
preserved unchanged in `outputs/compensation_step_a/lopo_run/`.

* **Compute:** BiLSTM folds H01–H05 were trained on the laptop (NVIDIA
  MX350) and H06–P09 on Google Colab (Tesla T4). The protocol, data,
  seeds and code were identical for all folds, but GPU arithmetic is not
  bit-reproducible across hardware. The RF folds are deterministic: they
  ran on CPU and reproduced the per-fold accuracies of the earlier
  laptop run exactly.
* **Integrity:** `verify_step_a_run_all.py` (output in
  `verification.txt`) checked the following:
  * the configuration fingerprints of all three models match the
    current code and the Toronto data rebuilt on this machine;
  * all 57 fold checkpoints are valid;
  * the pooled metrics recomputed from the saved predictions match
    `metrics_<model>.json`.

Protocol (unchanged from `step_a_compare.py`):
* 20-frame windows, one fold per participant.
* BiLSTM: 2×64 units, at most 30 epochs, early stopping on the macro-F1
  of two inner-validation participants.
* RF: 200 trees, `balanced_subsample`.
* The primary metric is pooled macro-F1 over the four classes.

#### Verified metrics (`metrics.json`)

Pooled over 52,007 windows, 80% of which are "No Compensation".
Baselines need no training.

| Model | Macro-F1 | Accuracy | Macro-F1 healthy (H01–H10) | Macro-F1 patients (P01–P09) |
|---|---|---|---|---|
| Majority class (always "No Compensation") | 0.2220 | 0.7985 | 0.2078 | 0.2463 |
| Random, by label frequency | 0.2521 | 0.6542 | 0.2503 | 0.2315 |
| RF, original 21 Kinect features | 0.4378 | 0.7986 | 0.4584 | 0.2352 |
| RF, `shared_v1` | 0.4204 | 0.8223 | 0.4189 | 0.2449 |
| BiLSTM, `shared_v1` | 0.4611 | 0.7410 | 0.5059 | 0.2195 |

Per-class precision / recall / F1:

| Class (windows) | RF original | RF `shared_v1` | BiLSTM `shared_v1` |
|---|---|---|---|
| No Compensation (41,527) | 0.853 / 0.938 / 0.893 | 0.851 / 0.989 / 0.915 | 0.899 / 0.839 / 0.868 |
| Shoulder Elevation (3,147) | 0.378 / 0.080 / 0.132 | 0.316 / 0.101 / 0.153 | 0.264 / 0.339 / 0.297 |
| Trunk Rotation (4,703) | 0.362 / 0.352 / 0.357 | 0.290 / 0.080 / 0.125 | 0.240 / 0.297 / 0.266 |
| Lean Forward (2,630) | 0.619 / 0.263 / 0.369 | 0.683 / 0.380 / 0.488 | 0.367 / 0.476 / 0.414 |

Confusion matrices (rows are the true class, columns the predicted
class, in the order None, Shoulder Elevation, Trunk Rotation, Lean
Forward):

```
RF original       [[38935, 137, 2152, 303], [2074, 252, 700, 121], [2793, 255, 1654, 1], [1856, 23, 60, 691]]
RF shared_v1      [[41071, 195, 151, 110], [1849, 319, 722, 257], [3775, 456, 374, 98], [1547, 41, 42, 1000]]
BiLSTM shared_v1  [[34822, 2047, 3402, 1256], [1025, 1068, 609, 445], [2026, 817, 1396, 464], [856, 116, 405, 1253]]
```

#### Derived, post-hoc analysis (`analysis.txt`)

This analysis comes from
`python -m src.compensation_detection.step_a_analysis`. It was defined
**after** the metrics above had been seen. It describes the existing
predictions and was not used to change any model, hyperparameter or
the protocol.

Participant-level bootstrap: 2,000 resamples of the 19 participants,
95% intervals.

| | Pooled macro-F1 [95% CI] |
|---|---|
| RF original | 0.438 [0.356, 0.512] |
| RF `shared_v1` | 0.420 [0.371, 0.473] |
| BiLSTM `shared_v1` | 0.461 [0.385, 0.532] |
| RF `shared_v1` − RF original | −0.017 [−0.091, +0.058] |
| BiLSTM − RF `shared_v1` | +0.041 [−0.032, +0.102] |
| BiLSTM − RF original | +0.023 [−0.058, +0.115] |

Binary "any compensation" vs "No Compensation", pooled:

| | Precision | Recall | False alarms on true "No Compensation" |
|---|---|---|---|
| RF original | 0.592 | 0.358 | 6.2% |
| RF `shared_v1` | 0.879 | 0.316 | 1.1% |
| BiLSTM `shared_v1` | 0.495 | 0.627 | 16.1% |
| *Patients only:* RF original | 0.035 | 0.118 | 9.6% |
| *Patients only:* RF `shared_v1` | 0.152 | 0.083 | 1.4% |
| *Patients only:* BiLSTM `shared_v1` | 0.031 | 0.211 | 19.9% |

Per participant: accuracy (verified, from the fold checkpoints) and
false-alarm rate on true "No Compensation" windows (derived). "Majority"
is the accuracy of always predicting "No Compensation".

| Participant | Windows | Classes present | Majority | RF original acc / FA | RF `shared_v1` acc / FA | BiLSTM acc / FA |
|---|---|---|---|---|---|---|
| H01 | 3830 | 1,2,3,4 | 0.734 | 0.838 / 0.001 | 0.793 / 0.000 | 0.870 / 0.015 |
| H02 | 3551 | 1,2,3,4 | 0.616 | 0.674 / 0.009 | 0.677 / 0.001 | 0.686 / 0.076 |
| H03 | 2698 | 1,2,3,4 | 0.567 | 0.689 / 0.081 | 0.590 / 0.024 | 0.608 / 0.085 |
| H04 | 3197 | 1,2,3,4 | 0.755 | 0.792 / 0.008 | 0.819 / 0.003 | 0.757 / 0.113 |
| H05 | 3143 | 1,2,3,4 | 0.737 | 0.689 / 0.117 | 0.777 / 0.001 | 0.808 / 0.141 |
| H06 | 3335 | 1,2,3,4 | 0.679 | 0.719 / 0.029 | 0.722 / 0.008 | 0.539 / 0.386 |
| H07 | 3615 | 1,2,3,4 | 0.698 | 0.658 / 0.059 | 0.721 / 0.010 | 0.741 / 0.050 |
| H08 | 2532 | 1,2,3,4 | 0.771 | 0.817 / 0.014 | 0.833 / 0.015 | 0.768 / 0.107 |
| H09 | 5896 | 1,2,3,4 | 0.764 | 0.847 / 0.003 | 0.781 / 0.000 | 0.725 / 0.145 |
| H10 | 2707 | 1,2,3,4 | 0.749 | 0.802 / 0.129 | 0.806 / 0.050 | 0.687 / 0.257 |
| P01 | 2764 | 1 | 1.000 | 1.000 / 0.000 | 1.000 / 0.000 | 0.991 / 0.009 |
| P02 | 1330 | 1,2,3 | 0.960 | 0.960 / 0.000 | 0.953 / 0.007 | 0.860 / 0.104 |
| P03 | 1514 | 1,2 | 0.968 | 0.966 / 0.001 | 0.968 / 0.000 | 0.605 / 0.375 |
| P04 | 2081 | 1,3 | 0.927 | 0.927 / 0.000 | 0.927 / 0.000 | 0.886 / 0.045 |
| P05 | 1455 | 1,3,4 | 0.866 | 0.606 / 0.303 | 0.773 / 0.107 | 0.766 / 0.116 |
| P06 | 3880 | 1 | 1.000 | 0.922 / 0.078 | 1.000 / 0.000 | 0.851 / 0.149 |
| P07 | 2204 | 1 | 1.000 | 0.636 / 0.364 | 0.998 / 0.002 | 0.416 / 0.584 |
| P08 | 2101 | 1,2,4 | 0.972 | 0.901 / 0.072 | 0.931 / 0.042 | 0.703 / 0.277 |
| P09 | 174 | 1,3 | 0.994 | 0.994 / 0.000 | 0.994 / 0.000 | 0.931 / 0.064 |

#### Interpretation, limited to what this evidence supports

* **All three models exceed both baselines in pooled macro-F1.** The
  differences between models are not distinguishable at this sample
  size: every pairwise 95% interval includes zero. These results do
  not identify a best model.
* **The models make different trade-offs.**
  * The BiLSTM detects more compensation (recall 0.63) at a 16% false-
    alarm rate.
  * RF `shared_v1` raises few false alarms (1.1%) but misses about two
    thirds of compensation windows.
  * Which trade-off is acceptable is a clinical design decision. It is
    not settled by these numbers.
* **`shared_v1` features:** replacing the original Kinect features with
  `shared_v1` (both with RF) produced no detectable change in pooled
  macro-F1. The effect differs by class:
  * Trunk Rotation recall fell from 0.35 to 0.08;
  * Lean Forward recall rose from 0.26 to 0.38;
  * Shoulder Elevation stayed poorly detected by both (recall
    0.08–0.10).
* **Patients:**
  * Only 508 of the 17,503 patient windows (2.9%) are labelled as
    compensation, and four of the nine patients have none.
  * Patient macro-F1 is 0.22–0.24 for every model, no better than
    always predicting "No Compensation" (0.246). Binary detection F1 on
    patients is at most 0.11.
  * The models' discrimination comes almost entirely from healthy
    participants performing acted compensations. **There is no evidence
    here that any model detects compensation in stroke survivors.**
* **BiLSTM false alarms on patients:** the BiLSTM flags many windows
  from patients labelled as never compensating:

  | Patient | Normal windows flagged |
  |---|---|
  | P07 | 58% |
  | P03 | 38% |
  | P08 | 28% |
  | P06 | 15% |

* **Nothing here supports showing any of these models to patients or
  using them in adaptive decisions.** The compensation model stays
  disabled in the application (`COMPENSATION_MODEL_ENABLED = False`).
  Any further use requires Step B and a separate decision.
<!-- /STEP_A_RESULTS -->

## Database

SQLite at `data/rehab.db` (not in git). Schema upgrades are additive and
idempotent (`src/database/migrations.py`, tracked by `PRAGMA
user_version`, currently 2); a hashed, integrity-checked backup is
written to `data/backups/` before an existing database is migrated.

Schema 2 adds these session columns:
* `peak_rom`, `dtw_similarity`, `dtw_mean_error_deg`;
* `session_status`, `interrupted_reps`;
* `diagnostics_json`: pose model settings, threading, frame size and
  per-stage timing.

It also adds the `session_frames` table (the per-frame log).

Historical sessions keep NULL in all new columns. `metrics_version` NULL
marks sessions recorded by the original implementation, and 1 marks the
first generic analyzer. Neither is used by the current policy. A dry run
on a copy of the development database migrated 1 → 2 with every row
preserved.

## Validation status

| What | Evidence | Status |
|---|---|---|
| Automated tests (`python -m pytest`, Windows 11, Python 3.12.13, 2026-10-28) | 428 passed, 0 failed, 2 skipped (Toronto data not present; Step A final models not trained). All modules now collect and run on Windows | passing; synthetic inputs only |
| Static check (`pyflakes src tests`) | 4 warnings, all in original training scripts not modified in this work | — |
| Original rep counting / segmentation / DTW (legacy config) | characterization tests reproduce the original outputs exactly | preserved |
| Metrics v2 definitions (time-based counting, peak ROM, path-normalised DTW, glitch/gap handling) | automated, synthetic poses | implemented and tested; **not validated on webcam data** |
| MediaPipe world-axis convention and trunk frame orientation | real MediaPipe on one public video | consistent (see Coordinate conventions) |
| Angle accuracy, plane classification, rep-count agreement, score discrimination | `docs/validation_protocol.md`, `src/validation/evaluate_sessions.py` | **NOT TESTED**: no labelled recordings exist |
| Live webcam sessions | 3 sessions on the Windows laptop (2026-10-22/28; complexity 2, GUI-thread processing): 5.7–7.8 fps, arm tracked in 63–80% of frames | all `UNABLE_TO_ASSESS`; **no live attempt has been assessable** |
| Sustained processing capacity (threaded pipeline, video file instead of webcam, crowded multi-person scene, i5-13420H) | complexity 0: 9.2 fps, 1: 3.7 fps, 2: 2.2 fps | pessimistic scene; live single-person rate must be measured |
| Adaptive loop end to end (controller, approval, rejection, next attempt, range-limited patient) | automated, synthetic sessions through the real session page and database | implemented and tested; not demonstrated live |
| Step A model comparison (Toronto, Kinect) | README section above | not re-verifiable in this copy (outputs missing) |
| Compensation detection on MediaPipe | diagnostic: original model outputs "No Compensation" for everything | **NOT VALIDATED, not deployed**; Step B not started |
| Clinical validity of any measure, threshold or rule | — | **NOT VALIDATED** |

Synthetic and simulated inputs show the software behaves as designed;
they are not evidence of accuracy on real webcam data.

## Testing guide

| Area | Tests |
|---|---|
| Original rep counting / segmentation / DTW (golden values) | `test_characterization_shoulder_raise.py` |
| Live widget errors, overlay mirroring, missing data, BiLSTM disabled | `test_live_widget_robustness.py` |
| Exercise angles, analyzer, catalogue, recommender | `test_exercises.py` |
| Arm-tracking percentage, required landmarks, affected-side selection, live tracking status | `test_tracking.py` |
| Session page workflow and persistence | `test_rehab_session_flow.py`, `test_assessment_approval.py` |
| Database migration and backup | `test_migrations.py` |
| Adaptive policy branches and full loop | `test_adaptation_policy.py`, `test_adaptive_loop_end_to_end.py` |
| Adaptive controller: rules, safety, uncertainty, compensation exclusion, propagation to the next attempt | `test_adaptive_controller.py` |
| Progress page, dashboard | `test_progress_and_dashboard.py` |
| Experimental posture measurements | `test_geometric_measures.py` |
| Metrics v2: DTW normalisation, fps-independent counting, glitches, gaps, peak ROM, mirrored input, per-frame log | `test_metrics_v2.py` |
| Worker thread, unmirrored input, persistence of v2 metrics and frame logs, save failure, app close, range-limited patient loop, rejection | `test_live_pipeline_v2.py` |
| Recommendation rules (baseline ROM, restrictions, eligibility) and therapist-edited parameters | `test_recommendation_rules.py` |
| Compensation-model input guard, live/training feature mismatch, validation tools | `test_compensation_and_validation_tools.py` |
| shared_v1 features, Step B tooling | `test_shared_features.py`, `test_step_b_tooling.py` |
| Step A checkpointing/resume, derived analysis | `test_step_a_checkpointing.py`, `test_step_a_analysis.py` |

Live webcam validation: follow
[`docs/webcam_validation_checklist.md`](docs/webcam_validation_checklist.md)
and the quantitative protocol
[`docs/validation_protocol.md`](docs/validation_protocol.md) (criteria in
[`docs/validation_criteria.json`](docs/validation_criteria.json), fixed
before analysis).

## Known limitations

* **Live webcam testing is limited to a smoke test.** The exercise
  thresholds and the 3D angle conventions (MediaPipe world axes) are
  unverified against any reference measurement.
* **No validated compensation detection.** The original BiLSTM is
  disabled. The Step A comparison is complete but Kinect-only: no
  model was distinguishable from the others, and none exceeded the
  majority baseline on stroke patients. Step B has not started, and
  geometric posture measurements are experimental.
* **Thresholds and adaptive rules are engineering defaults**, not
  clinically validated; therapist review is required.
* **Reference movements are synthetic**, not clinician recordings.
* **Forward reach and trunk-related measures depend on MediaPipe depth**,
  which is noisy.
* <a id="frame-rate"></a>**Frame rate is the current release blocker.**
  * MediaPipe inference dominates the per-frame cost. Analysis and
    drawing take about 7 ms. Moving capture and inference to a worker
    thread keeps the interface responsive but cannot by itself raise the
    frame rate.
  * Earlier measurements:
    * Ubuntu laptop (i5-1135G7, battery): complexity 2 took about 95 ms
      per frame, giving 8.7–9.3 fps.
    * Windows laptop (i5-13420H): 5.7–7.8 fps in live sessions.
  * In both benchmarks on the Windows laptop, complexity 1 was about
    3× faster than complexity 2, so the default is now **1**
    (`src/pose_estimation/config.py`). Whether it reaches 10 fps live
    has **not been measured yet**; every session now records its
    per-stage timing in `diagnostics_json`.
  * If the rate stays below 10 fps:
    * use mains power and a performance power plan;
    * try `REHAB_POSE_COMPLEXITY=0` (lite, lower accuracy);
    * as a code change, read the camera on a separate thread so that
      decoding overlaps inference.
  * The 10 fps gate itself is unchanged.
* **Setup movement before and after an exercise** is analysed like
  exercise movement:
  * Frames without a person count against the tracking percentage.
  * Large setup movements can form extra scored segments, or reps that
    the movement-plane check rejects as off-plane (session 42: 4 long,
    jerky segments at 36–105° from the forward plane).
* **Bilateral impairment** is analysed one arm at a time (left by default,
  as in the original application).
* **Historical sessions** recorded before metrics v1 are shown separately
  and are not used by the adaptive policy.
* The original Toronto training scripts in `src/compensation_detection/`
  (`train_*`, `build_*`) still use paths relative to the project root.
* **Live sessions recorded before metrics v2** (the 3 sessions in the
  development database) are kept but never used by the adaptive policy.

## Project layout

```
src/
  ui/                 PyQt6 application (app.py, main_window.py, pages/)
  exercises/          catalogue, angle measurements, generic analyzer
  adaptation/         per-attempt controller, policy, approval workflow
  progress/           history grouping/formatting for the UI
  database/           SQLite access and migrations
  compensation_detection/
                      original Toronto/Kinect pipeline (training scripts,
                      disabled BiLSTM), geometric_measures, shared_v1,
                      Step A comparison, Step B evaluation
  data_collection/    Step B landmark recorder
  simulation/         synthetic poses/sessions for tests (not data)
  quality_assessment/ DTW
  validation/         axis-convention check, evaluation of recorded
                      sessions against manual counts / goniometer
  pose_estimation/    shared MediaPipe settings (config.py); standalone
                      prototype (inference.py, BiLSTM disabled)
  paths.py            project-root-relative paths
tests/                pytest suite
docs/                 validation protocol and criteria, Step B protocol,
                      compensation model card, webcam checklist
```

## Housekeeping awaiting a decision

These duplicates are kept untouched pending the owner's approval:

* `ui/` (top level): an older copy of `src/ui/reference_skeleton.py` and
  an empty file; nothing imports it.
* `src/database/db_backup.py`: an older copy of `db.py` (missing two
  tables); nothing imports it.
* `data/toronto/robot/data_new`: byte-identical to
  `data/toronto/data_new`.

`pip check` reports version conflicts only for packages this project
does not use (`mmpose`, `mmengine`, `openxlab`).
