# Stroke Rehab — camera-based upper-limb exercise analysis

A PyQt6 desktop application that uses a webcam and MediaPipe Pose to
guide seated upper-limb exercises, count repetitions, compare each
movement with a reference using Dynamic Time Warping (DTW), and propose
conservative next-session difficulty changes for therapist approval.

> **Research prototype.** Nothing in this repository has been clinically
> validated. Live webcam behaviour of the current version has **not been
> tested** (see [Validation status](#validation-status)).

## Quick start

Requires Python 3.12 (tested with 3.12.3 on Ubuntu, using the project's
`venv/`).

```bash
python3 -m venv venv
venv/bin/pip install -r requirements-dev.txt   # or requirements.txt
venv/bin/python -m src.database.migrations     # verified backup + schema upgrade
venv/bin/python -m src.ui.app                  # start the application
```

`python src/ui/app.py` also works from any directory; all data paths
are resolved from the project root (`src/paths.py`).

Run the tests (no webcam needed; Qt renders offscreen):

```bash
venv/bin/python -m pytest                       # full suite
venv/bin/python -m pytest tests/test_characterization_shoulder_raise.py  # original rep/DTW behaviour
venv/bin/python -m pyflakes src tests           # static check
```

`pytest.ini` disables third-party plugin autoloading because ROS
packages on some machines' `PYTHONPATH` register incompatible plugins.
Tests never touch `data/rehab.db`: they use temporary databases.

## Using the application

1. **Patients** — add a patient and the affected side/joint.
2. **Assessment** — record a clinical assessment. A rule-based
   recommender suggests an implemented exercise; the therapist may
   choose another supported exercise, then approves it. Approval stores
   the exercise, side and starting parameters.
3. **Rehabilitation** — shows the approved exercise, targets and
   instructions. *Start Exercise* opens the webcam; the reference
   animation (synthetic, labelled as such) plays beside the live view.
   *End Session* saves the session (one attempt) and runs the adaptive
   controller. The controller shows feedback and what to do next
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
  * the attempt's session row: reps, ROM peak, DTW similarity,
    off-plane reps, frames measured, frame errors and fps;
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
    exercise, other settings, or not the latest session), tracking is
    poor (the same gate as the policy, below), or recorded values are
    invalid (NaN, similarity outside 0–100, negative counts). Nothing is
    scored and settings are unchanged.
  * `REPEAT`: this attempt missed a target, which can be reps, ROM, DTW
    similarity ≥ 70 or ≤ 20% off-plane reps. No DTW score counts as
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
* **Thresholds:** the controller adds none; it uses `PolicyConfig`. The
  action, feedback and compensation note are stored in
  `adaptation_decisions.inputs_json["controller"]`, so no schema change
  was needed.
* **Approval:** every change of settings, easier or harder, still
  requires therapist approval.

Policy `conservative_v1` (all thresholds configurable in `PolicyConfig`):

* Uses only sessions of the same exercise and side with the current
  metrics version, the **current parameters**, arm tracked in ≥ 80% of
  frames, ≤ 5% and < 30 frame errors (30 consecutive errors halt live
  analysis, leaving the rest of the session unmeasured) and a known
  frame rate ≥ 10 fps. Excluded sessions and the reason are logged.
* Fewer than 2 such sessions → no change (`INSUFFICIENT_DATA`).
* **Increase** only if both of the last 2 sessions met the rep target,
  the range target and DTW similarity ≥ 70, with ≤ 20% off-plane reps
  and no similarity drop > 15 points. The range target rises one step if
  both peaks already reached the next level; otherwise reps rise one step.
* **Decrease** if both sessions completed < 60% of target reps (reps
  down one step) or peaked more than two steps below the range target.
* Mixed or middling results → maintain. One parameter, one step, within
  bounds. Every decision stores its inputs, previous and proposed
  parameters and a plain-language rationale.
* Recorded values that cannot be correct (NaN, similarity outside
  0–100, negative counts) exclude a session.
* **Not used:** movement smoothness, experimental posture measurements,
  any compensation classifier output — none is validated. A missing
  compensation output is recorded as "not assessed", never as "no
  compensation".
* Every change requires therapist approval; a proposal based on a
  prescription that is no longer active cannot be approved.

## Compensation detection status

**The original Kinect-trained BiLSTM is disabled** in the application
and in the standalone prototype `src/pose_estimation/inference.py`. Its
code and weights (`outputs/compensation_model_lstm/`) are unchanged.

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
user_version`); a hashed, integrity-checked backup is written to
`data/backups/` before an existing database is migrated. Historical
sessions keep NULL in all new columns; `metrics_version` NULL marks
sessions recorded by the original implementation, whose angle columns
are not comparable with current measurements.

## Validation status

| What | Status |
|---|---|
| Automated tests (`pytest`) | 283 passed, 0 failed, 1 skipped (2026-10-14). The skipped test loads Step A final models, which have not been trained (`--final-models` not run) |
| Static check (`pyflakes src tests`) | 4 warnings, all unused variables/imports in original training scripts not modified in this work; none in new or changed code |
| Rep counting / segmentation / DTW of the original implementation | preserved: characterization tests reproduce the original outputs exactly |
| Multi-exercise analysis | offline only, synthetic poses |
| Adaptive loop end to end (controller, approval, next attempt) | offline only, synthetic sessions through the real session page and database |
| Live webcam operation | **Smoke-tested only**: 2 Shoulder Flexion sessions on one laptop webcam (2026-10-14). After the required-landmark fix, the arm was tracked in 88% of frames (previously 38%), but frame rate was 8.7 fps, below the 10 fps gate, so **no attempt has yet been assessable**. Angles, thresholds and DTW scores are not validated against a reference measurement |
| Step A model comparison (Toronto, Kinect) | **COMPLETE**, all 19 folds × 3 models, integrity verified (`outputs/compensation_step_a/lopo_run_colab/`). Pooled macro-F1: RF original 0.438, RF `shared_v1` 0.420, BiLSTM 0.461 (majority baseline 0.222); differences not distinguishable; no model beats the baseline on patients. Final models not trained |
| Compensation detection on MediaPipe | **NOT VALIDATED**; Step B not started |
| Clinical validity of any measure or threshold | **NOT VALIDATED** |

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
| shared_v1 features, Step B tooling | `test_shared_features.py`, `test_step_b_tooling.py` |
| Step A checkpointing/resume, derived analysis | `test_step_a_checkpointing.py`, `test_step_a_analysis.py` |

Live webcam validation: follow
[`docs/webcam_validation_checklist.md`](docs/webcam_validation_checklist.md).

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
* **Frame rate is the current release blocker on the development
  laptop** (Intel i5-1135G7, measured on battery with the powersave
  governor):
  * MediaPipe Pose with `model_complexity=2` (heavy) takes ~95 ms per
    frame, about 94% of processing time. That caps the rate at ~10 fps
    before camera decoding, and the webcam sessions achieved 8.7–9.3 fps.
  * The 10 fps gate correctly marks those attempts UNABLE_TO_ASSESS.
  * Input resolution does not change this (95–98 ms per frame from
    640×360 to 1280×720).
  * `model_complexity=1` measured ~42 ms per frame (~20 fps), but it
    would change the pose model used for every measurement (and for the
    Step B recorder). It is not applied.
  * Performance on mains power with a performance power profile has not
    been measured.
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
  paths.py            project-root-relative paths
tests/                pytest suite
docs/                 Step B protocol
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
