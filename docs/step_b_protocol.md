# Step B — Labelled MediaPipe data collection and external validation

Status: **PREPARED, NOT STARTED.**

* The Step A leave-one-participant-out comparison is **complete and
  verified**. See the README's "Step A results" section and
  `outputs/compensation_step_a/lopo_run_colab/`.
* **Remaining prerequisite:** the frozen models in
  `outputs/compensation_step_a/final_models_not_deployed/` do not exist
  yet. They are produced by `step_a_compare --final-models`, a training
  run that needs separate approval and has not been run.
* The tooling is implemented and has been tested offline with
  synthetic data only. No recordings exist yet.
* Collection requires a webcam, approval of this protocol and, for any
  human participants, the ethics and consent arrangements below.

### What Step A established, and what it did not (Kinect data only)

* **All models beat the majority baseline overall.** Pooled macro-F1
  over 19 participants:
  * RF original features 0.438;
  * RF `shared_v1` 0.420;
  * BiLSTM `shared_v1` 0.461;
  * majority baseline 0.222.

  The differences between models were not distinguishable: every
  participant-bootstrap 95% interval of the pairwise differences
  includes zero.
* **The models trade false alarms for recall.**
  * BiLSTM: recall of any compensation 0.63, false alarms on 16% of
    "No Compensation" windows.
  * RF `shared_v1`: recall 0.32, false alarms 1.1%.
* **Shoulder Elevation was poorly detected by every model** (F1
  0.13–0.30). With `shared_v1`, RF detected Trunk Rotation poorly
  (recall 0.08).
* **On stroke patients, no model exceeded the majority baseline**
  (patient macro-F1 0.22–0.24 versus 0.246).
  * Only 2.9% of patient windows are labelled as compensation.
  * The BiLSTM flagged 15–58% of normal windows in four patients who
    were never labelled as compensating.
* **Nothing in Step A shows that any model works on MediaPipe data or
  in stroke survivors.** Step B tests the first of these, on acted
  compensation. Phase B1 cannot test the second.

The team may wish to consider, **before any Step B data are analysed**,
whether to report the false-alarm rate on "No Compensation" windows as
a secondary metric (section 7). Step A showed that pooled macro-F1
hides this trade-off. This is a suggestion only: the analysis plan
below has not been changed.

## 1. Purpose

Step A evaluated compensation classifiers on Kinect skeletons from the
Toronto Rehab Stroke Pose dataset. Two of them used `shared_v1`, a
feature set designed to be computable identically from MediaPipe world
landmarks. Kinect-to-Kinect scores do **not** show that these models
work on webcam data.

Step B answers one question first:

> Do the **frozen** Step A models (`outputs/compensation_step_a/
> final_models_not_deployed/`) classify compensation in labelled
> MediaPipe recordings better than a majority-class baseline?

Fine-tuning on the new data is a separate, later decision (section 8).

## 2. Ethics, consent and data protection

* Recording people — including healthy volunteers — for research
  normally requires approval from the relevant institutional ethics
  committee. Obtain it before recording anyone.
* Written informed consent; participants may withdraw and have their
  data deleted.
* Use pseudonymous participant codes (`S01`, `S02`, …). Keep the code
  key separately from the data. No names in file names or metadata.
* The recorder stores **landmarks only, not video** (`video_saved:
  false`). Video would only be recorded under explicit, separate consent
  and is not supported by the current tool.
* Stroke survivors: only under clinical supervision, with a therapist
  deciding whether simulated compensations are safe. The initial phase
  uses healthy adults only.

## 3. Participants

* Phase B1: at least 10 healthy adults (more is better; subject-level
  confidence intervals with fewer than ~8 participants are very wide).
* Record which participants also took part in any development work, so
  they can be excluded from the external test set.
* Phase B2 (optional, separate approval): stroke survivors with
  naturally occurring compensation, labelled by a therapist.

Healthy participants **simulate** compensation. Results from Phase B1
describe detection of acted compensation by healthy adults, as in most
of the Toronto data, and must be reported as such.

## 4. Setup

* Webcam on a tripod, upright (level), at about shoulder height,
  2–2.5 m in front of a chair without armrests, frontal view.
  `shared_v1` assumes the camera's vertical axis is the true vertical.
* Whole trunk including both hips visible throughout; no table in front
  of the participant (tables hide the hips).
* Even lighting, plain background, fitted clothing.
* Same MediaPipe settings as the application: the shared
  `src/pose_estimation/config.py` (`model_complexity` from
  `REHAB_POSE_COMPLEXITY`, default 1; `smooth_landmarks=True`;
  confidence 0.7). The recorder uses this config and stores the settings
  in each recording's metadata. Use the same complexity for every
  recording and for the application being evaluated.
* Note the camera model, resolution and measured frame rate (the
  recorder stores the frame rate).

## 5. Tasks and conditions

For each participant, each arm (LEFT and RIGHT), each task:

| Task | Why |
|---|---|
| `forward_reach` (seated, reach forward and back) | matches Toronto `Rch_Fwr_Bck` |
| `side_to_side_reach` (seated) | matches Toronto `Rch_Sd2Sd_Bck` |
| `shoulder_flexion`, `shoulder_abduction` | the application's exercises (not in Toronto: evaluated separately) |

Conditions per task (5 repetitions each, order randomised per
participant, written instruction read aloud):

1. Normal movement (label 1)
2. Shoulder elevation / hiking on the moving side (label 2)
3. Trunk rotation towards the moving side (label 3)
4. Leaning the trunk forward (label 4)

Rest 2–3 s between repetitions with the arm relaxed.

## 6. Labelling

Two layers, kept separate:

1. **Operator labels (live).** During recording the operator presses
   `1`–`4` when the instructed condition begins and `0` while resting or
   setting up. Stored per frame in the `.npz` (`operator_label`).
2. **Reviewer annotation (offline).** A second person, blind to the
   intended condition where possible, reviews a skeleton replay and
   writes `annotations.csv`:
   `recording_id,start_s,end_s,label,annotator` (seconds from the
   start of the recording). Frames outside annotated intervals are
   treated as unlabelled.

Report agreement between operator and reviewer labels (Cohen's κ on
frames labelled by both). A subset (≥20%) should be annotated by a
second reviewer to report inter-rater κ. Where a performed movement did
not match the instruction (e.g. no visible compensation), the reviewer
label takes precedence; never relabel to agree with a model output.

## 7. Analysis plan (fix before looking at results)

* Command:
  `python -m src.compensation_detection.step_b_evaluate --recordings
  data/step_b --annotations data/step_b/annotations.csv --out
  outputs/step_b_report.json`
* Unit: 20-frame windows after resampling to 30 Hz, label-homogeneous,
  frames with any of the 8 shared joints below visibility 0.5 excluded.
* Primary metric: pooled macro-F1 over the four classes with a 95%
  participant-level bootstrap CI; compared with the majority-class
  baseline (always "No Compensation").
* Secondary: per-class F1, per-participant macro-F1, per task (Toronto-
  like tasks vs application exercises), frame rate achieved.
* Decision thresholds (to be set by the project team **before** the
  data are analysed; suggested starting point): a model may be shown to
  therapists as an *experimental* indicator only if the lower CI bound
  of its macro-F1 exceeds the baseline's point estimate and no class has
  F1 below an agreed floor. It may not be shown to patients or used in
  adaptation on the basis of Phase B1 alone.

## 8. After external validation

Only if the external results justify it:

* Fine-tune or retrain with the new data using **leave-one-participant-
  out** (or a fixed held-out participant set never used for any choice).
* Keep synthetic or augmented data (e.g. Kinect skeletons with simulated
  MediaPipe noise) in a separate training-only pool; never in evaluation.
* Any model change gets a new version name and a new external
  evaluation; existing weights are never overwritten.

## 9. Known limitations to state with any result

* Healthy participants acting compensation ≠ stroke survivors'
  compensation.
* Kinect and MediaPipe hip landmarks may not be anatomically identical.
* MediaPipe depth is noisy; `shared_v1` avoids most depth-dependent
  quantities but trunk rotation still depends on it.
* Toronto exercises are reaching tasks; the application's flexion and
  abduction exercises are outside the training distribution.
* The Step A models learned from Toronto labels, which come almost
  entirely from healthy participants acting compensations. In Step A
  they did not beat the majority baseline on stroke patients.
* Part of the Step A BiLSTM evaluation ran on different GPUs (folds
  H01–H05 on an MX350, the rest on a T4). Results of the same protocol
  rerun on other hardware may differ slightly.
