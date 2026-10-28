# Compensation models: model card and deployment decision

**Deployment status: NOT deployed.** No compensation classifier runs in
the application. `COMPENSATION_MODEL_ENABLED = False` in
`src/pose_estimation/inference.py`; nothing under `src/ui` loads a model.

## Classes

| Index | Toronto label | Class |
|---|---|---|
| 0 | 1 | No Compensation |
| 1 | 2 | Shoulder Elevation |
| 2 | 3 | Trunk Rotation |
| 3 | 4 | Lean Forward |

Stored class index = Toronto label − 1.

## Original checkpoint: `outputs/compensation_model_lstm/compensation_lstm_model.pt`

| Item | Value |
|---|---|
| Architecture | 2-layer bidirectional LSTM, 64 units per direction, dropout 0.3, linear head on the last time step |
| Input | 20-frame window × 21 features |
| Feature order | `FEATURE_COLUMNS` in `src/compensation_detection/build_temporal_windows.py` (7 base features, then their frame-to-frame deltas, then 5-frame rolling standard deviations) |
| Feature scaling | none (raw degrees and shoulder-width units) |
| Training data | Toronto Rehab Stroke Pose, Kinect v2 skeletons: pre-normalised to the spine-shoulder joint and shoulder width, Savitzky-Golay smoothed (non-causal), Y-up axes, spine joints used |
| Split | train H03 H04 H05 H07 H08 H09 H10 P01 P03 P04 P05 P06 P07 P08 P09; test H01 H02 H06 P02 (participant-wise; no participant in both) |
| Schema stored in checkpoint | **no** (input size, hidden size, layers, classes only) |

### The reported test metrics are optimistically biased

`train_lstm_classifier.py` (original version) evaluated the **test**
participants after every epoch and used that macro-F1 to pick the best
epoch, to drive the learning-rate scheduler and to stop early. The
reported test macro-F1 of 0.548 (`metrics.json`) was therefore used for
model selection and is not an untouched held-out estimate. It must not
be presented as one.

The script has been corrected (v2). The best epoch, scheduler and early
stopping now use validation participants H05 and P05, taken from the
training set. The test participants are evaluated once, after training,
together with the majority-class baseline. It writes to
`outputs/compensation_model_lstm_v2/`, and the checkpoint stores the
feature names, class names, split, scaling and coordinate source. **It
has not been run**: the Toronto data (`data/processed_toronto/`) is not
on this machine. Expected cost: tens of minutes on a laptop GPU, longer
on CPU.

The leave-one-participant-out comparison (`step_a_compare.py`) already
used inner validation participants. Its results (README, "Step A
results") cannot be re-checked on this machine because
`outputs/compensation_step_a/` is not present in this copy of the
project.

## Why the model cannot be used on webcam (MediaPipe) input

Established by inspecting the code and by running it:

* `live_features.py` computes the 21 features from MediaPipe
  **normalised image** landmarks (y down, x mirrored relative to Kinect,
  anisotropic). For an upright trunk, `trunk_vertical_angle` is about
  180° live against about 0° in training (test:
  `test_live_trunk_angle_uses_a_different_convention_from_training`).
* Training used spine joints, which MediaPipe does not have. It also used
  Kinect pre-normalisation and non-causal smoothing; live features use
  neither.
* Diagnostic run (2026-10-28, synthetic MediaPipe-style poses performing
  arm raises, 61 windows per condition). The original checkpoint
  predicted **"No Compensation" for every window** in every condition:
  neutral, shoulder elevation (6 cm hike), trunk rotation 25° and
  forward lean 25°. Mean confidence was 0.76–0.90. This reproduces the
  observed "always No Compensation" behaviour. The cause is
  out-of-distribution input, not a lack of compensation.

`CompensationLSTM.check_input_schema()` now refuses features whose
source is not `toronto_kinect`, and features whose names or order differ
from those stored in a checkpoint.

## What would be needed to deploy compensation detection

1. Features computable identically from both sensors: `shared_v1`
   (Step A).
2. Final Step A models trained (`step_a_compare --final-models`, needs
   approval) and frozen.
3. Labelled MediaPipe recordings and external evaluation: Step B
   (`docs/step_b_protocol.md`). This needs ethics approval and at least
   10 participants.
4. A deployment decision based on the Step B result, which must exceed
   the majority baseline with a confidence interval.

Even then, Step B uses healthy adults acting compensations. On stroke
patients in Toronto, no model beat the majority baseline. Compensation
output must never drive difficulty changes without separate validation;
the adaptive controller records it as context only.
