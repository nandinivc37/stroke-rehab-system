# Webcam validation checklist

Everything below is **NOT TESTED** until a row is filled in during a
real webcam session. Record the date, tester, camera model, resolution
and machine for each run. Use a test patient, not a real one, and a
copy of the database (`data/rehab.db` → back up first).

Mark each item **PASS / FAIL / N/A** with a note. A FAIL is a finding,
not something to tune away silently.

## 0. Setup

| # | Check | Result |
|---|---|---|
| 0.1 | `venv/bin/python -m pytest` passes on the test machine | |
| 0.2 | App starts with `venv/bin/python -m src.ui.app`; camera opens from the Rehabilitation page | |
| 0.3 | Camera upright, ~shoulder height, 2–2.5 m, whole trunk incl. hips visible | |
| 0.4 | Live view is mirrored; overlay text reads normally (not mirrored) | |
| 0.5 | Unplug the camera mid-session: a clear message appears, the app does not crash | |

## 1. Frame rate and tracking coverage

| # | Check | Result |
|---|---|---|
| 1.1 | Session `mean_fps` (Progress → session detail / DB) — record value. Policy needs ≥ 10 fps; if lower, adaptation will never change parameters | |
| 1.2 | Arm-tracked % ≥ 80% for a normal seated session | |
| 1.3 | Hips hidden (e.g. behind a table): shoulder angles still measured (camera-vertical fallback); note how often | |
| 1.4 | Walking out of frame: reps do not change, "Pose not detected" shown | |

## 2. Per exercise (repeat for LEFT and RIGHT)

For each of **Shoulder Flexion, Shoulder Abduction, Elbow Flexion,
Forward Reach**, with a goniometer or a phone inclinometer as reference
where possible:

| # | Check | Result |
|---|---|---|
| 2.1 | Reference animation shows the correct movement and side (flexion/reach/elbow: side view; abduction: front view) | |
| 2.2 | Displayed angle at rest, ~45°, ~90° and peak vs reference measurement (record the differences) | |
| 2.3 | 5 deliberate full repetitions above target → 5 reps counted | |
| 2.4 | 3 partial repetitions below target → 0 reps; attempts still scored | |
| 2.5 | Holding the arm still at the top does not add reps; fast and slow reps both count | |
| 2.6 | Flexion: 3 sideways raises → 0 reps, 3 "off-plane" (and vice versa for abduction) | |
| 2.7 | Elbow flexion started with the forearm on a table: note behaviour (expected to fail the instructions) | |
| 2.8 | Forward reach with trunk lean only (arm still): note whether "reach" increases (known depth/lean limitation) | |
| 2.9 | Movement similarity (DTW) higher for a movement that follows the reference timing than for a jerky or partial one | |

## 3. Session persistence

| # | Check | Result |
|---|---|---|
| 3.1 | After *End Session*, a new row exists with exercise, side, parameters, reps / target, peak, similarity, fps, coverage | |
| 3.2 | A session with no movement stores similarity as missing ("—"), not 0 | |
| 3.3 | Progress page shows the session in the correct exercise group with trend points | |
| 3.4 | Experimental posture measurements appear in session detail, labelled experimental, and **not** on the patient overlay | |

## 4. Therapist-approved adaptation

| # | Check | Result |
|---|---|---|
| 4.1 | First session for a new prescription → "not enough sessions yet", no proposal | |
| 4.2 | Two eligible sessions meeting all targets → one PENDING proposal (one parameter, one step) with a readable rationale | |
| 4.3 | Proposal not applied before approval (Rehabilitation page still shows old targets) | |
| 4.4 | Approve with a reviewer name → next session shows and records the new targets | |
| 4.5 | Reject → targets unchanged; rejected proposal cannot later be approved | |
| 4.6 | Low-fps or poorly tracked session → excluded, reason visible in the decision log | |

## 5. Not covered by this checklist

* Accuracy of compensation detection (no validated classifier; Step B).
* Clinical appropriateness of thresholds, targets or step sizes
  (requires therapist review).
* Performance with stroke survivors.
