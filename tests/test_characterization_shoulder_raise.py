"""
Characterization tests for the ORIGINAL shoulder-raise analysis.

Golden values were recorded by running the unmodified LivePoseWidget
(commit 22e71b3) on the scripted inputs in tests/legacy_inputs.py.
They pin down rep counting, movement segmentation and the DTW-based
similarity score so that refactoring cannot silently change them.

Inputs are synthetic; these tests say nothing about webcam accuracy.
"""

import pytest

from tests.legacy_inputs import (
    SCENARIOS,
    occluded_arm_results,
    scenario_results,
)
from tests.widget_runner import run_legacy_widget

GOLDEN = {
    "three_full_reps": dict(reps=3, max=120.0, min=0.0, traj=153,
                            sim=96.16, segs=3, active=False),
    "partial_raises": dict(reps=0, max=70.0, min=0.0, traj=96,
                           sim=87.77, segs=2, active=False),
    "glitch_spikes": dict(reps=0, max=95.0, min=0.0, traj=24,
                          sim=82.63, segs=4, active=False),
    "ends_mid_movement": dict(reps=1, max=120.0, min=0.0, traj=68,
                              sim=94.46, segs=1, active=True),
    "hover_between_thresholds": dict(reps=1, max=100.0, min=0.0, traj=42,
                                     sim=93.5, segs=0, active=True),
}

OCCLUDED_GOLDEN = dict(reps=1, max=120.0, min=0.0, traj=40, sim=97.67,
                       segs=1, active=False)


@pytest.mark.parametrize("side", ["LEFT", "RIGHT"])
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_legacy_scenarios_match_golden(qapp, name, side):
    assert run_legacy_widget(scenario_results(name, side), side) == GOLDEN[name]


def test_occluded_arm_and_missing_pose_match_golden(qapp):
    assert run_legacy_widget(occluded_arm_results(), "LEFT") == OCCLUDED_GOLDEN
