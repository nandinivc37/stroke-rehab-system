"""
Drive the live widget offscreen with scripted pose results.

This is the only place that knows how to construct the widget for the
characterization tests, so a refactor of the widget API only needs a
change here, never in the golden values.
"""

from src.simulation.synthetic_pose import FakeCamera, ScriptedPose
from src.exercises.legacy import LEGACY_SHOULDER_RAISE_2D
from src.ui.live_pose_widget import LivePoseWidget


def run_legacy_widget(results, side):
    widget = LivePoseWidget(
        exercise=LEGACY_SHOULDER_RAISE_2D,
        affected_side=side,
    )
    widget.camera = FakeCamera(len(results))
    widget.pose = ScriptedPose(results)

    for _ in results:
        widget.update_frame()

    stats = widget.get_stats()

    return dict(
        reps=stats["reps"],
        max=round(stats["max_angle"], 4),
        min=round(stats["min_angle"], 4),
        traj=len(stats["trajectory"]),
        sim=stats["trajectory_similarity"],
        segs=len(widget.analyzer.movement_segments),
        active=widget.analyzer.movement_active,
    )
