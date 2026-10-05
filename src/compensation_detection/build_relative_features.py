from pathlib import Path

import numpy as np
import pandas as pd

from src.compensation_detection.preprocess_toronto import (
    load_dataset,
)
from src.compensation_detection.normalize import (
    normalize_sequence,
)


INPUT_FILE = Path(
    "data/processed_toronto/compensation_features_temporal.csv"
)

OUTPUT_FILE = Path(
    "data/processed_toronto/compensation_relative_windows.npz"
)

WINDOW_SIZE = 20
EPSILON = 1e-6


# Toronto joint indices
SPINE_BASE = 0
SHOULDER_LEFT = 4
WRIST_LEFT = 6
SHOULDER_RIGHT = 8
WRIST_RIGHT = 10
SPINE_SHOULDER = 20


BASE_FEATURES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "trunk_vertical_angle",
    "shoulder_x_angle",
    "shoulder_height_difference",
    "trunk_forward_displacement",
    "trunk_length",
    "left_elbow_angle_delta",
    "right_elbow_angle_delta",
    "trunk_vertical_angle_delta",
    "shoulder_x_angle_delta",
    "shoulder_height_difference_delta",
    "trunk_forward_displacement_delta",
    "trunk_length_delta",
    "left_elbow_angle_rolling_std",
    "right_elbow_angle_rolling_std",
    "trunk_vertical_angle_rolling_std",
    "shoulder_x_angle_rolling_std",
    "shoulder_height_difference_rolling_std",
    "trunk_forward_displacement_rolling_std",
    "trunk_length_rolling_std",
]


RELATIVE_FEATURES = [
    "shoulder_vertical_displacement",
    "shoulder_to_wrist_vertical_ratio",
    "trunk_depth_displacement",
    "trunk_to_wrist_depth_ratio",
    "body_orientation_change",
]


def normalize_vector(vector):
    norm = np.linalg.norm(vector)

    if norm < EPSILON:
        return np.zeros(3, dtype=np.float32)

    return vector / norm


def build_reference_axes(frame):
    """
    Build a body-relative coordinate system from
    the first frame of a temporal window.

    vertical_axis:
        spine base -> spine shoulder

    lateral_axis:
        left shoulder -> right shoulder

    depth_axis:
        perpendicular to the lateral and vertical axes
    """

    vertical_vector = (
        frame[SPINE_SHOULDER]
        - frame[SPINE_BASE]
    )

    lateral_vector = (
        frame[SHOULDER_RIGHT]
        - frame[SHOULDER_LEFT]
    )

    vertical_axis = normalize_vector(
        vertical_vector
    )

    lateral_axis = normalize_vector(
        lateral_vector
    )

    depth_axis = normalize_vector(
        np.cross(
            lateral_axis,
            vertical_axis
        )
    )

    return (
        vertical_axis,
        lateral_axis,
        depth_axis,
    )


def project(vector, axis):
    return float(
        np.dot(vector, axis)
    )


def calculate_relative_features(
    positions,
    start,
    end,
    affected_side,
):
    """
    Calculate movement-level features for one
    20-frame temporal window.

    start = first frame index
    end   = exclusive end index
    """

    window = positions[start:end]

    first_frame = window[0]

    (
        vertical_axis,
        lateral_axis,
        depth_axis,
    ) = build_reference_axes(first_frame)

    # ---------------------------------------------------------
    # Select affected-side wrist
    # ---------------------------------------------------------

    if affected_side == "L":
        wrist_index = WRIST_LEFT
    else:
        wrist_index = WRIST_RIGHT

    # ---------------------------------------------------------
    # Shoulder center
    # ---------------------------------------------------------

    shoulder_center = (
        window[:, SHOULDER_LEFT]
        + window[:, SHOULDER_RIGHT]
    ) / 2.0

    # ---------------------------------------------------------
    # Shoulder movement
    # ---------------------------------------------------------

    shoulder_displacement = (
        shoulder_center[-1]
        - shoulder_center[0]
    )

    shoulder_vertical_displacement = abs(
        project(
            shoulder_displacement,
            vertical_axis,
        )
    )

    # ---------------------------------------------------------
    # Wrist movement
    # ---------------------------------------------------------

    wrist_displacement = (
        window[-1, wrist_index]
        - window[0, wrist_index]
    )

    wrist_vertical_displacement = abs(
        project(
            wrist_displacement,
            vertical_axis,
        )
    )

    shoulder_to_wrist_vertical_ratio = (
        shoulder_vertical_displacement
        / (
            wrist_vertical_displacement
            + EPSILON
        )
    )

    # ---------------------------------------------------------
    # Trunk movement
    # ---------------------------------------------------------

    trunk_displacement = (
        window[-1, SPINE_SHOULDER]
        - window[0, SPINE_SHOULDER]
    )

    trunk_depth_displacement = abs(
        project(
            trunk_displacement,
            depth_axis,
        )
    )

    wrist_depth_displacement = abs(
        project(
            wrist_displacement,
            depth_axis,
        )
    )

    trunk_to_wrist_depth_ratio = (
        trunk_depth_displacement
        / (
            wrist_depth_displacement
            + EPSILON
        )
    )

    # ---------------------------------------------------------
    # Body orientation change
    # ---------------------------------------------------------

    first_shoulder_vector = (
        window[0, SHOULDER_RIGHT]
        - window[0, SHOULDER_LEFT]
    )

    last_shoulder_vector = (
        window[-1, SHOULDER_RIGHT]
        - window[-1, SHOULDER_LEFT]
    )

    first_lateral = project(
        first_shoulder_vector,
        lateral_axis,
    )

    first_depth = project(
        first_shoulder_vector,
        depth_axis,
    )

    last_lateral = project(
        last_shoulder_vector,
        lateral_axis,
    )

    last_depth = project(
        last_shoulder_vector,
        depth_axis,
    )

    first_angle = np.arctan2(
        first_depth,
        first_lateral,
    )

    last_angle = np.arctan2(
        last_depth,
        last_lateral,
    )

    angle_difference = (
        last_angle - first_angle
    )

    # Wrap angle to [-pi, pi]
    angle_difference = (
        angle_difference + np.pi
    ) % (2 * np.pi) - np.pi

    body_orientation_change = abs(
        np.degrees(angle_difference)
    )

    return np.array(
        [
            shoulder_vertical_displacement,
            shoulder_to_wrist_vertical_ratio,
            trunk_depth_displacement,
            trunk_to_wrist_depth_ratio,
            body_orientation_change,
        ],
        dtype=np.float32,
    )


def get_affected_side(trial_name):
    """
    Determine affected side from the final
    token in the Toronto trial name.

    Examples:
        Rch_Fwr_Bck_L      -> L
        Rch_Fwr_Bck_L_1    -> L
        Rch_Fwr_Bck_R      -> R
        Rch_Sd2Sd_Bck_R_2  -> R
    """

    tokens = trial_name.split("_")

    for token in reversed(tokens):
        if token in {"L", "R"}:
            return token

    raise ValueError(
        f"Could not determine affected side "
        f"from trial name: {trial_name}"
    )


def main():

    print("=" * 60)
    print("TORONTO RELATIVE MOVEMENT WINDOW BUILDER")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load existing temporal feature dataset
    # ---------------------------------------------------------

    dataframe = pd.read_csv(INPUT_FILE)

    print(
        f"\nTemporal feature dataset: "
        f"{dataframe.shape}"
    )

    # ---------------------------------------------------------
    # Load original Toronto skeleton data
    # ---------------------------------------------------------

    dataset = load_dataset()

    trial_lookup = {}

    for trial in dataset:

        key = (
            trial["participant"],
            trial["trial"],
        )

        normalized_positions = normalize_sequence(
            trial["positions"]
        )

        trial_lookup[key] = {
            "positions": normalized_positions,
            "labels": trial["labels"],
        }

    print(
        f"Loaded skeleton sequences: "
        f"{len(trial_lookup)}"
    )

    # ---------------------------------------------------------
    # Build windows
    # ---------------------------------------------------------

    windows = []
    labels = []
    participants = []
    trials = []

    grouped = dataframe.groupby(
        ["participant", "trial"],
        sort=False,
    )

    for (participant, trial_name), group in grouped:

        group = group.sort_values("frame")

        key = (
            participant,
            trial_name,
        )

        if key not in trial_lookup:
            raise KeyError(
                f"Skeleton data not found for "
                f"{participant} / {trial_name}"
            )

        positions = trial_lookup[key]["positions"]
        frame_labels = group["label"].to_numpy(
            dtype=np.int64
        )

        # Safety check
        if len(positions) != len(group):
            raise ValueError(
                f"Frame count mismatch for "
                f"{participant} / {trial_name}: "
                f"skeleton={len(positions)}, "
                f"features={len(group)}"
            )

        affected_side = get_affected_side(
            trial_name
        )

        # Existing 21 features
        base_features = group[
            BASE_FEATURES
        ].to_numpy(
            dtype=np.float32
        )

        for start in range(
            0,
            len(group) - WINDOW_SIZE + 1,
        ):

            end = start + WINDOW_SIZE

            window_labels = frame_labels[
                start:end
            ]

            # Keep exactly the same rule as Exp 4:
            # all 20 frames must have one label.
            if not np.all(
                window_labels == window_labels[0]
            ):
                continue

            base_window = base_features[
                start:end
            ]

            relative = calculate_relative_features(
                positions,
                start,
                end,
                affected_side,
            )

            # Flatten 20 × 21 = 420 values
            flattened_base = (
                base_window.reshape(-1)
            )

            # 420 + 5 = 425 features
            combined = np.concatenate(
                [
                    flattened_base,
                    relative,
                ]
            )

            windows.append(combined)
            labels.append(
                window_labels[0]
            )
            participants.append(
                participant
            )
            trials.append(
                trial_name
            )

    X = np.asarray(
        windows,
        dtype=np.float32,
    )

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    participants = np.asarray(
        participants
    )

    trials = np.asarray(
        trials
    )

    # ---------------------------------------------------------
    # Verify
    # ---------------------------------------------------------

    print("\nGenerated windows:")
    print(f"Windows: {len(X)}")
    print(f"Feature shape: {X.shape}")

    print("\nExpected:")
    print("Windows: 52007")
    print("Features per window: 425")

    print("\nLabel distribution:")

    unique_labels, counts = np.unique(
        y,
        return_counts=True,
    )

    for label, count in zip(
        unique_labels,
        counts,
    ):
        print(
            f"Label {label}: {count}"
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=y,
        participants=participants,
        trials=trials,
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print("\n" + "=" * 60)
    print("RELATIVE MOVEMENT WINDOW BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
