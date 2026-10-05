from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/processed_toronto/compensation_features_temporal.csv"
)

OUTPUT_FILE = Path(
    "data/processed_toronto/compensation_temporal_windows.npz"
)

WINDOW_SIZE = 20

FEATURE_COLUMNS = [
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

LABEL_COLUMN = "label"


def build_windows(dataframe):
    windows = []
    labels = []
    participants = []
    trials = []

    grouped = dataframe.groupby(
        ["participant", "trial"],
        sort=False
    )

    for (participant, trial), group in grouped:

        group = group.sort_values("frame")

        features = group[FEATURE_COLUMNS].to_numpy(
            dtype=np.float32
        )

        frame_labels = group[LABEL_COLUMN].to_numpy(
            dtype=np.int64
        )

        for start in range(
            0,
            len(group) - WINDOW_SIZE + 1
        ):
            end = start + WINDOW_SIZE

            window_features = features[start:end]
            window_labels = frame_labels[start:end]

            # Keep only windows where every frame
            # has the same compensation label.
            if not np.all(window_labels == window_labels[0]):
                continue

            windows.append(window_features)
            labels.append(window_labels[0])
            participants.append(participant)
            trials.append(trial)

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(participants),
        np.asarray(trials),
    )


def main():

    print("=" * 60)
    print("TORONTO TEMPORAL WINDOW BUILDER")
    print("=" * 60)

    dataframe = pd.read_csv(INPUT_FILE)

    print(f"\nInput dataset shape: {dataframe.shape}")
    print(
        f"Participants: "
        f"{dataframe['participant'].nunique()}"
    )
    print(
        f"Trials: "
        f"{dataframe['trial'].nunique()}"
    )

    windows, labels, participants, trials = build_windows(
        dataframe
    )

    print("\nWindow configuration:")
    print(f"Window size: {WINDOW_SIZE} frames")
    print(f"Features per frame: {len(FEATURE_COLUMNS)}")

    print("\nGenerated windows:")
    print(f"Windows: {len(windows)}")

    if len(windows) > 0:
        print(
            f"Window shape: {windows.shape}"
        )

    print("\nLabel distribution:")

    unique_labels, counts = np.unique(
        labels,
        return_counts=True
    )

    for label, count in zip(unique_labels, counts):
        print(
            f"Label {label}: {count}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_FILE,
        X=windows,
        y=labels,
        participants=participants,
        trials=trials,
    )

    print(
        f"\nSaved windows to: {OUTPUT_FILE}"
    )

    print("\n" + "=" * 60)
    print("TEMPORAL WINDOW BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
