from pathlib import Path

import numpy as np
import pandas as pd

from src.compensation_detection.preprocess_toronto import (
    find_trials,
    load_trial,
)
from src.compensation_detection.normalize import normalize_sequence
from src.compensation_detection.smoothing import smooth_sequence
from src.compensation_detection.features import extract_frame_features


DATASET_ROOT = Path("data/toronto/data_new")

OUTPUT_DIR = Path("data/processed_toronto")
OUTPUT_FILE = OUTPUT_DIR / "compensation_features_shoulder.csv"


# Features for which frame-to-frame changes are useful.
TEMPORAL_FEATURES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "trunk_vertical_angle",
    "shoulder_x_angle",
    "shoulder_height_difference",
    "trunk_forward_displacement",
    "trunk_length",
]


def add_temporal_features(feature_rows):
    """
    Add frame-to-frame and short-term movement features.

    Temporal calculations are performed independently within
    each trial. No information crosses trial boundaries.
    """

    dataframe = pd.DataFrame(feature_rows)

    if dataframe.empty:
        return dataframe

    result = []

    for (_, _), trial_dataframe in dataframe.groupby(
        ["participant", "trial"],
        sort=False,
    ):
        trial_dataframe = trial_dataframe.copy()
        trial_dataframe = trial_dataframe.sort_values("frame")

        for feature in TEMPORAL_FEATURES:

            # Frame-to-frame change.
            trial_dataframe[f"{feature}_delta"] = (
                trial_dataframe[feature].diff().fillna(0.0)
            )

            # Short-term movement magnitude.
            trial_dataframe[f"{feature}_rolling_std"] = (
                trial_dataframe[feature]
                .rolling(window=5, min_periods=1)
                .std()
                .fillna(0.0)
            )

        result.append(trial_dataframe)

    return pd.concat(result, ignore_index=True)


def build_dataset():
    rows = []

    trials = find_trials(DATASET_ROOT)

    print(f"Found {len(trials)} trials.")

    for trial_number, trial_path in enumerate(
        trials,
        start=1,
    ):
        print(
            f"[{trial_number}/{len(trials)}] "
            f"Processing "
            f"{trial_path.parent.name}/{trial_path.name}"
        )

        try:
            trial = load_trial(trial_path)

            positions = trial["positions"]
            labels = trial["labels"]

            # -------------------------------------------------
            # 1. Normalize
            # -------------------------------------------------

            normalized = normalize_sequence(
                positions
            )

            # -------------------------------------------------
            # 2. Smooth
            # -------------------------------------------------

            smoothed = smooth_sequence(
                normalized
            )

            # -------------------------------------------------
            # 3. Extract static features
            # -------------------------------------------------

            for frame_index, frame in enumerate(
                smoothed
            ):
                features = extract_frame_features(
                    frame
                )

                row = {
                    "participant": trial["participant"],
                    "trial": trial["trial"],
                    "frame": frame_index,
                    "label": int(labels[frame_index]),
                }

                row.update(features)

                rows.append(row)

        except Exception as error:
            print(
                f"[ERROR] {trial_path}: {error}"
            )

    # ---------------------------------------------------------
    # 4. Add temporal features
    # ---------------------------------------------------------

    print("\nAdding temporal features...")

    dataframe = add_temporal_features(rows)

    # ---------------------------------------------------------
    # 5. Save
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 60)
    print("TEMPORAL DATASET CREATED")
    print("=" * 60)

    print(
        f"Rows       : {len(dataframe)}"
    )

    print(
        f"Columns    : {len(dataframe.columns)}"
    )

    print(
        f"Output     : {OUTPUT_FILE}"
    )

    print("\nLabel distribution:")

    print(
        dataframe["label"]
        .value_counts()
        .sort_index()
    )

    print("\nParticipants:")

    print(
        dataframe["participant"].nunique()
    )

    print("\nTrials:")

    print(
        dataframe[
            ["participant", "trial"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print("\nTemporal features:")

    temporal_columns = [
        column
        for column in dataframe.columns
        if "_delta" in column
        or "_rolling_std" in column
    ]

    print(
        "\n".join(temporal_columns)
    )

    return dataframe


if __name__ == "__main__":
    build_dataset()