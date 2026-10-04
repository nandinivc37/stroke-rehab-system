from pathlib import Path

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
OUTPUT_FILE = OUTPUT_DIR / "compensation_features.csv"


def build_dataset():
    rows = []

    trials = find_trials(DATASET_ROOT)

    print(f"Found {len(trials)} trials.")

    for trial_number, trial_path in enumerate(trials, start=1):

        print(
            f"[{trial_number}/{len(trials)}] "
            f"Processing {trial_path.parent.name}/{trial_path.name}"
        )

        try:
            trial = load_trial(trial_path)

            positions = trial["positions"]
            labels = trial["labels"]

            # Normalize
            normalized = normalize_sequence(positions)

            # Smooth
            smoothed = smooth_sequence(normalized)

            # Extract frame-level features
            for frame_index, frame in enumerate(smoothed):

                features = extract_frame_features(frame)

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

    dataframe = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("DATASET CREATED")
    print("=" * 60)

    print(f"Rows       : {len(dataframe)}")
    print(f"Columns    : {len(dataframe.columns)}")
    print(f"Output     : {OUTPUT_FILE}")

    print("\nLabel distribution:")
    print(dataframe["label"].value_counts().sort_index())

    print("\nParticipants:")
    print(dataframe["participant"].nunique())

    print("\nTrials:")
    print(dataframe[["participant", "trial"]].drop_duplicates().shape[0])

    return dataframe


if __name__ == "__main__":
    build_dataset()