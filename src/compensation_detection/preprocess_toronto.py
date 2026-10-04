from pathlib import Path
import csv
import numpy as np


# ---------------------------------------------------------
# Toronto dataset configuration
# ---------------------------------------------------------

DATASET_ROOT = Path("data/toronto/data_new")

JOINTS_PER_FRAME = 25
COORDINATES_PER_JOINT = 3


# ---------------------------------------------------------
# CSV loading
# ---------------------------------------------------------

def load_joint_positions(csv_path):
    """
    Load Toronto Joint_Positions.csv.

    The dataset stores one x,y,z coordinate triplet per row.
    25 rows correspond to one frame.

    Returns:
        numpy array of shape (frames, 25, 3)
    """

    rows = []

    with csv_path.open("r", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            if not row:
                continue

            values = [float(value) for value in row]

            if len(values) != COORDINATES_PER_JOINT:
                raise ValueError(
                    f"Expected {COORDINATES_PER_JOINT} values per row "
                    f"in {csv_path}, got {len(values)}"
                )

            rows.append(values)

    positions = np.asarray(rows, dtype=np.float32)

    if len(positions) % JOINTS_PER_FRAME != 0:
        raise ValueError(
            f"{csv_path} contains {len(positions)} joint rows. "
            f"This is not divisible by {JOINTS_PER_FRAME}."
        )

    frame_count = len(positions) // JOINTS_PER_FRAME

    positions = positions.reshape(
        frame_count,
        JOINTS_PER_FRAME,
        COORDINATES_PER_JOINT
    )

    return positions


def load_labels(csv_path):
    """
    Load one label per frame from Labels.csv.

    Returns:
        numpy array of shape (frames,)
    """

    labels = []

    with csv_path.open("r", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            if not row:
                continue

            labels.append(int(row[0]))

    return np.asarray(labels, dtype=np.int64)


# ---------------------------------------------------------
# Trial loading
# ---------------------------------------------------------

def load_trial(trial_path):
    """
    Load and validate one Toronto trial.

    Returns:
        dictionary containing:
            participant
            trial
            positions
            labels
    """

    joint_file = trial_path / "Joint_Positions.csv"
    label_file = trial_path / "Labels.csv"

    if not joint_file.exists() or not label_file.exists():
        return None

    positions = load_joint_positions(joint_file)
    labels = load_labels(label_file)

    if len(positions) != len(labels):
        raise ValueError(
            f"Frame/label mismatch in {trial_path}:\n"
            f"  frames = {len(positions)}\n"
            f"  labels = {len(labels)}"
        )

    return {
        "participant": trial_path.parent.name,
        "trial": trial_path.name,
        "positions": positions,
        "labels": labels,
    }


# ---------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------

def find_trials(dataset_root=DATASET_ROOT):
    """
    Find all trial directories containing both required CSV files.
    """

    trials = []

    for participant_path in sorted(dataset_root.iterdir()):

        if not participant_path.is_dir():
            continue

        for trial_path in sorted(participant_path.iterdir()):

            if not trial_path.is_dir():
                continue

            joint_file = trial_path / "Joint_Positions.csv"
            label_file = trial_path / "Labels.csv"

            if joint_file.exists() and label_file.exists():
                trials.append(trial_path)

    return trials


def load_dataset(dataset_root=DATASET_ROOT):
    """
    Load all valid Toronto trials.

    Returns:
        list of trial dictionaries
    """

    dataset = []

    trials = find_trials(dataset_root)

    print(f"Found {len(trials)} trials.")

    for trial_path in trials:

        try:
            trial = load_trial(trial_path)

            if trial is not None:
                dataset.append(trial)

        except Exception as error:
            print(f"[ERROR] {trial_path}: {error}")

    print(f"Successfully loaded {len(dataset)} trials.")

    return dataset


# ---------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------

def print_summary(dataset):
    """
    Print a compact summary of the loaded dataset.
    """

    total_frames = sum(
        len(trial["labels"])
        for trial in dataset
    )

    participants = sorted(
        set(trial["participant"] for trial in dataset)
    )

    print("\n" + "=" * 50)
    print("TORONTO DATASET SUMMARY")
    print("=" * 50)

    print(f"Participants : {len(participants)}")
    print(f"Trials       : {len(dataset)}")
    print(f"Frames       : {total_frames}")

    print("\nParticipants:")
    print(", ".join(participants))

    print("\nExample trial:")

    if dataset:
        example = dataset[0]

        print(f"Participant : {example['participant']}")
        print(f"Trial       : {example['trial']}")
        print(f"Positions   : {example['positions'].shape}")
        print(f"Labels      : {example['labels'].shape}")
        print(
            f"Label values: "
            f"{np.unique(example['labels']).tolist()}"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    dataset = load_dataset()

    print_summary(dataset)