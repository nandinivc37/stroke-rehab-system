from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)


INPUT_FILE = Path(
    "data/processed_toronto/compensation_temporal_windows.npz"
)

OUTPUT_DIR = Path(
    "outputs/compensation_model_window"
)

WINDOW_SIZE = 20
FEATURES_PER_FRAME = 21

LABEL_NAMES = [
    "No Compensation",
    "Shoulder Elevation",
    "Trunk Rotation",
    "Lean-Forward",
]


def main():

    print("=" * 60)
    print("TORONTO 20-FRAME WINDOW CLASSIFIER")
    print("=" * 60)

    data = np.load(INPUT_FILE)

    X = data["X"]
    y = data["y"]
    participants = data["participants"]
    trials = data["trials"]

    print(f"\nWindow dataset shape: {X.shape}")
    print(f"Labels shape        : {y.shape}")
    print(
        f"Participants        : "
        f"{len(np.unique(participants))}"
    )

    # ---------------------------------------------------------
    # Flatten each temporal window
    # ---------------------------------------------------------

    X_flat = X.reshape(
        X.shape[0],
        WINDOW_SIZE * FEATURES_PER_FRAME
    )

    print(
        f"\nFlattened feature shape: {X_flat.shape}"
    )

    # ---------------------------------------------------------
    # Same participant-wise split used previously
    # ---------------------------------------------------------

    train_participants = {
        "H03", "H04", "H05",
        "H07", "H08", "H09", "H10",
        "P01", "P03", "P04", "P05",
        "P06", "P07", "P08", "P09",
    }

    test_participants = {
        "H01", "H02", "H06", "P02",
    }

    train_mask = np.isin(
        participants,
        list(train_participants)
    )

    test_mask = np.isin(
        participants,
        list(test_participants)
    )

    X_train = X_flat[train_mask]
    X_test = X_flat[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    print("\nTrain participants:")
    print(sorted(np.unique(participants[train_mask])))

    print("\nTest participants:")
    print(sorted(np.unique(participants[test_mask])))

    print(
        f"\nTraining windows: {len(X_train)}"
    )

    print(
        f"Testing windows : {len(X_test)}"
    )

    # ---------------------------------------------------------
    # Train model
    # ---------------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    y_pred = model.predict(X_test)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("20-FRAME WINDOW MODEL PERFORMANCE")
    print("=" * 60)

    print(
        f"\nMacro F1    : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1 : {weighted_f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[1, 2, 3, 4],
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[1, 2, 3, 4]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Comp.",
            "Shoulder Elev.",
            "Trunk Rotation",
            "Lean-Forward",
        ],
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    display.plot(ax=ax)

    ax.set_title(
        "20-Frame Temporal Window Compensation Detection"
    )

    plt.tight_layout()

    confusion_path = (
        OUTPUT_DIR / "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300
    )

    plt.close()

    # Model
    model_path = (
        OUTPUT_DIR /
        "compensation_window_model.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    # Save metadata
    metadata_path = (
        OUTPUT_DIR /
        "model_metadata.txt"
    )

    with metadata_path.open("w") as file:

        file.write(
            "Toronto 20-Frame Temporal Window Model\n"
        )

        file.write(
            f"Window size: {WINDOW_SIZE}\n"
        )

        file.write(
            f"Features per frame: "
            f"{FEATURES_PER_FRAME}\n"
        )

        file.write(
            f"Flattened features: "
            f"{WINDOW_SIZE * FEATURES_PER_FRAME}\n"
        )

        file.write(
            f"Macro F1: {macro_f1:.4f}\n"
        )

        file.write(
            f"Weighted F1: {weighted_f1:.4f}\n"
        )

    print(
        f"\nConfusion matrix saved to: "
        f"{confusion_path}"
    )

    print(
        f"Model saved to: {model_path}"
    )

    print(
        f"Metadata saved to: {metadata_path}"
    )

    print("\n" + "=" * 60)
    print("20-FRAME WINDOW TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
