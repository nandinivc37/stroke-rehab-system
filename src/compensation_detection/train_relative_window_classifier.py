from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit


INPUT_FILE = Path(
    "data/processed_toronto/compensation_relative_windows.npz"
)

OUTPUT_DIR = Path(
    "outputs/compensation_model_relative_window"
)

MODEL_FILE = (
    OUTPUT_DIR
    / "compensation_relative_window_model.joblib"
)

RANDOM_STATE = 42


def main():

    print("=" * 60)
    print("EXP 5 - RELATIVE WINDOW CLASSIFIER")
    print("=" * 60)

    data = np.load(INPUT_FILE)

    X = data["X"]
    y = data["y"]
    participants = data["participants"]

    print(f"\nDataset shape: {X.shape}")
    print(f"Labels shape:  {y.shape}")

    # ---------------------------------------------------------
    # Same participant split as Experiments 1-4
    # ---------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups=participants,
        )
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    train_participants = sorted(
        set(participants[train_idx])
    )

    test_participants = sorted(
        set(participants[test_idx])
    )

    print("\nTrain participants:")
    print(", ".join(train_participants))

    print("\nTest participants:")
    print(", ".join(test_participants))

    print(
        f"\nTrain windows: {len(X_train)}"
    )

    print(
        f"Test windows:  {len(X_test)}"
    )

    # ---------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    y_pred = model.predict(X_test)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
    )

    print("\n" + "=" * 60)
    print("EXP 5 RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy:     {accuracy:.4f}"
    )

    print(
        f"Macro F1:     {macro_f1:.4f}"
    )

    print(
        f"Weighted F1:  {weighted_f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Compensation",
                "Shoulder Elevation",
                "Trunk Rotation",
                "Lean-Forward",
            ],
            digits=4,
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred,
        )
    )

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    importances = model.feature_importances_

    print("\nTop 15 Feature Importances:")

    top_indices = np.argsort(
        importances
    )[::-1][:15]

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):

        if index < 420:
            feature_name = (
                f"temporal_feature_{index}"
            )
        else:
            relative_names = [
                "shoulder_vertical_displacement",
                "shoulder_to_wrist_vertical_ratio",
                "trunk_depth_displacement",
                "trunk_to_wrist_depth_ratio",
                "body_orientation_change",
            ]

            feature_name = relative_names[
                index - 420
            ]

        print(
            f"{rank:2d}. "
            f"{feature_name:<45} "
            f"{importances[index]:.6f}"
        )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_FILE}"
    )

    print("\n" + "=" * 60)
    print("EXP 5 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
