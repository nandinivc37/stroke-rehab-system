from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
import joblib


INPUT_FILE = Path("data/processed_toronto/compensation_features.csv")
OUTPUT_DIR = Path("outputs/compensation_model")

FEATURE_COLUMNS = [
    "left_elbow_angle",
    "right_elbow_angle",
    "trunk_vertical_angle",
    "shoulder_x_angle",
    "shoulder_height_difference",
    "trunk_forward_displacement",
    "trunk_length",
]

LABEL_COLUMN = "label"
GROUP_COLUMN = "participant"


def main():
    print("=" * 60)
    print("TORONTO COMPENSATION CLASSIFIER")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load processed dataset
    # ---------------------------------------------------------
    dataframe = pd.read_csv(INPUT_FILE)

    print(f"\nDataset shape: {dataframe.shape}")
    print(f"Participants: {dataframe[GROUP_COLUMN].nunique()}")

    X = dataframe[FEATURE_COLUMNS]
    y = dataframe[LABEL_COLUMN]
    groups = dataframe[GROUP_COLUMN]

    # ---------------------------------------------------------
    # 2. Subject-wise train/test split
    # ---------------------------------------------------------
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    train_participants = sorted(
        dataframe.iloc[train_indices][GROUP_COLUMN].unique()
    )

    test_participants = sorted(
        dataframe.iloc[test_indices][GROUP_COLUMN].unique()
    )

    print("\nTrain participants:")
    print(train_participants)

    print("\nTest participants:")
    print(test_participants)

    print(f"\nTraining frames: {len(X_train)}")
    print(f"Testing frames : {len(X_test)}")

    # ---------------------------------------------------------
    # 3. Train Random Forest
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
    # 4. Predictions
    # ---------------------------------------------------------
    y_pred = model.predict(X_test)

    # ---------------------------------------------------------
    # 5. Evaluation
    # ---------------------------------------------------------
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
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(f"\nMacro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[1, 2, 3, 4],
            target_names=[
                "No Compensation",
                "Shoulder Elevation",
                "Trunk Rotation",
                "Lean-Forward",
            ],
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # 6. Save output directory
    # ---------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 7. Save confusion matrix
    # ---------------------------------------------------------
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[1, 2, 3, 4],
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

    fig, ax = plt.subplots(figsize=(8, 6))
    display.plot(ax=ax)
    ax.set_title("Toronto Compensation Detection - Confusion Matrix")
    plt.tight_layout()

    confusion_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 8. Save feature importance
    # ---------------------------------------------------------
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance_path = OUTPUT_DIR / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    print(f"\nFeature importance saved to: {importance_path}")
    print("\nFeature importance:")
    print(importance.to_string(index=False))

    # ---------------------------------------------------------
    # 9. Save trained model
    # ---------------------------------------------------------
    model_path = OUTPUT_DIR / "compensation_model.joblib"

    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()