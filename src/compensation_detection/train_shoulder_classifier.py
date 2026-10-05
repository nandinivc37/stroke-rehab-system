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


INPUT_FILE = Path(
    "data/processed_toronto/compensation_features_shoulder.csv"
)

OUTPUT_DIR = Path(
    "outputs/compensation_model_shoulder"
)


STATIC_FEATURES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "trunk_vertical_angle",
    "shoulder_x_angle",
    "shoulder_height_difference",
    "trunk_forward_displacement",
    "trunk_length",
]


SHOULDER_FEATURES = [
    "left_shoulder_vertical_offset",
    "right_shoulder_vertical_offset",
    "shoulder_center_vertical_offset",
    "left_shoulder_spine_distance",
    "right_shoulder_spine_distance",
    "left_upper_arm_angle",
    "right_upper_arm_angle",
]


TEMPORAL_FEATURES = [
    f"{feature}_delta"
    for feature in STATIC_FEATURES
] + [
    f"{feature}_rolling_std"
    for feature in STATIC_FEATURES
]


FEATURE_COLUMNS = (
    STATIC_FEATURES
    + SHOULDER_FEATURES
    + TEMPORAL_FEATURES
)


LABEL_COLUMN = "label"
GROUP_COLUMN = "participant"


def main():

    print("=" * 60)
    print("TORONTO SHOULDER-AWARE COMPENSATION CLASSIFIER")
    print("=" * 60)

    dataframe = pd.read_csv(INPUT_FILE)

    print(f"\nDataset shape: {dataframe.shape}")
    print(
        f"Participants: "
        f"{dataframe[GROUP_COLUMN].nunique()}"
    )

    X = dataframe[FEATURE_COLUMNS]
    y = dataframe[LABEL_COLUMN]
    groups = dataframe[GROUP_COLUMN]

    # ---------------------------------------------------------
    # Subject-wise split
    # ---------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_indices, test_indices = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    print("\nTrain participants:")
    print(
        sorted(
            dataframe.iloc[train_indices][GROUP_COLUMN]
            .unique()
        )
    )

    print("\nTest participants:")
    print(
        sorted(
            dataframe.iloc[test_indices][GROUP_COLUMN]
            .unique()
        )
    )

    print(f"\nTraining frames: {len(X_train)}")
    print(f"Testing frames : {len(X_test)}")

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------

    y_pred = model.predict(X_test)

    # ---------------------------------------------------------
    # Metrics
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
    print("SHOULDER-AWARE MODEL PERFORMANCE")
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
    # Output directory
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Confusion matrix
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

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    display.plot(ax=ax)

    ax.set_title(
        "Shoulder-Aware Compensation Detection"
    )

    plt.tight_layout()

    confusion_path = (
        OUTPUT_DIR /
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300,
    )

    plt.close()

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance":
                model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance_path = (
        OUTPUT_DIR /
        "feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    print(
        f"\nFeature importance saved to: "
        f"{importance_path}"
    )

    print("\nTop 20 features:")

    print(
        importance.head(20)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    model_path = (
        OUTPUT_DIR /
        "compensation_shoulder_model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nModel saved to: {model_path}"
    )

    print("\n" + "=" * 60)
    print("SHOULDER-AWARE TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
