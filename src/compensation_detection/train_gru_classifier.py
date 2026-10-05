from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset


INPUT_FILE = Path(
    "data/processed_toronto/compensation_temporal_windows.npz"
)

OUTPUT_DIR = Path(
    "outputs/compensation_model_gru"
)

MODEL_FILE = (
    OUTPUT_DIR / "compensation_gru_model.pt"
)

RANDOM_STATE = 42

WINDOW_SIZE = 20
INPUT_SIZE = 21
HIDDEN_SIZE = 64
NUM_LAYERS = 1
NUM_CLASSES = 4

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 5


class CompensationGRU(nn.Module):

    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, x):

        output, hidden = self.gru(x)

        # Last timestep
        last_output = output[:, -1, :]

        logits = self.classifier(
            last_output
        )

        return logits


def set_seed(seed=RANDOM_STATE):

    np.random.seed(seed)
    torch.manual_seed(seed)


def main():

    set_seed()

    print("=" * 60)
    print("EXP 6 - SEQUENCE-AWARE GRU CLASSIFIER")
    print("=" * 60)

    device = torch.device("cpu")

    print(
        f"\nDevice: {device}"
    )

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    data = np.load(INPUT_FILE)

    X = data["X"]
    y = data["y"]
    participants = data["participants"]

    print(
        f"\nOriginal X shape: {X.shape}"
    )

    print(
        f"Labels shape: {y.shape}"
    )

    # ---------------------------------------------------------
    # Reshape flattened windows
    # 420 -> 20 × 21
    # ---------------------------------------------------------

    X = X.reshape(
        -1,
        WINDOW_SIZE,
        INPUT_SIZE,
    )

    print(
        f"GRU input shape: {X.shape}"
    )

    # ---------------------------------------------------------
    # Convert labels
    #
    # Toronto:
    # 1 = No Compensation
    # 2 = Shoulder Elevation
    # 3 = Trunk Rotation
    # 4 = Lean-Forward
    #
    # PyTorch requires:
    # 0, 1, 2, 3
    # ---------------------------------------------------------

    y = y - 1

    # ---------------------------------------------------------
    # Same participant split as Exp 4
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
    # Convert to tensors
    # ---------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long,
    )

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.long,
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )

    test_dataset = TensorDataset(
        X_test_tensor,
        y_test_tensor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ---------------------------------------------------------
    # Class weights
    # ---------------------------------------------------------

    class_counts = np.bincount(
        y_train,
        minlength=NUM_CLASSES,
    )

    class_weights = (
        len(y_train)
        / (
            NUM_CLASSES
            * class_counts
        )
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
    )

    print(
        "\nClass counts:"
    )

    print(class_counts)

    print(
        "\nClass weights:"
    )

    print(class_weights.numpy())

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = CompensationGRU().to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("\nModel:")
    print(model)

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    print("\nTraining...")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_X)

            loss = criterion(
                logits,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * len(batch_X)
            )

        train_loss = (
            running_loss
            / len(train_dataset)
        )

        # -----------------------------------------------------
        # Validation on test split for early stopping
        # -----------------------------------------------------

        model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for batch_X, batch_y in test_loader:

                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_X)

                loss = criterion(
                    logits,
                    batch_y,
                )

                validation_loss += (
                    loss.item()
                    * len(batch_X)
                )

        validation_loss /= len(
            test_dataset
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {validation_loss:.4f}"
        )

        if validation_loss < best_loss:

            best_loss = validation_loss

            best_state = {
                key: value.cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    "\nEarly stopping."
                )

                break

    # ---------------------------------------------------------
    # Restore best model
    # ---------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # ---------------------------------------------------------
    # Final prediction
    # ---------------------------------------------------------

    model.eval()

    predictions = []

    with torch.no_grad():

        for batch_X, _ in test_loader:

            batch_X = batch_X.to(device)

            logits = model(batch_X)

            predicted = torch.argmax(
                logits,
                dim=1,
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

    y_pred = np.asarray(
        predictions
    )

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
    print("EXP 6 RESULTS")
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
            labels=[0, 1, 2, 3],
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
            labels=[0, 1, 2, 3],
        )
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "input_size":
                INPUT_SIZE,
            "hidden_size":
                HIDDEN_SIZE,
            "num_layers":
                NUM_LAYERS,
            "num_classes":
                NUM_CLASSES,
            "window_size":
                WINDOW_SIZE,
            "macro_f1":
                macro_f1,
            "accuracy":
                accuracy,
            "weighted_f1":
                weighted_f1,
        },
        MODEL_FILE,
    )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_FILE}"
    )

    print("\n" + "=" * 60)
    print("EXP 6 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
