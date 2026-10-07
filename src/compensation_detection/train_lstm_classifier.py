import json
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
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("data/processed_toronto/compensation_temporal_windows.npz")
OUTPUT_DIR = Path("outputs/compensation_model_lstm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

TRAIN_PARTICIPANTS = {
    "H03", "H04", "H05", "H07", "H08", "H09", "H10",
    "P01", "P03", "P04", "P05", "P06", "P07", "P08", "P09",
}

TEST_PARTICIPANTS = {
    "H01", "H02", "H06", "P02",
}

NUM_CLASSES = 4
INPUT_SIZE = 21
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.30

BATCH_SIZE = 128
MAX_EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 7


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("LSTM COMPENSATION CLASSIFIER")
print("=" * 70)
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

print()


# ============================================================
# Load exact existing dataset
# ============================================================

data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]
participants = data["participants"]
trials = data["trials"]

print("Dataset:")
print(f"  X shape:           {X.shape}")
print(f"  y shape:           {y.shape}")
print(f"  participants:      {participants.shape}")
print(f"  trials:            {trials.shape}")
print()

# Verify expected dimensions
assert X.shape[1:] == (20, 21), X.shape
assert len(X) == len(y) == len(participants) == len(trials)

# Verify labels
unique_labels = np.unique(y)
assert np.array_equal(unique_labels, np.array([1, 2, 3, 4]))

# Verify participant split
all_participants = set(participants)

assert TRAIN_PARTICIPANTS.isdisjoint(TEST_PARTICIPANTS)
assert TRAIN_PARTICIPANTS | TEST_PARTICIPANTS == all_participants

train_mask = np.isin(participants, list(TRAIN_PARTICIPANTS))
test_mask = np.isin(participants, list(TEST_PARTICIPANTS))

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("Participant-wise split:")
print(f"  Train participants: {sorted(TRAIN_PARTICIPANTS)}")
print(f"  Test participants:  {sorted(TEST_PARTICIPANTS)}")
print()
print(f"  Train windows: {len(X_train):,}")
print(f"  Test windows:  {len(X_test):,}")
print()


# ============================================================
# Convert labels from 1-4 to 0-3
# ============================================================

y_train_zero = y_train - 1
y_test_zero = y_test - 1


# ============================================================
# Class weights
# ============================================================

class_counts = np.bincount(y_train_zero, minlength=NUM_CLASSES)

class_weights = len(y_train_zero) / (
    NUM_CLASSES * class_counts
)

print("Training class counts:")
for i, count in enumerate(class_counts):
    print(f"  Class {i + 1}: {count:,}")

print()

print("Class weights:")
for i, weight in enumerate(class_weights):
    print(f"  Class {i + 1}: {weight:.4f}")

print()


# ============================================================
# PyTorch datasets
# ============================================================

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_zero, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_zero, dtype=torch.long)

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
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# BiLSTM model
# ============================================================

class BiLSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes,
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        # Last timestep
        last_output = output[:, -1, :]

        last_output = self.dropout(last_output)

        return self.classifier(last_output)


model = BiLSTMClassifier(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
).to(device)

print(model)
print()


# ============================================================
# Loss / optimizer / scheduler
# ============================================================

weight_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device,
)

criterion = nn.CrossEntropyLoss(
    weight=weight_tensor
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
)


# ============================================================
# Evaluation
# ============================================================

def evaluate():

    model.eval()

    all_predictions = []
    all_targets = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for batch_x, batch_y in test_loader:

            batch_x = batch_x.to(
                device,
                non_blocking=True,
            )

            batch_y = batch_y.to(
                device,
                non_blocking=True,
            )

            logits = model(batch_x)

            loss = criterion(
                logits,
                batch_y,
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            total_loss += (
                loss.item() * len(batch_y)
            )

            total_samples += len(batch_y)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                batch_y.cpu().numpy()
            )

    avg_loss = total_loss / total_samples

    accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        all_targets,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    return (
        avg_loss,
        accuracy,
        macro_f1,
        weighted_f1,
        np.array(all_targets),
        np.array(all_predictions),
    )


# ============================================================
# Training
# ============================================================

best_macro_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0

history = []

best_model_path = (
    OUTPUT_DIR / "compensation_lstm_model.pt"
)

print("=" * 70)
print("TRAINING")
print("=" * 70)

for epoch in range(1, MAX_EPOCHS + 1):

    model.train()

    total_train_loss = 0.0
    total_train_samples = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(
            device,
            non_blocking=True,
        )

        batch_y = batch_y.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(batch_x)

        loss = criterion(
            logits,
            batch_y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_train_loss += (
            loss.item() * len(batch_y)
        )

        total_train_samples += len(batch_y)

    train_loss = (
        total_train_loss /
        total_train_samples
    )

    (
        test_loss,
        accuracy,
        macro_f1,
        weighted_f1,
        _,
        _,
    ) = evaluate()

    scheduler.step(macro_f1)

    current_lr = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "learning_rate": current_lr,
    })

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Macro F1: {macro_f1:.4f} | "
        f"Weighted F1: {weighted_f1:.4f} | "
        f"LR: {current_lr:.2e}"
    )

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "num_classes": NUM_CLASSES,
                "best_macro_f1": best_macro_f1,
                "best_epoch": best_epoch,
            },
            best_model_path,
        )

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print()
        print(
            f"Early stopping at epoch {epoch}."
        )
        break


# ============================================================
# Load best model
# ============================================================

checkpoint = torch.load(
    best_model_path,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ============================================================
# Final evaluation
# ============================================================

(
    test_loss,
    accuracy,
    macro_f1,
    weighted_f1,
    targets,
    predictions,
) = evaluate()

report = classification_report(
    targets,
    predictions,
    labels=[0, 1, 2, 3],
    target_names=[
        "No Compensation",
        "Shoulder Elevation",
        "Trunk Rotation",
        "Lean Forward",
    ],
    digits=4,
    zero_division=0,
)

cm = confusion_matrix(
    targets,
    predictions,
    labels=[0, 1, 2, 3],
)


# ============================================================
# Save results
# ============================================================

metrics = {
    "model": "2-layer Bidirectional LSTM",
    "input_shape": [20, 21],
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "dropout": DROPOUT,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "best_epoch": int(best_epoch),
    "best_macro_f1": float(best_macro_f1),
    "final_accuracy": float(accuracy),
    "final_macro_f1": float(macro_f1),
    "final_weighted_f1": float(weighted_f1),
    "test_loss": float(test_loss),
    "device": str(device),
    "gpu": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None
    ),
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "train_windows": int(len(X_train)),
    "test_windows": int(len(X_test)),
    "train_participants": sorted(TRAIN_PARTICIPANTS),
    "test_participants": sorted(TEST_PARTICIPANTS),
}

with open(
    OUTPUT_DIR / "metrics.json",
    "w",
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
) as f:

    f.write(report)

np.savetxt(
    OUTPUT_DIR / "confusion_matrix.csv",
    cm,
    fmt="%d",
    delimiter=",",
)

np.save(
    OUTPUT_DIR / "confusion_matrix.npy",
    cm,
)

with open(
    OUTPUT_DIR / "training_history.json",
    "w",
) as f:

    json.dump(
        history,
        f,
        indent=2,
    )


# ============================================================
# Final output
# ============================================================

print()
print("=" * 70)
print("FINAL LSTM RESULTS")
print("=" * 70)

print(f"Best epoch:       {best_epoch}")
print(f"Accuracy:         {accuracy:.4f}")
print(f"Macro F1:         {macro_f1:.4f}")
print(f"Weighted F1:      {weighted_f1:.4f}")
print()

print(report)

print("Confusion matrix:")
print(cm)

print()
print(f"Model saved to: {best_model_path}")
print(f"Results saved to: {OUTPUT_DIR}")

print()
print("=" * 70)
print("COMPARISON WITH WINDOW RANDOM FOREST")
print("=" * 70)

RF_MACRO_F1 = 0.5354

print(f"RF Macro F1:      {RF_MACRO_F1:.4f}")
print(f"LSTM Macro F1:    {macro_f1:.4f}")
print(f"Difference:       {macro_f1 - RF_MACRO_F1:+.4f}")

if macro_f1 > RF_MACRO_F1:
    print()
    print("🏆 LSTM BEATS THE RANDOM FOREST.")
else:
    print()
    print("🏆 RANDOM FOREST REMAINS THE BEST MODEL.")
