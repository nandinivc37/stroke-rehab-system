from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


MODEL_PATH = Path(
    "outputs/compensation_model_lstm/compensation_lstm_model.pt"
)


CLASS_NAMES = {
    0: "No Compensation",
    1: "Shoulder Elevation",
    2: "Trunk Rotation",
    3: "Lean Forward",
}


class BiLSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size=21,
        hidden_size=64,
        num_layers=2,
        num_classes=4,
        dropout=0.3,
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

        last_output = output[:, -1, :]

        last_output = self.dropout(
            last_output
        )

        return self.classifier(last_output)


class CompensationLSTM:

    def __init__(self, model_path=MODEL_PATH):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )

        self.model = BiLSTMClassifier(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            num_classes=checkpoint["num_classes"],
            dropout=checkpoint["dropout"],
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)

        self.model.eval()

    def predict(self, window):

        """
        Predict compensation from a
        20-frame × 21-feature window.

        Args:
            window: numpy array of shape
                    (20, 21)

        Returns:
            class_name
            confidence
            class_index
        """

        window = np.asarray(
            window,
            dtype=np.float32,
        )

        if window.shape != (20, 21):

            raise ValueError(
                f"Expected window shape "
                f"(20, 21), got {window.shape}"
            )

        tensor = torch.from_numpy(
            window
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            logits = self.model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1,
            )

        class_index = int(
            prediction.item()
        )

        confidence = float(
            confidence.item()
        )

        class_name = CLASS_NAMES[
            class_index
        ]

        return (
            class_name,
            confidence,
            class_index,
        )

    def predict_proba(self, window):

        """
        Return probabilities for all four
        compensation classes.
        """

        window = np.asarray(
            window,
            dtype=np.float32,
        )

        if window.shape != (20, 21):

            raise ValueError(
                f"Expected window shape "
                f"(20, 21), got {window.shape}"
            )

        tensor = torch.from_numpy(
            window
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            logits = self.model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

        return probabilities[
            0
        ].cpu().numpy()
