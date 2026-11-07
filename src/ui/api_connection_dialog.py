"""Small login dialog for connecting the desktop app to a NARS API."""

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from src.api.client import APIClientError, NARSAPIClient


class APIConnectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect to NARS API")
        self.setMinimumWidth(420)
        self.api_client = None

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.url_input = QLineEdit("http://127.0.0.1:8000")
        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)

        form.addRow("API URL:", self.url_input)
        form.addRow("Username:", self.username_input)
        form.addRow("Password:", self.password_input)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Connect")
        buttons.accepted.connect(self._connect)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _connect(self):
        client = None
        try:
            client = NARSAPIClient(self.url_input.text().strip())
            user = client.login(
                self.username_input.text().strip(),
                self.password_input.text(),
            )
            if user.get("role") not in {"therapist", "admin"}:
                raise APIClientError(
                    "Only therapist and administrator accounts can start rehabilitation sessions."
                )
        except (ValueError, APIClientError) as exc:
            if client is not None:
                client.close()
            QMessageBox.warning(self, "API Connection Failed", str(exc))
            return

        self.api_client = client
        self.password_input.clear()
        self.accept()

    def reject(self):
        if self.api_client is not None:
            self.api_client.close()
            self.api_client = None
        super().reject()
