from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.database.db import get_dashboard_counts


class DashboardPage(QWidget):

    # Emitted when the user asks to start a rehabilitation session.
    start_session_requested = pyqtSignal()

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(45, 40, 45, 40)
        layout.setSpacing(20)

        title = QLabel("Dashboard")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Patient rehabilitation and movement assessment"
        )
        subtitle.setObjectName("pageSubtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(20)

        card = QFrame()
        card.setObjectName("heroCard")

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(35, 35, 35, 35)
        card_layout.setSpacing(15)

        welcome = QLabel("Welcome to Stroke Rehab")
        welcome.setObjectName("heroTitle")

        description = QLabel(
            "A markerless, camera-based rehabilitation "
            "platform for movement assessment and "
            "progress tracking."
        )
        description.setObjectName("heroText")
        description.setWordWrap(True)

        button = QPushButton("Start New Session")
        button.setObjectName("primaryButton")
        button.setMinimumWidth(180)
        button.clicked.connect(self.start_session_requested.emit)
        self.start_button = button

        card_layout.addWidget(welcome)
        card_layout.addWidget(description)
        card_layout.addSpacing(10)
        card_layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(card)

        # -------------------------------------------------
        # Summary counts from stored data
        # -------------------------------------------------

        stats_row = QHBoxLayout()
        stats_row.setSpacing(14)

        self.stat_labels = {}
        for key, title in (
            ("patients", "Patients"),
            ("sessions_last_7_days", "Sessions (last 7 days)"),
            ("pending_adaptations", "Changes awaiting approval"),
            ("last_session", "Last session"),
        ):
            stat_card = QFrame()
            stat_card.setObjectName("heroCard")
            stat_layout = QVBoxLayout(stat_card)
            stat_layout.setContentsMargins(18, 14, 18, 14)

            caption = QLabel(title)
            caption.setStyleSheet("color: #6b7280; font-size: 12px;")
            value = QLabel("—")
            value.setStyleSheet(
                "color: #172554; font-size: 20px; font-weight: 700;"
            )

            stat_layout.addWidget(caption)
            stat_layout.addWidget(value)
            stats_row.addWidget(stat_card)
            self.stat_labels[key] = value

        layout.addLayout(stats_row)
        layout.addStretch()

        self.refresh()

    def refresh(self):
        counts = get_dashboard_counts()
        for key, label in self.stat_labels.items():
            value = counts.get(key)
            label.setText("—" if value is None else str(value))
