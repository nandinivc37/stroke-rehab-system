from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DashboardPage(QWidget):

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
        button.setFixedWidth(180)

        card_layout.addWidget(welcome)
        card_layout.addWidget(description)
        card_layout.addSpacing(10)
        card_layout.addWidget(button)

        layout.addWidget(card)
        layout.addStretch()
