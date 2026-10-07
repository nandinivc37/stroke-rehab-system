from PyQt6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)


class ProgressPage(QWidget):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(45, 40, 45, 40)
        layout.setSpacing(20)

        title = QLabel("Progress")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Track rehabilitation performance across sessions"
        )
        subtitle.setObjectName("pageSubtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        placeholder = QLabel(
            "Progress history and session trends will appear here."
        )
        placeholder.setObjectName("heroText")

        layout.addWidget(placeholder)
        layout.addStretch()
