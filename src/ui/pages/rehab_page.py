from PyQt6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)


class RehabPage(QWidget):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(45, 40, 45, 40)
        layout.setSpacing(20)

        title = QLabel("Rehabilitation")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Live movement analysis and exercise monitoring"
        )
        subtitle.setObjectName("pageSubtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        placeholder = QLabel(
            "Live rehabilitation session will appear here."
        )
        placeholder.setObjectName("heroText")

        layout.addWidget(placeholder)
        layout.addStretch()
