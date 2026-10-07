import sys

from PyQt6.QtWidgets import QApplication

from src.ui.main_window import MainWindow
from src.ui.styles import APP_STYLE


def main():

    app = QApplication(sys.argv)

    app.setApplicationName(
        "Stroke Rehab System"
    )

    app.setStyleSheet(APP_STYLE)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
