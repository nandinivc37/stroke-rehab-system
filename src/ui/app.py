import sys
from pathlib import Path

# Allow `python src/ui/app.py` from any directory, in addition to
# `python -m src.ui.app` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.ui.main_window import MainWindow  # noqa: E402
from src.ui.styles import apply_theme  # noqa: E402


def main():

    app = QApplication(sys.argv)

    app.setApplicationName(
        "Stroke Rehab System"
    )

    apply_theme(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
