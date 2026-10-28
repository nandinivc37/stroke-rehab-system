import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.ui.pages.dashboard_page import DashboardPage
from src.ui.pages.patients_page import PatientsPage
from src.ui.pages.assessment_page import AssessmentPage
from src.ui.pages.rehab_page import RehabPage
from src.ui.pages.progress_page import ProgressPage


REHAB_PAGE_INDEX = 3


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Stroke Rehab System")
        self.setMinimumSize(1100, 700)

        self.build_ui()

    def build_ui(self):

        # -----------------------------------------
        # Central widget
        # -----------------------------------------

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # -----------------------------------------
        # Sidebar
        # -----------------------------------------

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(230)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(
            18, 28, 18, 20
        )
        sidebar_layout.setSpacing(8)

        title = QLabel("STROKE REHAB")
        title.setObjectName("appTitle")

        subtitle = QLabel("Rehabilitation System")
        subtitle.setObjectName("appSubtitle")

        sidebar_layout.addWidget(title)
        sidebar_layout.addWidget(subtitle)
        sidebar_layout.addSpacing(35)

        # -----------------------------------------
        # Navigation
        # -----------------------------------------

        self.nav_buttons = []

        nav_items = [
            "Dashboard",
            "Patients",
            "Assessment",
            "Rehabilitation",
            "Progress",
        ]

        for index, item in enumerate(nav_items):

            button = QPushButton(item)
            button.setObjectName("navButton")
            button.setCursor(
                Qt.CursorShape.PointingHandCursor
            )

            button.clicked.connect(
                lambda checked=False, i=index:
                self.show_page(i)
            )

            sidebar_layout.addWidget(button)
            self.nav_buttons.append(button)

        sidebar_layout.addStretch()

        therapist_label = QLabel(
            "Therapist Mode"
        )
        therapist_label.setStyleSheet(
            "color: #bfdbfe; font-size: 12px;"
        )

        sidebar_layout.addWidget(
            therapist_label
        )

        # -----------------------------------------
        # Page container
        # -----------------------------------------

        content = QFrame()
        content.setObjectName("content")

        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        self.pages = QStackedWidget()

        self.dashboard_page = DashboardPage()
        self.dashboard_page.start_session_requested.connect(
            lambda: self.show_page(REHAB_PAGE_INDEX)
        )

        self.pages.addWidget(
            self.dashboard_page
        )

        self.patients_page = PatientsPage()
        self.patients_page.patients_changed.connect(
            self.refresh_patient_lists
        )

        self.pages.addWidget(
            self.patients_page
        )

        self.pages.addWidget(
            AssessmentPage()
        )

        self.rehab_page = RehabPage()
        self.patients_page.patient_in_session = (
            lambda patient_id: self.rehab_page.live_widget is not None
            and self.rehab_page.current_patient_id == patient_id
        )

        self.pages.addWidget(
            self.rehab_page
        )

        self.pages.addWidget(
            ProgressPage()
        )

        content_layout.addWidget(self.pages)

        # -----------------------------------------
        # Add sidebar + content
        # -----------------------------------------

        main_layout.addWidget(sidebar)
        main_layout.addWidget(content)

        content.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        # Start on Dashboard
        self.show_page(0)

    # ---------------------------------------------
    # Page navigation
    # ---------------------------------------------

    def closeEvent(self, event):
        """Do not lose a running session when the window is closed."""
        try:
            self.rehab_page.save_interrupted_session()
        except Exception:
            logging.getLogger(__name__).exception(
                "Could not save the interrupted session"
            )
        super().closeEvent(event)

    def refresh_patient_lists(self):
        """Reload pages after a registration, without an app restart.
        Each page's refresh() keeps its current selection (and the
        session page does not reload mid-session)."""
        for i in range(self.pages.count()):
            page = self.pages.widget(i)
            if page is not self.patients_page and hasattr(page, "refresh"):
                page.refresh()

    def show_page(self, index):

        self.pages.setCurrentIndex(index)

        # Pages that show database content reload it when displayed.
        page = self.pages.widget(index)
        if hasattr(page, "refresh"):
            page.refresh()

        # Highlight active navigation button
        for i, button in enumerate(
            self.nav_buttons
        ):

            if i == index:
                button.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #2563eb;
                        color: white;
                        border: none;
                        text-align: left;
                        padding: 12px 18px;
                        font-size: 14px;
                        font-weight: bold;
                        border-radius: 8px;
                    }
                    """
                )

            else:
                button.setStyleSheet("")