from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

APP_STYLE = """
QMainWindow {
    background-color: #f4f7fb;
}

QWidget {
    font-family: "DejaVu Sans";
    color: #1f2937;
}

QFrame#sidebar {
    background-color: #172554;
}

QLabel#appTitle {
    color: white;
    font-size: 22px;
    font-weight: bold;
}

QLabel#appSubtitle {
    color: #bfdbfe;
    font-size: 12px;
}

QPushButton#navButton {
    background-color: transparent;
    color: #dbeafe;
    border: none;
    text-align: left;
    padding: 12px 18px;
    font-size: 14px;
    font-weight: normal;
    border-radius: 8px;
}

QPushButton#navButton:hover {
    background-color: #1e3a8a;
}

QPushButton#navButton:pressed {
    background-color: #2563eb;
}

QFrame#content {
    background-color: #f4f7fb;
}

QLabel#pageTitle {
    font-size: 28px;
    font-weight: bold;
    color: #111827;
}

QLabel#pageSubtitle {
    font-size: 14px;
    color: #6b7280;
}

QFrame#heroCard {
    background-color: white;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
}

QLabel#heroTitle {
    font-size: 30px;
    font-weight: bold;
    color: #172554;
}

QLabel#heroText {
    font-size: 15px;
    color: #6b7280;
}

QPushButton#primaryButton {
    background-color: #2563eb;
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: bold;
}

QPushButton#primaryButton:hover {
    background-color: #1d4ed8;
}

QPushButton#primaryButton:pressed {
    background-color: #1e40af;
}

QPushButton#primaryButton:disabled {
    background-color: #e2e8f0;
    color: #475569;
}

/* Default (secondary) buttons: without these rules they take the
   system palette, which is dark under a dark desktop theme. */
QPushButton {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #9ca3af;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #eff6ff;
    color: #1e3a8a;
    border: 1px solid #2563eb;
}

QPushButton:pressed {
    background-color: #dbeafe;
    color: #172554;
}

QPushButton:disabled {
    background-color: #f3f4f6;
    color: #4b5563;
    border: 1px solid #d1d5db;
}

QTableWidget {
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    gridline-color: #e5e7eb;
    font-size: 14px;
    selection-background-color: #dbeafe;
    selection-color: #172554;
}

QHeaderView::section {
    background-color: #eff6ff;
    color: #172554;
    padding: 10px;
    border: none;
    border-bottom: 1px solid #dbe7f5;
    font-weight: bold;
}

QLineEdit,
QSpinBox,
QComboBox,
QTextEdit {
    background-color: white;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    padding: 7px;
}

QLineEdit:focus,
QSpinBox:focus,
QComboBox:focus,
QTextEdit:focus {
    border: 1px solid #2563eb;
}

QLineEdit,
QSpinBox,
QDoubleSpinBox,
QComboBox,
QTextEdit {
    color: #1f2937;
}

QLineEdit:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled,
QComboBox:disabled,
QTextEdit:disabled {
    background-color: #f3f4f6;
    color: #4b5563;
}

/* Expanded dropdown list (a separate popup window). */
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #d1d5db;
    selection-background-color: #dbeafe;
    selection-color: #172554;
    outline: none;
}

QComboBox QAbstractItemView::item {
    min-height: 24px;
    padding: 2px 6px;
}

QComboBox QAbstractItemView::item:hover,
QComboBox QAbstractItemView::item:selected {
    background-color: #dbeafe;
    color: #172554;
}

QComboBox QAbstractItemView::item:disabled {
    color: #6b7280;
}

QToolTip {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #d1d5db;
}
QDialog {
    background-color: #ffffff;
}

QDialog QLabel {
    color: #374151;
    font-size: 13px;
}

QDialog QLabel#pageTitle {
    color: #172554;
    font-size: 28px;
    font-weight: bold;
}

QDialog QFormLayout QLabel {
    color: #374151;
    font-size: 13px;
}

QDialog QLineEdit,
QDialog QSpinBox,
QDialog QComboBox,
QDialog QTextEdit {
    color: #1f2937;
    background-color: #ffffff;
    border: 1px solid #d1d5db;
}

QDialog QComboBox {
    padding: 7px;
}

QDialog QPushButton {
    color: #374151;
    background-color: #e5e7eb;
    border: 1px solid #d1d5db;
    padding: 9px 18px;
    border-radius: 6px;
}

QDialog QPushButton:hover {
    background-color: #dbeafe;
}

QDialog QPushButton:disabled {
    background-color: #f3f4f6;
    color: #4b5563;
}

QDialog QPushButton#primaryButton {
    color: white;
    background-color: #2563eb;
    border: none;
    font-weight: bold;
}

QDialog QPushButton#primaryButton:hover {
    background-color: #1d4ed8;
}

QHeaderView {
    background-color: #eff6ff;
}

QTableCornerButton::section {
    background-color: #eff6ff;
    border: none;
}

"""


# The application has one (light) design. Without an explicit palette,
# Qt follows the desktop colour scheme, and under a dark desktop theme
# every control the stylesheet does not fully style (buttons, dropdown
# lists, dialogs, scroll bars) gets a dark background behind the
# stylesheet's dark text.
LIGHT_PALETTE = {
    QPalette.ColorRole.Window: "#f4f7fb",
    QPalette.ColorRole.WindowText: "#1f2937",
    QPalette.ColorRole.Base: "#ffffff",
    QPalette.ColorRole.AlternateBase: "#f9fafb",
    QPalette.ColorRole.Text: "#1f2937",
    QPalette.ColorRole.Button: "#ffffff",
    QPalette.ColorRole.ButtonText: "#1f2937",
    QPalette.ColorRole.BrightText: "#ffffff",
    QPalette.ColorRole.Highlight: "#dbeafe",
    QPalette.ColorRole.HighlightedText: "#172554",
    QPalette.ColorRole.PlaceholderText: "#6b7280",
    QPalette.ColorRole.ToolTipBase: "#ffffff",
    QPalette.ColorRole.ToolTipText: "#1f2937",
    QPalette.ColorRole.Link: "#1d4ed8",
}

DISABLED_TEXT = "#4b5563"


def light_palette():
    palette = QPalette()
    for role, colour in LIGHT_PALETTE.items():
        palette.setColor(role, QColor(colour))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role,
                         QColor(DISABLED_TEXT))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base,
                     QColor("#f3f4f6"))
    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.Button, QColor("#f3f4f6"))
    return palette


def apply_theme(app: QApplication):
    """Fusion style + light palette + stylesheet, whatever the desktop."""
    app.setStyle("Fusion")
    app.setPalette(light_palette())
    app.setStyleSheet(APP_STYLE)
