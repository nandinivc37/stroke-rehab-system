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
