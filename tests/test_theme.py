"""
The application theme does not depend on the desktop colour scheme.

A dark desktop palette (GNOME Yaru-dark values) must be fully replaced,
and text/background pairs must meet WCAG AA contrast (4.5:1).
"""

import pytest
from PyQt6.QtGui import QColor, QPalette

from src.ui.styles import APP_STYLE, apply_theme

R = QPalette.ColorRole
G = QPalette.ColorGroup


def luminance(colour):
    def channel(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return (0.2126 * channel(colour.red()) + 0.7152 * channel(colour.green())
            + 0.0722 * channel(colour.blue()))


def contrast(a, b):
    la, lb = sorted([luminance(QColor(a)), luminance(QColor(b))],
                    reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.fixture
def themed(qapp):
    saved = (qapp.palette(), qapp.styleSheet(), qapp.style().name())
    dark = QPalette()
    for role in (R.Window, R.Base, R.Button):
        dark.setColor(role, QColor("#2a2a2a"))
    for role in (R.WindowText, R.Text, R.ButtonText):
        dark.setColor(role, QColor("#ffffff"))
    qapp.setPalette(dark)
    apply_theme(qapp)
    yield qapp.palette()
    qapp.setPalette(saved[0])
    qapp.setStyleSheet(saved[1])
    qapp.setStyle(saved[2])


def test_dark_desktop_palette_is_replaced(themed):
    assert themed.color(R.Window).name() == "#f4f7fb"
    assert themed.color(R.Base).name() == "#ffffff"
    assert themed.color(R.Button).name() == "#ffffff"


@pytest.mark.parametrize("group, text, background", [
    (G.Active, R.Text, R.Base),
    (G.Active, R.WindowText, R.Window),
    (G.Active, R.ButtonText, R.Button),
    (G.Active, R.HighlightedText, R.Highlight),
    (G.Active, R.PlaceholderText, R.Base),
    (G.Disabled, R.ButtonText, R.Button),
    (G.Disabled, R.Text, R.Base),
])
def test_palette_pairs_meet_wcag_aa(themed, group, text, background):
    assert contrast(themed.color(group, text),
                    themed.color(group, background)) >= 4.5


@pytest.mark.parametrize("text, background", [
    ("#1f2937", "#ffffff"),   # secondary button, dropdown item
    ("#1e3a8a", "#eff6ff"),   # secondary hover
    ("#172554", "#dbeafe"),   # secondary pressed, dropdown selected/hover
    ("#4b5563", "#f3f4f6"),   # disabled button / input
    ("#ffffff", "#2563eb"),   # primary
    ("#ffffff", "#1d4ed8"),   # primary hover
    ("#ffffff", "#1e40af"),   # primary pressed
    ("#475569", "#e2e8f0"),   # primary disabled
    ("#6b7280", "#ffffff"),   # danger/reject disabled, disabled list item
])
def test_stylesheet_pairs_meet_wcag_aa(text, background):
    assert text in APP_STYLE or text == "#6b7280"
    assert contrast(text, background) >= 4.5


def test_every_control_state_is_styled():
    for selector in (
        "QPushButton:hover", "QPushButton:pressed", "QPushButton:disabled",
        "QPushButton#primaryButton:pressed",
        "QPushButton#primaryButton:disabled",
        "QComboBox QAbstractItemView",
        "QComboBox QAbstractItemView::item:selected",
        "QComboBox QAbstractItemView::item:disabled",
        "QComboBox:disabled", "QDialog QPushButton:disabled",
    ):
        assert selector in APP_STYLE, selector
