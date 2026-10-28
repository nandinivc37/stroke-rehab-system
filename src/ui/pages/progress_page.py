import json
import logging

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.adaptation.workflow import (
    AdaptationError,
    decision_history,
    pending_proposals,
    review_proposal,
)
from src.database.db import get_all_patients, get_sessions
from src.exercises.catalogue import get_exercise_config
from src.progress.history import (
    MISSING,
    PEAK_FIELD,
    TABLE_HEADERS,
    fmt,
    group_sessions,
    peak_series,
    posture_text,
    series,
    table_row,
    target_series,
)

logger = logging.getLogger(__name__)

# Chart colours (validated reference palette, light mode): one data
# series plus a neutral dashed reference line for the prescribed target.
SERIES_COLOUR = "#2a78d6"
TARGET_COLOUR = "#52514e"
GRID_COLOUR = "#e5e7eb"
TEXT_COLOUR = "#374151"


def _card():
    card = QFrame()
    card.setStyleSheet("""
        QFrame {
            background-color: white;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
        }
        QLabel { border: none; }
    """)
    return card


def _section_title(text):
    label = QLabel(text)
    label.setStyleSheet(
        "font-size: 15px; font-weight: 700; color: #172554; border: none;"
    )
    return label


class ProgressPage(QWidget):

    def __init__(self):
        super().__init__()

        self.groups = {}
        self.group_keys = []
        self.current_sessions = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        body = QWidget()
        body.setObjectName("progressBody")
        body.setStyleSheet("QWidget#progressBody { background: #f4f7fb; }")
        scroll.setWidget(body)

        layout = QVBoxLayout(body)
        layout.setContentsMargins(45, 40, 45, 40)
        layout.setSpacing(16)

        title = QLabel("Progress")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Session history, trends and therapist review of proposed "
            "difficulty changes"
        )
        subtitle.setObjectName("pageSubtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        # -------------------------------------------------
        # Selectors
        # -------------------------------------------------

        selectors = QHBoxLayout()
        selectors.addWidget(QLabel("Patient:"))
        self.patient_combo = QComboBox()
        self.patient_combo.setMinimumWidth(220)
        self.patient_combo.currentIndexChanged.connect(self.patient_changed)
        selectors.addWidget(self.patient_combo)

        selectors.addSpacing(20)
        selectors.addWidget(QLabel("Exercise:"))
        self.group_combo = QComboBox()
        self.group_combo.setMinimumWidth(320)
        self.group_combo.currentIndexChanged.connect(self.group_changed)
        selectors.addWidget(self.group_combo)
        selectors.addStretch()

        layout.addLayout(selectors)

        # -------------------------------------------------
        # Pending adaptation proposals
        # -------------------------------------------------

        proposal_card = _card()
        proposal_layout = QVBoxLayout(proposal_card)
        proposal_layout.setContentsMargins(18, 14, 18, 14)

        proposal_layout.addWidget(
            _section_title("Proposed changes awaiting therapist review")
        )

        self.proposal_label = QLabel()
        self.proposal_label.setWordWrap(True)
        proposal_layout.addWidget(self.proposal_label)

        review_row = QHBoxLayout()
        review_row.addWidget(QLabel("Reviewer:"))
        self.reviewer_input = QLineEdit()
        self.reviewer_input.setPlaceholderText("Therapist name")
        self.reviewer_input.setMaximumWidth(220)
        review_row.addWidget(self.reviewer_input)
        self.review_note_input = QLineEdit()
        self.review_note_input.setPlaceholderText("Note (optional)")
        review_row.addWidget(self.review_note_input)

        self.reject_button = QPushButton("Reject")
        self.reject_button.setStyleSheet("""
            QPushButton {
                background: white; color: #b91c1c;
                border: 1px solid #fca5a5; border-radius: 8px;
                padding: 9px 18px; font-weight: 600;
            }
            QPushButton:disabled { color: #6b7280; border-color: #d1d5db; }
        """)
        self.reject_button.clicked.connect(
            lambda: self.review_current(False)
        )
        self.approve_button = QPushButton("Approve for next session")
        self.approve_button.setObjectName("primaryButton")
        self.approve_button.clicked.connect(
            lambda: self.review_current(True)
        )
        review_row.addWidget(self.reject_button)
        review_row.addWidget(self.approve_button)
        proposal_layout.addLayout(review_row)

        disclaimer = QLabel(
            "Rule-based decision support with engineering default "
            "thresholds; not clinically validated. Changes apply only "
            "after approval."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet("color: #6b7280; font-size: 12px;")
        proposal_layout.addWidget(disclaimer)

        layout.addWidget(proposal_card)

        # -------------------------------------------------
        # Trend charts
        # -------------------------------------------------

        chart_card = _card()
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(18, 14, 18, 14)
        chart_layout.addWidget(_section_title("Trends"))

        self.chart_note = QLabel()
        self.chart_note.setWordWrap(True)
        self.chart_note.setStyleSheet("color: #6b7280; font-size: 12px;")
        chart_layout.addWidget(self.chart_note)

        self.figure = Figure(figsize=(8, 6.2), dpi=100)
        self.figure.patch.set_facecolor("white")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(520)
        chart_layout.addWidget(self.canvas)

        layout.addWidget(chart_card)

        # -------------------------------------------------
        # Session table
        # -------------------------------------------------

        table_card = _card()
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(18, 14, 18, 14)
        table_layout.addWidget(_section_title("Sessions"))

        self.table = QTableWidget(0, len(TABLE_HEADERS))
        self.table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        table_header = self.table.horizontalHeader()
        table_header.setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        table_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(220)
        self.table.itemSelectionChanged.connect(self.session_selected)
        table_layout.addWidget(self.table)

        self.session_detail = QLabel(
            f"Select a session for details. {MISSING} = not recorded."
        )
        self.session_detail.setWordWrap(True)
        self.session_detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        table_layout.addWidget(self.session_detail)

        layout.addWidget(table_card)

        # -------------------------------------------------
        # Decision history
        # -------------------------------------------------

        decision_card = _card()
        decision_layout = QVBoxLayout(decision_card)
        decision_layout.setContentsMargins(18, 14, 18, 14)
        decision_layout.addWidget(_section_title("Adaptation decisions"))

        self.decision_table = QTableWidget(0, 5)
        self.decision_table.setHorizontalHeaderLabels(
            ["Date", "Exercise", "Decision", "Status", "Rationale"]
        )
        self.decision_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        header = self.decision_table.horizontalHeader()
        for column in range(4):
            header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.decision_table.setWordWrap(True)
        self.decision_table.setMinimumHeight(180)
        decision_layout.addWidget(self.decision_table)

        layout.addWidget(decision_card)

        self.current_proposal = None
        self.refresh()

    # -----------------------------------------------------
    # Loading
    # -----------------------------------------------------

    def refresh(self):
        current = self.patient_combo.currentData()

        self.patient_combo.blockSignals(True)
        self.patient_combo.clear()
        for patient in get_all_patients():
            self.patient_combo.addItem(f"{patient[0]} - {patient[1]}",
                                       patient[0])
        self.patient_combo.blockSignals(False)

        if current is not None:
            index = self.patient_combo.findData(current)
            if index >= 0:
                self.patient_combo.setCurrentIndex(index)

        self.patient_changed()

    def patient_changed(self, *_):
        patient_id = self.patient_combo.currentData()

        self.groups = (
            group_sessions(get_sessions(patient_id))
            if patient_id is not None else {}
        )
        self.group_keys = list(self.groups)

        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        for key in self.group_keys:
            count = len(self.groups[key])
            self.group_combo.addItem(f"{key.label()} — {count} session(s)")
        self.group_combo.blockSignals(False)

        self.load_proposals(patient_id)
        self.load_decisions(patient_id)
        self.group_changed()

    def group_changed(self, *_):
        index = self.group_combo.currentIndex()

        if 0 <= index < len(self.group_keys):
            key = self.group_keys[index]
            self.current_sessions = self.groups[key]
        else:
            key = None
            self.current_sessions = []

        self.fill_table(self.current_sessions)
        self.draw_charts(key, self.current_sessions)

    # -----------------------------------------------------
    # Proposals
    # -----------------------------------------------------

    def load_proposals(self, patient_id):
        proposals = pending_proposals(patient_id) if patient_id else []
        self.current_proposal = proposals[0] if proposals else None

        has = self.current_proposal is not None
        self.approve_button.setEnabled(has)
        self.reject_button.setEnabled(has)

        if not has:
            self.proposal_label.setText("No pending proposals.")
            return

        proposal = self.current_proposal
        config = get_exercise_config(proposal["exercise_id"])
        before = json.loads(proposal["previous_parameters_json"])
        after = json.loads(proposal["parameters_json"])

        changes = []
        for parameter in config.adaptive_parameters:
            old, new = before.get(parameter.name), after.get(parameter.name)
            marker = "  ← changed" if old != new else ""
            changes.append(
                f"{parameter.label}: {old:g} → {new:g} {parameter.unit}"
                f"{marker}"
            )

        self.proposal_label.setText(
            f"<b>{proposal['decision']}</b> for {proposal['exercise']} "
            f"({proposal['affected_side']})<br>"
            + "<br>".join(changes)
            + f"<br><span style='color:#374151'>{proposal['reason']}</span>"
        )

    def review_current(self, approve):
        if self.current_proposal is None:
            return

        try:
            status = review_proposal(
                self.current_proposal["recommendation_id"],
                approve,
                self.reviewer_input.text(),
                self.review_note_input.text().strip(),
            )
        except AdaptationError as error:
            QMessageBox.warning(self, "Cannot review proposal", str(error))
            return

        self.review_note_input.clear()
        QMessageBox.information(
            self,
            "Proposal reviewed",
            f"Proposal {status.lower()}."
            + (" It will be used in the next session." if approve else ""),
        )
        self.patient_changed()

    # -----------------------------------------------------
    # Decisions
    # -----------------------------------------------------

    def load_decisions(self, patient_id):
        rows = decision_history(patient_id) if patient_id else []
        self.decision_table.setRowCount(len(rows))

        for r, row in enumerate(rows):
            config = get_exercise_config(row["exercise_id"])
            values = [
                row["created_at"] or MISSING,
                f"{config.display_name if config else row['exercise_id']} "
                f"({row['affected_side']})",
                (f"{row['decision']} ({row['action']})"
                 if row.get("action") else row["decision"]),
                row["proposal_status"] or "no change proposed",
                row["rationale"],
            ]
            for c, value in enumerate(values):
                self.decision_table.setItem(r, c, QTableWidgetItem(value))

        self.decision_table.resizeRowsToContents()

    # -----------------------------------------------------
    # Session table
    # -----------------------------------------------------

    def fill_table(self, sessions):
        self.table.setRowCount(len(sessions))

        # Newest first in the table; charts read oldest to newest. Row
        # headers show the session number used on the chart x-axis.
        self.table.setVerticalHeaderLabels(
            [str(n) for n in range(len(sessions), 0, -1)]
        )

        for r, session in enumerate(reversed(sessions)):
            for c, value in enumerate(table_row(session)):
                item = QTableWidgetItem(value)
                if c > 1:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(r, c, item)

        self.session_detail.setText(
            f"Select a session for details. {MISSING} = not recorded."
        )

    def session_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return

        session = list(reversed(self.current_sessions))[rows[0].row()]
        lines = []

        if session.get("metrics_version") is None:
            lines.append(
                "Recorded by the original implementation: only reps and "
                "angle range were stored; the angle definition differs "
                "from current measurements."
            )
        else:
            parameters = json.loads(session["parameters_json"] or "{}")
            lines.append(
                "Prescribed: " + ", ".join(
                    f"{k} = {v:g}" for k, v in sorted(parameters.items())
                )
            )
            lines.append(
                f"Off-plane repetitions (not counted): "
                f"{fmt(session.get('off_plane_reps'), '{:d}')}; "
                f"frame errors: {fmt(session.get('frame_errors'), '{:d}')}."
            )
            posture = posture_text(session)
            if posture:
                lines.append(posture)

        self.session_detail.setText("\n".join(lines))

    # -----------------------------------------------------
    # Charts
    # -----------------------------------------------------

    def draw_charts(self, key, sessions):
        self.figure.clear()

        if key is None or not sessions:
            self.chart_note.setText("No sessions recorded for this patient.")
            self.canvas.draw_idle()
            return

        config = get_exercise_config(key.exercise)
        unit = "°" if key.legacy or config is None else config.angle_unit

        panels = [
            ("Repetitions", "reps", "target_reps", ""),
            ("Peak range", PEAK_FIELD, "target_range", unit),
        ]
        if not key.legacy:
            panels.append(
                ("Movement score (DTW shape + range, 0-100)",
                 "trajectory_similarity", None, "")
            )

        if key.legacy:
            self.chart_note.setText(
                "Legacy sessions: recorded before metrics v1 with a "
                "different angle definition; shown separately and not "
                "comparable with current measurements."
            )
        else:
            self.chart_note.setText(
                "Solid line: measured. Dashed line: prescribed target. "
                f"Missing values are omitted, not drawn as zero. "
                f"{len(sessions)} session(s), oldest to newest."
            )

        axes = self.figure.subplots(len(panels), 1, sharex=True)
        if len(panels) == 1:
            axes = [axes]

        for ax, (label, field, target_name, suffix) in zip(axes, panels):
            xs, ys = series(sessions, field) if field != PEAK_FIELD                 else peak_series(sessions)
            ax.plot(xs, ys, color=SERIES_COLOUR, linewidth=1.6,
                    marker="o", markersize=6, label="Measured")

            if target_name and not key.legacy:
                tx, ty = target_series(sessions, target_name)
                if tx:
                    ax.step(tx, ty, where="mid", color=TARGET_COLOUR,
                            linewidth=1.2, linestyle=(0, (4, 3)),
                            label="Target")

            if xs:
                ax.annotate(
                    f"{ys[-1]:.0f}{suffix}", (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=TEXT_COLOUR,
                )

            if field == "trajectory_similarity":
                # Fixed scale: auto-zooming exaggerates small differences.
                ax.set_ylim(0, 105)
            elif field == "reps":
                ax.set_ylim(bottom=0)

            ax.set_title(label + (f" ({suffix})" if suffix else ""),
                         fontsize=10, color=TEXT_COLOUR, loc="left")
            ax.grid(axis="y", color=GRID_COLOUR, linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(GRID_COLOUR)
            ax.tick_params(colors=TEXT_COLOUR, labelsize=8)

            handles, _ = ax.get_legend_handles_labels()
            if len(handles) > 1:
                ax.legend(fontsize=8, frameon=False, loc="upper left")

        axes[-1].set_xlabel("Session number", fontsize=9, color=TEXT_COLOUR)
        axes[-1].xaxis.get_major_locator().set_params(integer=True)

        self.figure.tight_layout()
        self.canvas.draw_idle()
