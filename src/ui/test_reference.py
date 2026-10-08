from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout

from src.ui.reference_skeleton import ReferenceSkeletonWidget


app = QApplication([])

window = QWidget()
window.setWindowTitle("Reference Exercise")

layout = QVBoxLayout(window)

skeleton = ReferenceSkeletonWidget(
    exercise_name="Shoulder Flexion",
    affected_side="LEFT",
)

layout.addWidget(skeleton)

window.resize(500, 600)
window.show()

app.exec()