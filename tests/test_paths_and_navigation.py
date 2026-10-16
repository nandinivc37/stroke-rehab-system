"""WS1: working-directory independence and dashboard navigation."""

import subprocess
import sys

from src.paths import DB_PATH, LSTM_MODEL_PATH, PROJECT_ROOT


def test_paths_are_absolute_and_inside_project():
    assert PROJECT_ROOT.is_absolute()
    assert (PROJECT_ROOT / "src" / "paths.py").exists()
    for path in (DB_PATH, LSTM_MODEL_PATH):
        assert path.is_absolute()
        assert PROJECT_ROOT in path.parents


def test_database_path_independent_of_cwd(tmp_path):
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import src.database.db as db; print(db.DB_PATH)" % str(PROJECT_ROOT)
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == str(DB_PATH)
    assert not (tmp_path / "data").exists()


def test_dashboard_start_button_opens_rehab_page(qapp, temp_db):
    from src.ui.main_window import REHAB_PAGE_INDEX, MainWindow
    from src.ui.pages.rehab_page import RehabPage

    window = MainWindow()
    assert window.pages.currentIndex() == 0

    window.dashboard_page.start_button.click()

    assert window.pages.currentIndex() == REHAB_PAGE_INDEX
    assert isinstance(window.pages.currentWidget(), RehabPage)
