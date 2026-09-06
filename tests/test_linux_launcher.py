import os
from pathlib import Path


LAUNCHER_PATH = Path("啟動網站.sh")


def test_linux_launcher_prepares_and_starts_application() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert launcher.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in launcher
    assert 'python3 -m venv .venv' in launcher
    assert '".venv/bin/python" -m pip install -r requirements.txt' in launcher
    assert 'cp ".env.example" ".env"' in launcher
    assert 'exec ".venv/bin/python" src/app.py' in launcher


def test_linux_launcher_is_executable() -> None:
    assert os.access(LAUNCHER_PATH, os.X_OK)


def test_linux_launcher_reports_common_failures() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "command -v python3" in launcher
    assert "套件安裝失敗" in launcher
    assert "exit 1" in launcher
