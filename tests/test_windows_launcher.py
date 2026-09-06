from pathlib import Path


def test_windows_launcher_prepares_and_starts_application() -> None:
    launcher = Path("啟動網站.bat").read_text(encoding="utf-8")

    assert 'cd /d "%~dp0"' in launcher
    assert "-m venv .venv" in launcher
    assert "-m pip install -r requirements.txt" in launcher
    assert 'copy /Y ".env.example" ".env"' in launcher
    assert '".venv\\Scripts\\python.exe" src\\app.py' in launcher


def test_windows_launcher_reports_common_failures() -> None:
    launcher = Path("啟動網站.bat").read_text(encoding="utf-8")

    assert ":python_not_found" in launcher
    assert ":dependency_failed" in launcher
    assert ":app_failed" in launcher
    assert "exit /b 1" in launcher
