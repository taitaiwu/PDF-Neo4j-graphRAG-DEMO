from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


def test_start_script_skips_install_until_requirements_change(tmp_path: Path) -> None:
    project = tmp_path / "project"
    python = project / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    shutil.copy2(Path(__file__).parents[1] / "start.sh", project / "start.sh")
    (project / "requirements.txt").write_text("gradio==5.0\n", encoding="utf-8")
    (project / ".env.example").write_text("", encoding="utf-8")
    python.write_text(
        "#!/usr/bin/env bash\n"
        "case \"$*\" in\n"
        "  *\"-c\"*) sha256sum requirements.txt | cut -d' ' -f1 ;;\n"
        "  *\"-m pip install\"*) printf 'install\\n' >> install.log ;;\n"
        "  *\"src/app.py\"*) exit 0 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    python.chmod(0o755)

    def start() -> str:
        result = subprocess.run(
            ["bash", "start.sh"],
            cwd=project,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PATH": "/usr/bin:/bin"},
        )
        return result.stdout

    assert "安裝必要套件" in start()
    assert (project / "install.log").read_text(encoding="utf-8").splitlines() == ["install"]

    assert "略過安裝" in start()
    assert (project / "install.log").read_text(encoding="utf-8").splitlines() == ["install"]

    (project / "requirements.txt").write_text("gradio==5.1\n", encoding="utf-8")
    start()
    assert (project / "install.log").read_text(encoding="utf-8").splitlines() == [
        "install", "install",
    ]
