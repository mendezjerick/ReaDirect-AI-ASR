import subprocess
import sys


def test_check_training_environment_runs_without_crashing():
    result = subprocess.run([sys.executable, "scripts/check_training_environment.py"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "Python:" in result.stdout
    assert "Recommended RTX 3060" in result.stdout
