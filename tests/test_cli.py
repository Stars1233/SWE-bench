import subprocess
import sys


def test_smoke_test():
    cmd = [sys.executable, "-m", "swebench.harness.run_evaluation", "--help"]
    result = subprocess.run(cmd, capture_output=True)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0


def test_one_instance(tmp_path):
    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--predictions_path",
        "gold",
        "--max_workers",
        "1",
        "--instance_ids",
        "sympy__sympy-20590",
        "--run_id",
        "validate-gold",
        # keep the run report out of the checkout; without this it overwrites the
        # committed gold.validate-gold.json every time the suite runs
        "--report_dir",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0
