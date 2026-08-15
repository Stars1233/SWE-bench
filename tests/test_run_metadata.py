"""A run records what it graded against (#report).

Re-grading needs the expected tests and the log parser, which live in the dataset
rather than in the run's logs. Without a record, `swebench report` has to be told
the dataset again, and grades against the wrong one silently if told wrongly.
"""

import json

import pytest

from swebench.harness import run_evaluation


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(run_evaluation, "RUN_EVALUATION_LOG_DIR", tmp_path)
    return tmp_path


def test_metadata_round_trips(run_dir):
    run_evaluation.write_run_metadata(
        "my-run", "SWE-bench/SWE-bench_Verified", "test", None
    )
    got = run_evaluation.read_run_metadata("my-run")
    assert got["dataset"] == "SWE-bench/SWE-bench_Verified"
    assert got["split"] == "test"
    assert got["created_at"]


def test_the_task_repo_is_recorded_when_one_was_used(run_dir):
    run_evaluation.write_run_metadata("my-run", "org/ds", "dev", "/path/to/tasks")
    assert run_evaluation.read_run_metadata("my-run")["task_repo"] == "/path/to/tasks"


def test_a_run_without_metadata_reads_as_none(run_dir):
    (run_dir / "old-run").mkdir()
    assert run_evaluation.read_run_metadata("old-run") is None


def test_metadata_is_written_beside_the_instance_logs(run_dir):
    path = run_evaluation.write_run_metadata("my-run", "org/ds", "test", None)
    assert path == run_dir / "my-run" / "run.json"
    assert json.loads(path.read_text())["dataset"] == "org/ds"
