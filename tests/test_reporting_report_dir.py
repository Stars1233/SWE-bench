"""`report_dir` must be honored by make_run_report (#498).

`main()` created the directory but never passed it through, so the final report
was always written relative to the CWD -- which silently overwrites committed
report files like gold.validate-gold.json when run from a checkout.
"""

from swebench.harness.reporting import make_run_report


def _fixture():
    predictions = {
        "x__x-1": {
            "instance_id": "x__x-1",
            "model_name_or_path": "gold",
            "model_patch": "diff",
        }
    }
    full_dataset = [{"instance_id": "x__x-1"}]
    return predictions, full_dataset


def test_report_written_into_report_dir(tmp_path):
    predictions, full_dataset = _fixture()
    out = make_run_report(predictions, full_dataset, "run-a", report_dir=str(tmp_path))
    assert out.parent == tmp_path
    assert out.name == "gold.run-a.json"
    assert out.exists()


def test_report_dir_is_created_if_absent(tmp_path):
    predictions, full_dataset = _fixture()
    nested = tmp_path / "deep" / "nested"
    out = make_run_report(predictions, full_dataset, "run-b", report_dir=str(nested))
    assert out.parent == nested and out.exists()


def test_defaults_to_cwd(tmp_path, monkeypatch):
    predictions, full_dataset = _fixture()
    monkeypatch.chdir(tmp_path)
    out = make_run_report(predictions, full_dataset, "run-c")
    assert out.exists() and out.resolve().parent == tmp_path.resolve()
