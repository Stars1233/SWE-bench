"""Gold patches come from the task repo when one is given.

A run against a task repo that graded the dataset's patch would report on code the
tree does not contain, so a tree whose gold.patch had drifted would still look
correct -- which is the one thing building from the tree is meant to prove.
"""

import json

import pytest
import yaml

from swebench.harness.utils import get_predictions_from_file


def _task(repo, instance_id, patch):
    d = repo / "tasks" / instance_id
    d.mkdir(parents=True)
    (d / "task.yaml").write_text(
        yaml.safe_dump(
            {
                "instance_id": instance_id,
                "repo": "x/y",
                "version": "1.0",
                "split": "test",
                "datasets": ["x/y-dataset"],
            }
        )
    )
    (d / "tests.json").write_text(
        json.dumps({"FAIL_TO_PASS": ["test_a"], "PASS_TO_PASS": []})
    )
    (d / "gold.patch").write_text(patch)
    (d / "test.patch").write_text("diff --git a/t b/t\n")
    (d / "problem_statement.md").write_text("an issue")
    (d / "eval.sh").write_text("#!/bin/bash\npytest\n")
    (d / "Dockerfile").write_text("FROM scratch")


def test_gold_patches_come_from_the_task_repo(tmp_path):
    (tmp_path / "sweb.yaml").write_text(
        yaml.safe_dump({"datasets": ["x/y-dataset"], "splits": ["test"]})
    )
    _task(tmp_path, "a__a-1", "diff --git a/from b/tree\n")

    preds = get_predictions_from_file("gold", "x/y-dataset", "test", str(tmp_path))
    assert [p["instance_id"] for p in preds] == ["a__a-1"]
    assert preds[0]["model_patch"] == "diff --git a/from b/tree\n"
    assert preds[0]["model_name_or_path"] == "gold"


def test_without_a_task_repo_the_dataset_is_used(monkeypatch):
    """Omitting --task-repo keeps the published dataset as the source."""
    called = {}

    def fake_load(dataset_name, split):
        called["args"] = (dataset_name, split)
        return [{"instance_id": "a__a-1", "patch": "diff --git a/from b/hub\n"}]

    monkeypatch.setattr("swebench.harness.utils.load_swebench_dataset", fake_load)
    preds = get_predictions_from_file("gold", "x/y-dataset", "test")
    assert called["args"] == ("x/y-dataset", "test")
    assert preds[0]["model_patch"] == "diff --git a/from b/hub\n"
