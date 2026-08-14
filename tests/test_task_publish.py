"""Publishing reads the tree, and refuses to publish a malformed one.

A task repo is the source of truth for a dataset, so anything wrong in the tree
becomes a broken instance on a leaderboard. Compiling and pushing therefore run
the checks first rather than trusting the caller to have run them.
"""

import pytest
import yaml

from swebench.task_checks import check_task_repo, errors, expected_image
from swebench.task_publish import CheckFailed, compile_splits, guard


def _repo(tmp_path, splits=("test",), **task_overrides):
    (tmp_path / "sweb.yaml").write_text(
        yaml.safe_dump({"dataset": "SWE-bench/Example", "splits": list(splits)})
    )
    return tmp_path


def _task(repo, instance_id, split="test", **overrides):
    d = repo / "tasks" / instance_id
    d.mkdir(parents=True)
    meta = {
        "instance_id": instance_id,
        "repo": "x/y",
        "version": "1.0",
        "base_commit": "abc123",
        "split": split,
        "image": expected_image(instance_id),
        "log_parser": "parse_log_pytest",
        "eval_type": "pass_and_fail",
        "FAIL_TO_PASS": ["test_a"],
        "PASS_TO_PASS": [],
        **overrides,
    }
    (d / "task.yaml").write_text(yaml.safe_dump(meta))
    (d / "problem_statement.md").write_text("an issue")
    (d / "gold.patch").write_text("diff --git a/x b/x\n")
    (d / "test.patch").write_text("diff --git a/t b/t\n")
    (d / "eval.sh").write_text(
        "#!/bin/bash\n: '>>>>> Start Test Output'\npytest\n: '>>>>> End Test Output'\n"
    )
    (d / "Dockerfile").write_text("FROM scratch")
    return d


def test_a_well_formed_repo_has_no_problems(tmp_path):
    _task(_repo(tmp_path), "a__a-1")
    assert check_task_repo(tmp_path) == []


def test_compile_groups_by_split_and_drops_the_split_key(tmp_path):
    repo = _repo(tmp_path, splits=("dev", "test"))
    _task(repo, "a__a-1", split="test")
    _task(repo, "b__b-2", split="dev")
    splits = compile_splits(tmp_path)
    assert sorted(splits) == ["dev", "test"]
    # `split` tells us which parquet to write, it is not a dataset column
    assert "split" not in splits["test"][0]
    assert splits["test"][0]["instance_id"] == "a__a-1"


def test_deprecated_tasks_are_not_compiled(tmp_path):
    repo = _repo(tmp_path)
    _task(repo, "a__a-1")
    _task(repo, "b__b-2", split="deprecated", FAIL_TO_PASS=[])
    splits = compile_splits(tmp_path)
    assert [r["instance_id"] for r in splits["test"]] == ["a__a-1"]


def test_publishing_a_malformed_repo_is_refused(tmp_path):
    repo = _repo(tmp_path)
    _task(repo, "a__a-1")
    (repo / "tasks" / "a__a-1" / "eval.sh").unlink()
    with pytest.raises(CheckFailed, match="eval.sh"):
        compile_splits(tmp_path)


def test_an_empty_fail_to_pass_blocks_publishing(tmp_path):
    """It would otherwise grade as fully resolved for every submission."""
    repo = _repo(tmp_path)
    _task(repo, "a__a-1", FAIL_TO_PASS=[])
    with pytest.raises(CheckFailed, match="nothing grades this task"):
        guard(tmp_path)


def test_an_unregistered_split_is_reported_and_fixable(tmp_path):
    repo = _repo(tmp_path, splits=("test",))
    _task(repo, "a__a-1", split="dev")
    problems = check_task_repo(tmp_path)
    assert any("not in sweb.yaml" in p.message for p in problems)

    check_task_repo(tmp_path, fix=True)
    assert yaml.safe_load((repo / "sweb.yaml").read_text())["splits"] == ["dev"]
    assert errors(check_task_repo(tmp_path)) == []


def test_a_wrong_image_name_is_reported_and_fixable(tmp_path):
    repo = _repo(tmp_path)
    _task(repo, "a__a-1", image="somewhere/else:latest")
    assert any(
        "is not the derived name" in p.message for p in check_task_repo(tmp_path)
    )

    check_task_repo(tmp_path, fix=True)
    meta = yaml.safe_load((repo / "tasks" / "a__a-1" / "task.yaml").read_text())
    assert meta["image"] == expected_image("a__a-1")


def test_image_names_follow_the_registry_convention():
    # docker hub disallows dunders, so instance ids are mapped
    assert expected_image("google__gson-2479") == (
        "swebench/sweb.eval.x86_64.google_1776_gson-2479:latest"
    )


def test_a_directory_without_task_json_is_reported(tmp_path):
    repo = _repo(tmp_path)
    _task(repo, "a__a-1")
    (repo / "tasks" / "leftovers").mkdir()
    assert any("not a task" in p.message for p in check_task_repo(tmp_path))
