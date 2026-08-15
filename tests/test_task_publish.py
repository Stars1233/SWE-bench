"""Publishing reads the tree, and refuses to publish a malformed one.

A task repo is the source of truth for a dataset, so anything wrong in the tree
becomes a broken instance on a leaderboard. Compiling and pushing therefore run
the checks first rather than trusting the caller to have run them.
"""

import json

import pytest
import yaml

from swebench.task.checks import check_task_repo, errors, expected_image
from swebench.task.publish import CheckFailed, compile_datasets, guard


def _repo(tmp_path, splits=("test",), **task_overrides):
    (tmp_path / "sweb.yaml").write_text(
        yaml.safe_dump({"datasets": ["SWE-bench/Example"], "splits": list(splits)})
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
        "datasets": ["SWE-bench/Example"],
        "FAIL_TO_PASS": ["test_a"],
        "PASS_TO_PASS": [],
        **overrides,
    }
    tests = {k: meta.pop(k) for k in ("FAIL_TO_PASS", "PASS_TO_PASS") if k in meta}
    (d / "tests.json").write_text(
        json.dumps(
            {
                "FAIL_TO_PASS": tests.get("FAIL_TO_PASS", ["test_a"]),
                "PASS_TO_PASS": tests.get("PASS_TO_PASS", []),
            }
        )
    )
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
    compiled = compile_datasets(tmp_path)["SWE-bench/Example"]
    assert sorted(compiled) == ["dev", "test"]
    # `split` and `datasets` route a task; they are not dataset columns
    assert "split" not in compiled["test"][0]
    assert "datasets" not in compiled["test"][0]
    assert compiled["test"][0]["instance_id"] == "a__a-1"


def test_deprecated_tasks_are_not_compiled(tmp_path):
    repo = _repo(tmp_path)
    _task(repo, "a__a-1")
    _task(repo, "b__b-2", split="deprecated", FAIL_TO_PASS=[])
    compiled = compile_datasets(tmp_path)["SWE-bench/Example"]
    assert [r["instance_id"] for r in compiled["test"]] == ["a__a-1"]


def test_publishing_a_malformed_repo_is_refused(tmp_path):
    repo = _repo(tmp_path)
    _task(repo, "a__a-1")
    (repo / "tasks" / "a__a-1" / "eval.sh").unlink()
    with pytest.raises(CheckFailed, match="eval.sh"):
        compile_datasets(tmp_path)


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


def test_one_tree_can_publish_several_datasets(tmp_path):
    """Verified and Lite are subsets of the same instances, not copies of them."""
    (tmp_path / "sweb.yaml").write_text(
        yaml.safe_dump({"datasets": ["org/base", "org/subset"], "splits": ["test"]})
    )
    _task(tmp_path, "a__a-1", datasets=["org/base", "org/subset"])
    _task(tmp_path, "b__b-2", datasets=["org/base"])

    compiled = compile_datasets(tmp_path)
    assert [r["instance_id"] for r in compiled["org/base"]["test"]] == ["a__a-1", "b__b-2"]
    assert [r["instance_id"] for r in compiled["org/subset"]["test"]] == ["a__a-1"]


def test_a_task_claiming_an_unpublished_dataset_is_reported(tmp_path):
    repo = _repo(tmp_path)
    _task(repo, "a__a-1", datasets=["SWE-bench/Example", "org/nope"])
    assert any("not published by" in p.message for p in check_task_repo(tmp_path))


def test_selecting_one_dataset_narrows_the_build(tmp_path):
    (tmp_path / "sweb.yaml").write_text(
        yaml.safe_dump({"datasets": ["org/base", "org/subset"], "splits": ["test"]})
    )
    _task(tmp_path, "a__a-1", datasets=["org/base", "org/subset"])
    _task(tmp_path, "b__b-2", datasets=["org/base"])

    only = compile_datasets(tmp_path, ["org/subset"])
    assert list(only) == ["org/subset"]
    assert [r["instance_id"] for r in only["org/subset"]["test"]] == ["a__a-1"]
