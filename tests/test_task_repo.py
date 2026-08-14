"""A task repo is the source of truth for its dataset.

The tree has to round trip exactly: a patch that loses its CRLF no longer applies,
and an instance id that silently disappears shrinks a run without anyone noticing.
"""

import pytest
import yaml

from swebench.task_repo import (
    asset_path,
    load_config,
    load_dockerfiles,
    load_task,
    load_task_repo,
    published_tasks,
    task_dirs,
)

CRLF_PATCH = "diff --git a/x b/x\r\n--- a/x\r\n+++ b/x\r\n+one\r\n"


def _config(repo, dataset="x/y-dataset", splits=("test",)):
    (repo / "sweb.yaml").write_text(
        yaml.safe_dump({"dataset": dataset, "splits": list(splits)})
    )


def _task(repo, instance_id, **overrides):
    d = repo / "tasks" / instance_id
    (d / "assets").mkdir(parents=True)
    meta = {
        "instance_id": instance_id,
        "repo": "x/y",
        "version": "1.0",
        "split": "test",
        **overrides,
    }
    (d / "task.yaml").write_text(yaml.safe_dump(meta))
    (d / "gold.patch").write_text("gold")
    (d / "problem_statement.md").write_text(f"the issue for {instance_id}")
    (d / "eval.sh").write_text("#!/bin/bash\necho hi\n")
    (d / "Dockerfile").write_text(f"FROM scratch # {instance_id}")
    with open(d / "test.patch", "w", newline="") as fh:
        fh.write(CRLF_PATCH)
    return d


def test_load_task_reads_files_and_metadata(tmp_path):
    d = _task(tmp_path, "a__a-1", version="2.0")
    inst = load_task(d)
    assert inst["instance_id"] == "a__a-1"
    assert inst["version"] == "2.0"
    assert inst["patch"] == "gold"
    assert inst["problem_statement"] == "the issue for a__a-1"
    assert inst["eval_script"].startswith("#!/bin/bash")


def test_crlf_in_a_patch_survives(tmp_path):
    d = _task(tmp_path, "a__a-1")
    assert load_task(d)["test_patch"] == CRLF_PATCH


def test_load_task_repo_returns_every_task(tmp_path):
    for i in ("a__a-1", "b__b-2"):
        _task(tmp_path, i)
    assert [i["instance_id"] for i in load_task_repo(tmp_path)] == ["a__a-1", "b__b-2"]


def test_selecting_instances_keeps_the_requested_order(tmp_path):
    for i in ("a__a-1", "b__b-2"):
        _task(tmp_path, i)
    got = load_task_repo(tmp_path, ["b__b-2", "a__a-1"])
    assert [i["instance_id"] for i in got] == ["b__b-2", "a__a-1"]


def test_unknown_instance_id_is_an_error(tmp_path):
    _task(tmp_path, "a__a-1")
    # silently returning fewer instances would shrink a run without saying so
    with pytest.raises(KeyError, match="nope__nope-1"):
        load_task_repo(tmp_path, ["a__a-1", "nope__nope-1"])


def test_dockerfiles_are_keyed_by_instance_id(tmp_path):
    for i in ("a__a-1", "b__b-2"):
        _task(tmp_path, i)
    assert load_dockerfiles(tmp_path, ["a__a-1"]) == {"a__a-1": "FROM scratch # a__a-1"}


def test_assets_live_under_the_task(tmp_path):
    _task(tmp_path, "a__a-1")
    p = asset_path(tmp_path, "a__a-1", "rendering/expected.png")
    assert p == tmp_path / "tasks" / "a__a-1" / "assets" / "rendering" / "expected.png"


def test_a_directory_without_task_json_is_not_a_task(tmp_path):
    _task(tmp_path, "a__a-1")
    (tmp_path / "tasks" / "scratch").mkdir()
    assert [d.name for d in task_dirs(tmp_path)] == ["a__a-1"]


def test_missing_tasks_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_task_repo(tmp_path)


def test_config_names_the_destination(tmp_path):
    _config(tmp_path, dataset="SWE-bench/Example", splits=("dev", "test"))
    assert load_config(tmp_path)["dataset"] == "SWE-bench/Example"
    assert load_config(tmp_path)["splits"] == ["dev", "test"]


def test_config_without_a_dataset_is_an_error(tmp_path):
    (tmp_path / "sweb.yaml").write_text(yaml.safe_dump({"splits": ["test"]}))
    with pytest.raises(ValueError, match="does not name a dataset"):
        load_config(tmp_path)


def test_tasks_group_by_split(tmp_path):
    _config(tmp_path, splits=("dev", "test"))
    _task(tmp_path, "a__a-1", split="test")
    _task(tmp_path, "b__b-2", split="dev")
    grouped = published_tasks(tmp_path)
    assert [i["instance_id"] for i in grouped["test"]] == ["a__a-1"]
    assert [i["instance_id"] for i in grouped["dev"]] == ["b__b-2"]


def test_unpublished_splits_stay_out_of_the_dataset(tmp_path):
    """A deprecated task keeps its directory but never reaches the dataset."""
    _config(tmp_path, splits=("test",))
    _task(tmp_path, "a__a-1", split="test")
    _task(tmp_path, "b__b-2", split="deprecated")
    grouped = published_tasks(tmp_path)
    assert [i["instance_id"] for i in grouped["test"]] == ["a__a-1"]
    assert "deprecated" not in grouped
    # still present in the tree, so it can be built by id
    assert len(load_task_repo(tmp_path)) == 2
