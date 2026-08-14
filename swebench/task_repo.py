"""Reading a task repo, the source of truth for a dataset's instances.

A task repo holds one directory per instance::

    sweb.yaml             the dataset this repo publishes, and its splits
    tasks/<instance_id>/
        task.yaml             metadata, including which split the task is in
        problem_statement.md  the issue text shown to a model
        gold.patch            the reference fix
        test.patch            the tests that grade it
        eval.sh               the script the harness runs
        Dockerfile            the image it runs in
        assets/               binary files a patch cannot carry, e.g. expected.png

Nothing here consults HuggingFace: a task repo can rebuild its dataset alone,
which is what makes local development of a new dataset possible.
"""

from __future__ import annotations

from pathlib import Path

import yaml

TASKS_SUBDIR = "tasks"
ASSETS_SUBDIR = "assets"

# dataset columns kept as their own file rather than inside task.yaml
AS_FILE = {
    "patch": "gold.patch",
    "test_patch": "test.patch",
    "eval_script": "eval.sh",
    "problem_statement": "problem_statement.md",
}
CONFIG_FILE = "sweb.yaml"
TASK_FILE = "task.yaml"


class _Dumper(yaml.SafeDumper):
    """Writes multi-line strings as literal blocks, so they stay readable."""


def _represent_str(dumper: yaml.SafeDumper, data: str):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _represent_str)


def dump_yaml(data: dict) -> str:
    return yaml.dump(
        data,
        Dumper=_Dumper,
        sort_keys=True,
        allow_unicode=True,
        default_flow_style=False,
    )


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _read(path: Path) -> str:
    # newline="" so CRLF inside a patch survives verbatim
    with open(path, newline="") as fh:
        return fh.read()


def tasks_root(repo_path: str | Path) -> Path:
    root = Path(repo_path) / TASKS_SUBDIR
    if not root.is_dir():
        raise FileNotFoundError(f"{repo_path} has no {TASKS_SUBDIR}/ directory")
    return root


def task_dirs(repo_path: str | Path) -> list[Path]:
    return sorted(
        d for d in tasks_root(repo_path).iterdir() if (d / TASK_FILE).exists()
    )


def load_task(task_dir: str | Path) -> dict:
    """One task directory, as the instance dict the harness expects."""
    task_dir = Path(task_dir)
    instance = load_yaml(task_dir / TASK_FILE)
    for column, filename in AS_FILE.items():
        instance[column] = _read(task_dir / filename)
    return instance


def load_task_repo(
    repo_path: str | Path, instance_ids: list[str] | None = None
) -> list[dict]:
    """Every instance in the repo, or just the ones asked for.

    Unknown ids are an error rather than a silent omission: a typo should not
    quietly shrink the run.
    """
    dirs = task_dirs(repo_path)
    if instance_ids is None:
        return [load_task(d) for d in dirs]

    by_id = {d.name: d for d in dirs}
    missing = sorted(set(instance_ids) - set(by_id))
    if missing:
        raise KeyError(f"not in {tasks_root(repo_path)}: {' '.join(missing)}")
    return [load_task(by_id[i]) for i in instance_ids]


def load_dockerfiles(
    repo_path: str | Path, instance_ids: list[str] | None = None
) -> dict[str, str]:
    """instance_id -> Dockerfile contents."""
    dirs = task_dirs(repo_path)
    if instance_ids is not None:
        wanted = set(instance_ids)
        dirs = [d for d in dirs if d.name in wanted]
    return {d.name: (d / "Dockerfile").read_text() for d in dirs}


def asset_path(repo_path: str | Path, instance_id: str, path: str) -> Path:
    """Where a task's binary asset lives, e.g. tasks/<id>/assets/<path>."""
    return tasks_root(repo_path) / instance_id / ASSETS_SUBDIR / path


def load_config(repo_path: str | Path) -> dict:
    """The repo's own description of what it publishes.

    Keeps the destination with the data, so a command cannot be pointed at the
    wrong dataset by mistake.
    """
    path = Path(repo_path) / CONFIG_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{repo_path} has no {CONFIG_FILE}")
    config = load_yaml(path)
    if not config.get("dataset"):
        raise ValueError(f"{path} does not name a dataset")
    config.setdefault("splits", ["test"])
    return config


def published_tasks(repo_path: str | Path) -> dict[str, list[dict]]:
    """Instances grouped by split, skipping splits the repo does not publish.

    A task in an unpublished split (e.g. deprecated) stays in the tree and can
    still be built by id; it just never reaches the dataset.
    """
    splits = set(load_config(repo_path)["splits"])
    grouped: dict[str, list[dict]] = {s: [] for s in splits}
    for instance in load_task_repo(repo_path):
        split = instance.get("split")
        if split in splits:
            grouped[split].append(instance)
    return grouped


def select_tasks(
    repo_path: str | Path, instance_ids: list[str] | None = None
) -> list[dict]:
    """The tasks a command should act on.

    With ids, exactly those, from any split, so an instance being repaired can
    be named even while it sits in an unpublished split. Without ids, only what
    the repo publishes, so deprecated tasks are not built or pushed by default.
    """
    if instance_ids:
        return load_task_repo(repo_path, instance_ids)
    return [
        instance
        for instances in published_tasks(repo_path).values()
        for instance in instances
    ]
