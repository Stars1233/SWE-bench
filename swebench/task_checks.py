"""Is a task repo well formed?

The tree is the source of truth, so a malformed task is a broken instance later:
a missing eval.sh cannot be run, a split nobody registered never gets published,
an image name that disagrees with the convention pulls the wrong container.

`check` reports; `check --fix` writes back only what can be derived from the
tree itself, and never guesses at data it cannot see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from swebench.task_repo import (
    AS_FILE,
    CONFIG_FILE,
    load_config,
    task_dirs,
    tasks_root,
)

# kept in the tree but deliberately not published, e.g. instances that were dropped
DEPRECATED_SPLIT = "deprecated"

REQUIRED_FILES = ("task.json", "Dockerfile", *AS_FILE.values())
REQUIRED_KEYS = (
    "instance_id",
    "repo",
    "version",
    "base_commit",
    "split",
    "image",
    "log_parser",
    "eval_type",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)
# the eval script has to bracket its test output or grading cannot read the run
EVAL_MARKERS = (">>>>> Start Test Output", ">>>>> End Test Output")


@dataclass
class Problem:
    where: str
    message: str
    fixable: bool = False
    # errors stop a publish; warnings are reported and do not
    severity: str = "error"

    def __str__(self) -> str:
        return f"{self.severity}: {self.where}: {self.message}"


def errors(problems: list[Problem]) -> list[Problem]:
    return [p for p in problems if p.severity == "error"]


def expected_image(instance_id: str, namespace: str = "swebench") -> str:
    """The image name the harness derives for an instance."""
    key = f"sweb.eval.x86_64.{instance_id}:latest".lower()
    return f"{namespace}/{key}".replace("__", "_1776_")


def _check_task(task_dir: Path, known_splits: set[str]) -> list[Problem]:
    problems = []
    where = f"tasks/{task_dir.name}"

    missing = [f for f in REQUIRED_FILES if not (task_dir / f).is_file()]
    if missing:
        problems.append(Problem(where, f"missing {', '.join(sorted(missing))}"))
    if not (task_dir / "task.json").is_file():
        return problems  # nothing further can be checked

    try:
        meta = json.loads((task_dir / "task.json").read_text())
    except json.JSONDecodeError as e:
        return [*problems, Problem(where, f"task.json is not valid JSON: {e}")]

    for key in REQUIRED_KEYS:
        if key not in meta:
            problems.append(Problem(where, f"task.json has no {key}"))
    if meta.get("instance_id") not in (None, task_dir.name):
        problems.append(
            Problem(where, f"task.json instance_id is {meta['instance_id']!r}")
        )
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        if key in meta and not isinstance(meta[key], list):
            problems.append(
                Problem(where, f"{key} is {type(meta[key]).__name__}, not a list")
            )
    if meta.get("FAIL_TO_PASS") == [] and meta.get("split") != DEPRECATED_SPLIT:
        problems.append(Problem(where, "no FAIL_TO_PASS, so nothing grades this task"))

    split = meta.get("split")
    if split is not None and split not in known_splits | {DEPRECATED_SPLIT}:
        problems.append(
            Problem(where, f"split {split!r} is not in {CONFIG_FILE}", fixable=True)
        )
    if "image" in meta and meta["image"] != expected_image(task_dir.name):
        problems.append(
            Problem(
                where, f"image {meta['image']!r} is not the derived name", fixable=True
            )
        )

    if (task_dir / "eval.sh").is_file():
        script = (task_dir / "eval.sh").read_text()
        for marker in EVAL_MARKERS:
            if marker not in script:
                problems.append(Problem(where, f"eval.sh has no {marker!r} marker"))
    for name in ("gold.patch", "test.patch"):
        path = task_dir / name
        if (
            path.is_file()
            and path.stat().st_size
            and "diff --git" not in path.read_text(errors="replace")
        ):
            problems.append(Problem(where, f"{name} does not look like a diff"))
    return problems


def check_task_repo(repo_path: str | Path, fix: bool = False) -> list[Problem]:
    """Every problem found, in reporting order. With fix, repair the fixable ones."""
    repo_path = Path(repo_path)
    problems: list[Problem] = []

    try:
        config = load_config(repo_path)
        known_splits = set(config["splits"])
    except (FileNotFoundError, ValueError) as e:
        return [Problem(CONFIG_FILE, str(e))]

    try:
        dirs = task_dirs(repo_path)
    except FileNotFoundError as e:
        return [Problem("tasks/", str(e))]
    if not dirs:
        return [Problem("tasks/", "no task directories")]

    for task_dir in dirs:
        problems.extend(_check_task(task_dir, known_splits))

    # splits are derivable from the tree, so an unregistered one can be filled in
    if fix:
        used = {
            json.loads((d / "task.json").read_text()).get("split")
            for d in dirs
            if (d / "task.json").is_file()
        }
        publishable = sorted(s for s in used if s and s != DEPRECATED_SPLIT)
        if publishable and publishable != sorted(known_splits):
            config["splits"] = publishable
            (repo_path / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")
            problems = [
                p for p in problems if not (p.fixable and CONFIG_FILE in p.message)
            ]
        for task_dir in dirs:
            meta_path = task_dir / "task.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text())
            derived = expected_image(task_dir.name)
            if meta.get("image") != derived:
                meta["image"] = derived
                meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
                problems = [
                    p
                    for p in problems
                    if not (
                        p.fixable
                        and p.where.endswith(task_dir.name)
                        and "image" in p.message
                    )
                ]

    # a directory with no task.json is invisible to the loaders, so say so
    strays = [
        d.name
        for d in tasks_root(repo_path).iterdir()
        if d.is_dir() and not (d / "task.json").is_file()
    ]
    problems.extend(
        Problem(f"tasks/{name}", "no task.json, so it is not a task") for name in strays
    )
    return problems
