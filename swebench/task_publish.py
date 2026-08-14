"""Compiling a task repo into the things people consume: a dataset, and images.

Both directions read the tree and nothing else, so what gets published is always
what the repo says. Every entry point checks the repo first: publishing from a
malformed tree is how a broken instance reaches a leaderboard.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from swebench.task_checks import check_task_repo, errors
from swebench.task_repo import load_config, published_tasks, select_tasks

# `split` says which parquet a task belongs in; it is not a dataset column
INTERNAL_KEYS = ("split",)


class CheckFailed(RuntimeError):
    """The repo is not well formed, so nothing was published."""


def guard(repo_path: str | Path) -> None:
    """Refuse to go further if the repo has errors."""
    blocking = errors(check_task_repo(repo_path))
    if blocking:
        listed = "\n".join(f"  {p}" for p in blocking[:20])
        more = f"\n  ... and {len(blocking) - 20} more" if len(blocking) > 20 else ""
        raise CheckFailed(f"{len(blocking)} problem(s) in {repo_path}:\n{listed}{more}")


def _rows_for_split(instances: list[dict]) -> list[dict]:
    return [
        {k: v for k, v in inst.items() if k not in INTERNAL_KEYS} for inst in instances
    ]


def compile_splits(repo_path: str | Path) -> dict[str, list[dict]]:
    """The dataset this repo publishes, split by split."""
    guard(repo_path)
    return {
        split: _rows_for_split(instances)
        for split, instances in published_tasks(repo_path).items()
    }


def write_parquets(repo_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Write one parquet per split, the layout `load_dataset` expects."""
    import pandas as pd

    config = load_config(repo_path)
    out = Path(out_dir) / config["dataset"].split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)

    written = {}
    for split, rows in compile_splits(repo_path).items():
        path = out / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        written[split] = path
    return written


def diff_against_hub(repo_path: str | Path) -> dict[str, dict]:
    """How the tree differs from what is published, per split.

    Reported per column rather than per row: "47 instances differ in eval_script"
    is actionable, a list of 47 ids is not.
    """
    from datasets import load_dataset

    config = load_config(repo_path)
    report = {}
    for split, rows in compile_splits(repo_path).items():
        tree = {r["instance_id"]: r for r in rows}
        try:
            live = {
                r["instance_id"]: r
                for r in load_dataset(config["dataset"], split=split)
            }
        except Exception as e:  # noqa: BLE001 - datasets raises many types when a dataset or split is absent
            report[split] = {"error": f"{type(e).__name__}: {e}"}
            continue
        added = sorted(set(tree) - set(live))
        removed = sorted(set(live) - set(tree))
        changed: dict[str, int] = {}
        for iid in set(tree) & set(live):
            for column in tree[iid]:
                if column in live[iid] and tree[iid][column] != live[iid][column]:
                    changed[column] = changed.get(column, 0) + 1
        report[split] = {"added": added, "removed": removed, "changed": changed}
    return report


def push_dataset(repo_path: str | Path, dry_run: bool = False) -> dict:
    """Overwrite the dataset named in config.json with the tree."""
    config = load_config(repo_path)
    splits = compile_splits(repo_path)
    plan = {
        "dataset": config["dataset"],
        "splits": {s: len(rows) for s, rows in splits.items()},
        "diff": diff_against_hub(repo_path),
    }
    if dry_run:
        return plan

    from datasets import Dataset

    for split, rows in splits.items():
        Dataset.from_list(rows).push_to_hub(config["dataset"], split=split)
    return plan


def _images_for(
    repo_path: str | Path, instance_ids: list[str] | None
) -> dict[str, str]:
    return {i["instance_id"]: i["image"] for i in select_tasks(repo_path, instance_ids)}


def push_images(
    repo_path: str | Path,
    instance_ids: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    """Push each task's image under exactly the name its task.json declares.

    The dataset tells the harness which image to pull, so publishing any other
    name produces a dataset that cannot be run.
    """
    guard(repo_path)
    images = _images_for(repo_path, instance_ids)

    import docker

    client = docker.from_env()
    local = {tag for image in client.images.list() for tag in (image.tags or [])}

    missing = sorted(name for name in images.values() if name not in local)
    pushable = sorted(name for name in images.values() if name in local)
    plan = {
        "to_push": pushable,
        "not_built": missing,
        "commands": [f"docker push {name}" for name in pushable],
    }
    if dry_run:
        return plan

    pushed, failed = [], {}
    for name in pushable:
        result = subprocess.run(
            ["docker", "push", name], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            pushed.append(name)
        else:
            failed[name] = (result.stderr.strip().splitlines() or ["push failed"])[-1]
    plan["pushed"] = pushed
    plan["failed"] = failed
    return plan


def summarize(plan: dict) -> str:
    return json.dumps(plan, indent=2, default=str)
