"""`swebench dataset` — build, collect and version task instances."""

from typing import Optional

import typer


def _list(values) -> list[str] | None:
    return list(values) if values else None


def _fail(error: Exception) -> int:
    typer.echo(str(error), err=True)
    return 1


dataset_app = typer.Typer(
    name="dataset",
    help="Build, collect and version task instances.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@dataset_app.command("build")
def build(
    task_repo: str = typer.Argument(
        ..., metavar="TASK_REPO", help="Path to a task repo"
    ),
    datasets: Optional[list[str]] = typer.Option(
        None, "-d", "--dataset", help="Only these datasets (repeatable)"
    ),
    out: str = typer.Option("data", "-o", "--out", help="Where to write the parquets"),
):
    """Compile a task repo into one parquet per split.

    Reads only the repo, so what you get is what the tree says. The repo is
    checked first; a malformed tree is not compiled.

    [yellow][bold]Examples:[/bold][/yellow]

        swebench dataset build ~/swe-bench-multilingual-tasks

        swebench dataset build ~/swe-bench-tasks -o /tmp/parquets
    """
    from swebench.task.publish import CheckFailed, write_parquets
    from swebench.task.repo import UnknownDataset

    try:
        written = write_parquets(task_repo, out, _list(datasets))
    except (CheckFailed, UnknownDataset) as e:
        raise typer.Exit(_fail(e)) from e
    for split, path in written.items():
        typer.echo(f"{split}: {path}")


@dataset_app.command("check")
def check(
    task_repo: str = typer.Argument(
        ..., metavar="TASK_REPO", help="Path to a task repo"
    ),
    fix: bool = typer.Option(False, "--fix", help="Write back what the tree implies"),
):
    """Check that a task repo is well formed.

    Every task needs its files, its metadata and a registered split. With --fix,
    the parts that can be derived from the tree are written back: the split list
    in sweb.yaml, and image names that follow the naming convention.

    [yellow][bold]Examples:[/bold][/yellow]

        swebench dataset check ~/swe-bench-multilingual-tasks

        swebench dataset check ~/swe-bench-tasks --fix
    """
    from swebench.task.checks import check_task_repo, errors

    problems = check_task_repo(task_repo, fix=fix)
    for problem in problems:
        typer.echo(str(problem))
    blocking = errors(problems)
    if blocking:
        typer.echo(
            f"\n{len(blocking)} error(s), {len(problems) - len(blocking)} warning(s)"
        )
        raise typer.Exit(1)
    typer.echo(f"{task_repo} looks well formed")


@dataset_app.command("diff")
def diff(
    task_repo: str = typer.Argument(
        ..., metavar="TASK_REPO", help="Path to a task repo"
    ),
    datasets: Optional[list[str]] = typer.Option(
        None, "-d", "--dataset", help="Only these datasets (repeatable)"
    ),
):
    """Show how a task repo differs from the dataset it publishes.

    [yellow][bold]Examples:[/bold][/yellow]

        swebench dataset diff ~/swe-bench-multilingual-tasks
    """
    from swebench.task.publish import CheckFailed, diff_against_hub, summarize
    from swebench.task.repo import UnknownDataset

    try:
        typer.echo(summarize(diff_against_hub(task_repo, _list(datasets))))
    except (CheckFailed, UnknownDataset) as e:
        raise typer.Exit(_fail(e)) from e


@dataset_app.command("push")
def push(
    task_repo: str = typer.Argument(
        ..., metavar="TASK_REPO", help="Path to a task repo"
    ),
    datasets: Optional[list[str]] = typer.Option(
        None, "-d", "--dataset", help="Only these datasets (repeatable)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be pushed"),
):
    """Overwrite the HuggingFace dataset named in the repo's sweb.yaml.

    [yellow][bold]Examples:[/bold][/yellow]

        swebench dataset push ~/swe-bench-multilingual-tasks --dry-run

        swebench dataset push ~/swe-bench-multilingual-tasks
    """
    from swebench.task.publish import CheckFailed, push_dataset, summarize
    from swebench.task.repo import UnknownDataset

    try:
        typer.echo(summarize(push_dataset(task_repo, _list(datasets), dry_run=dry_run)))
    except (CheckFailed, UnknownDataset) as e:
        raise typer.Exit(_fail(e)) from e


@dataset_app.command("collect")
def collect(
    repos: list[str] = typer.Argument(
        ..., help="GitHub repos, e.g. scikit-learn/scikit-learn"
    ),
    path_prs: str = typer.Option(
        "prs", "--path-prs", help="Where to write the scraped PRs"
    ),
    path_tasks: str = typer.Option(
        "tasks", "--path-tasks", help="Where to write candidate instances"
    ),
    max_pulls: Optional[int] = typer.Option(
        None, "--max-pulls", help="Stop after this many PRs"
    ),
    cutoff_date: Optional[str] = typer.Option(
        None, "--cutoff-date", help="Ignore PRs before YYYYMMDD"
    ),
):
    """Scrape pull requests from GitHub into candidate task instances.

    Produces <repo>-prs.jsonl and <repo>-task-instances.jsonl. The candidates
    still need versions, install specs and validation before they are usable.

    [yellow][bold]Examples:[/bold][/yellow]

        swebench dataset collect scikit-learn/scikit-learn

        swebench dataset collect psf/requests --max-pulls 200 --cutoff-date 20230101
    """
    from swebench.collect.get_tasks_pipeline import main as get_tasks

    get_tasks(
        repos=list(repos),
        path_prs=path_prs,
        path_tasks=path_tasks,
        max_pulls=max_pulls,
        cutoff_date=cutoff_date,
    )
