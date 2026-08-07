#!/usr/bin/env python
"""Generate a v5-format dataset from a v4 dataset.

v5 stops deriving environments from Python constants and instead reads four
extra fields off each dataset row:

    eval_script   the full test script, with the test_patch applied inline
    log_parser    name of the parser function for the repo
    eval_type     "fail_only" or "pass_and_fail"
    image         the instance image name

Those fields are produced by harness code that v5 deletes, so this generator
must run from a **v4 checkout** (main or port-mm). Without it there is no
reproducible way to rebuild a dataset after a fix changes an eval script.

Verified against the dataset v5 already ships: regenerating
SWE-bench_Multimodal/dev reproduces log_parser, eval_type and image exactly
for all 102 rows.

Usage:
    python tools/make_v5_dataset.py --dataset SWE-bench/SWE-bench_Multimodal \
        --split test --out data/SWE-bench_Multimodal/test.parquet
    python tools/make_v5_dataset.py --in mm_test.corrected8.json \
        --split test --out test.parquet
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from swebench.harness.constants import EvalType, FAIL_ONLY_REPOS
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.test_spec import make_test_spec

# the 12 public columns, in the order the shipped v5 parquets use
PUBLIC_COLUMNS = [
    "repo",
    "instance_id",
    "base_commit",
    "patch",
    "test_patch",
    "problem_statement",
    "hints_text",
    "created_at",
    "image_assets",
    "version",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
]
V5_COLUMNS = ["log_parser", "eval_type", "eval_script", "image"]


def image_name(instance_id: str, arch: str = "amd64") -> str:
    return f"{arch}.{instance_id.lower()}:latest"


def eval_type_for(repo: str) -> str:
    return (
        EvalType.FAIL_ONLY.value
        if repo in FAIL_ONLY_REPOS
        else EvalType.PASS_AND_FAIL.value
    )


def build_row(inst: dict, arch: str) -> dict:
    repo = inst["repo"]
    row = {c: inst.get(c, "") for c in PUBLIC_COLUMNS if c in inst or c != "image_assets"}
    row["log_parser"] = MAP_REPO_TO_PARSER[repo].__name__
    row["eval_type"] = eval_type_for(repo)
    row["eval_script"] = make_test_spec(inst).eval_script
    row["image"] = image_name(inst["instance_id"], arch)
    return row


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="HF dataset name")
    src.add_argument("--in", dest="in_path", help="local .json (list of instances)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="amd64", choices=["amd64", "arm64"])
    ap.add_argument(
        "--compare",
        help="existing v5 parquet to validate the generated fields against",
    )
    args = ap.parse_args()

    if args.dataset:
        from datasets import load_dataset

        rows = [dict(r) for r in load_dataset(args.dataset, split=args.split)]
    else:
        rows = json.loads(Path(args.in_path).read_text())

    out_rows, skipped = [], []
    for inst in rows:
        if inst["repo"] not in MAP_REPO_TO_PARSER:
            skipped.append((inst["instance_id"], "no log parser"))
            continue
        try:
            out_rows.append(build_row(inst, args.arch))
        except Exception as e:  # noqa: BLE001 - report and continue
            skipped.append((inst["instance_id"], f"{type(e).__name__}: {e}"))

    df = pd.DataFrame(out_rows)
    for c in PUBLIC_COLUMNS + V5_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[PUBLIC_COLUMNS + V5_COLUMNS]

    if args.compare:
        ref = pd.read_parquet(args.compare).set_index("instance_id")
        mine = df.set_index("instance_id")
        shared = ref.index.intersection(mine.index)
        print(f"comparing {len(shared)} shared instances against {args.compare}")
        for col in V5_COLUMNS:
            diff = sum(1 for i in shared if str(ref.loc[i, col]) != str(mine.loc[i, col]))
            print(f"  {col:12s} differs on {diff}/{len(shared)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for iid, why in skipped[:20]:
            print(f"   {iid}: {why}")


if __name__ == "__main__":
    main()
