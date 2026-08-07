#!/usr/bin/env python
"""Generate v5 flat Dockerfiles from the v4 harness.

v4 builds three layered images per instance (base -> env -> instance) and
COPYs setup_env.sh / setup_repo.sh in. v5 uses one self-contained Dockerfile
per instance, with those scripts inlined as `RUN <<EOF_<hash>` heredocs, plus
a block that pre-downloads the instance's image assets.

The environment fixes live in the v4 constants and test_spec code, so the flat
files have to be generated from a v4 checkout -- otherwise they ship without
them. The currently published SWE-bench Multimodal Dockerfiles predate the
repairs and contain none of the TeX Live 2024 pin, babel-french, or the carbon
accessibility-checker archive pin.

Output matches the layout of the dockerfile repos:

    <out>/<instance_id>.Dockerfile

Usage:
    python tools/make_v5_dockerfiles.py --dataset mm_test.corrected8.json \
        --out /path/to/swe-bench-multimodal-dockerfiles/src/dockerfiles
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from swebench.harness.test_spec.test_spec import make_test_spec


def heredoc_delim(*parts: str) -> str:
    """Stable per-content delimiter, matching the shipped files' EOF_<hex> form."""
    h = hashlib.sha256("".join(parts).encode()).hexdigest()[:12]
    return f"EOF_{h}"


def as_heredoc(script: str) -> str:
    d = heredoc_delim(script)
    while d in script:  # collision guard
        d = heredoc_delim(script, d)
    return f"RUN <<{d}\n{script.rstrip()}\n{d}\n"


def image_assets_block(instance: dict) -> str:
    """Pre-download the instance's images into /swebench/image_assets."""
    raw = instance.get("image_assets") or "{}"
    try:
        assets = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return ""
    lines = ["#!/bin/bash", "set -euxo pipefail", "mkdir -p /swebench/image_assets"]
    any_asset = False
    for key in ("problem_statement", "patch", "test_patch"):
        for item in assets.get(key) or []:
            # problem_statement entries are bare URLs; patch/test_patch entries are
            # {path, url} and must keep their repo-relative path
            if isinstance(item, dict):
                url = item.get("url")
                rel = item.get("path") or (url or "").rstrip("/").split("/")[-1]
            else:
                url = item
                rel = (url or "").rstrip("/").split("/")[-1]
            if not url or not rel:
                continue
            any_asset = True
            dest = f"/swebench/image_assets/{key}/{rel}"
            lines.append(f"mkdir -p $(dirname '{dest}')")
            lines.append(f"curl -fsSL -o '{dest}' '{url}' || true")
    return as_heredoc("\n".join(lines)) if any_asset else ""


def strip_from(dockerfile: str) -> str:
    """Drop the FROM line of a downstream layer; only the base FROM survives."""
    return "\n".join(
        l for l in dockerfile.split("\n") if not l.lstrip().startswith("FROM ")
    )


def drop_copy_and_run(dockerfile: str, script_name: str) -> str:
    """Remove the COPY/chmod/sed/RUN lines that handled a now-inlined script."""
    out = []
    for line in dockerfile.split("\n"):
        if script_name in line and re.match(r"\s*(COPY|RUN)\b", line):
            continue
        out.append(line)
    return "\n".join(out)


def build_flat(instance: dict) -> str:
    ts = make_test_spec(instance, namespace=None)

    base = ts.base_dockerfile.strip()
    # v5's published files target amd64/jammy; 22.04 and jammy are the same image
    base = base.replace("--platform=linux/x86_64", "--platform=linux/amd64")
    base = base.replace("ubuntu:22.04", "ubuntu:jammy")

    env = strip_from(ts.env_dockerfile)
    env = drop_copy_and_run(env, "setup_env.sh")

    inst = strip_from(ts.instance_dockerfile)
    inst = drop_copy_and_run(inst, "setup_repo.sh")

    parts = [
        base,
        "",
        env.strip(),
        "",
        as_heredoc(ts.setup_env_script),
        "",
        inst.strip(),
        "",
        as_heredoc(ts.install_repo_script),
        "",
        image_assets_block(instance),
        "",
        "WORKDIR /testbed",
        "",
    ]
    return "\n".join(p for p in parts if p is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="local .json (list of instances)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--instances", nargs="*", help="subset; default = all")
    args = ap.parse_args()

    rows = json.loads(Path(args.dataset).read_text())
    if args.instances:
        want = set(args.instances)
        rows = [r for r in rows if r["instance_id"] in want]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written, failed = 0, []
    for r in rows:
        try:
            (out / f"{r['instance_id']}.Dockerfile").write_text(build_flat(r))
            written += 1
        except Exception as e:  # noqa: BLE001
            failed.append((r["instance_id"], f"{type(e).__name__}: {e}"))

    print(f"wrote {written} Dockerfiles -> {out}")
    if failed:
        print(f"failed {len(failed)}:")
        for iid, why in failed[:20]:
            print(f"   {iid}: {why}")


if __name__ == "__main__":
    main()
