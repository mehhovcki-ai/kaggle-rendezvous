#!/usr/bin/env python3
"""Enroll a Kaggle account as a paws model-proxy notebook.

Two steps, on purpose:

  1. push a benchmark task (creation only, nothing runs). That is what makes
     Kaggle build a backing notebook with the benchmarks environment.
  2. run that backing notebook as a notebook, not as a task. The notebook
     starts the proxy, dials the rendezvous and blocks, so it keeps serving
     instead of being a one-shot evaluation.

Usage (from a checkout of kaggle-rendezvous):

    python enroll.py --key beta --username kirill

    --no-run      only push the task, do not start the notebook
    --dry-run     write the task file locally, do not push anything

Kaggle auth is whatever the `kaggle` CLI already uses; this shells out to it
instead of importing the package.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_URL = "wss://rendezvous.paws.best/connect"
HERE = Path(__file__).resolve().parent

HEADER = """# %% [markdown]
# paws bridge for {username}
#
# Pushed as a benchmark task so the environment carries kaggle_benchmarks,
# then the backing notebook is run directly. The key and URL are generated
# per account.

# %%
%pip install -q "fastapi==0.135.1" "uvicorn==0.42.0" "websockets==15.0.1"

# %%
import os

os.environ["RENDEZVOUS_URL"] = {url!r}
os.environ["RENDEZVOUS_KEY"] = {key!r}
os.environ.setdefault("PROXY_API_KEY", "local-only")
print("rendezvous:", os.environ["RENDEZVOUS_URL"])

# %%
# --- proxy ---------------------------------------------------------------
{proxy}

# %%
# --- bridge --------------------------------------------------------------
{bridge}

# %%
import time

import kaggle_benchmarks as kbench


@kbench.task(
    name={task_name!r},
    description="Runs the paws rendezvous proxy for this account.",
)
def {func_name}(llm=None) -> None:
    server = serve(port=8000)
    bridge = connect(port=8000).ready(timeout=90)
    print("connected as", bridge.notebook_id, "->", bridge.public_base_url)

    # No shutdown logic: the rendezvous stops routing new requests ten
    # minutes before the session cap. This just holds the session open.
    while True:
        time.sleep(60)


# Run on import, whether this executes as a task or as the backing notebook.
{func_name}()
"""


def read_source(name: str) -> str:
    path = HERE / name
    if not path.is_file():
        sys.exit(f"missing {name} next to enroll.py (run from the repo checkout)")
    return path.read_text()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def build_task(url: str, key: str, username: str, slug: str) -> str:
    return HEADER.format(
        username=username,
        url=url,
        key=key,
        proxy=read_source("proxy_cell.py").rstrip(),
        bridge=read_source("bridge_cell.py").rstrip(),
        task_name=slug.replace("-", " "),
        func_name=slug.replace("-", "_"),
    )


def kaggle_exe() -> str:
    exe = shutil.which("kaggle")
    if not exe:
        sys.exit("the kaggle CLI is not on PATH; install it or fix PATH")
    return exe


def kaggle(*argv: str, env: dict, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run([kaggle_exe(), *argv], capture_output=True, text=True, env=env)
    if check and proc.returncode != 0:
        print((proc.stdout or "") + (proc.stderr or ""), file=sys.stderr)
        sys.exit(f"kaggle {' '.join(argv[:2])} failed")
    return proc


def find_backing_notebook(slug: str, env: dict) -> str | None:
    """Find the notebook Kaggle created for the task, without needing a username.

    `kaggle config view` is a display format and its parsing broke on a real
    machine, so discovery goes through the CSV listing instead: refs are
    machine-readable and already carry owner/slug.
    """
    import csv
    import io

    listing = kaggle("kernels", "list", "-m", "--csv", "-s", slug,
                     "--page-size", "20", env=env, check=False)
    text = (listing.stdout or "") + (listing.stderr or "")
    for row in csv.DictReader(io.StringIO(text)):
        ref = (row.get("ref") or "").strip()
        if ref and slug in ref:
            return ref
    return None


def wait_for_task(slug: str, env: dict, timeout: int = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        proc = kaggle("benchmarks", "tasks", "status", slug, env=env, check=False)
        text = (proc.stdout or "") + (proc.stderr or "")
        if re.search(r"status:\s*(completed|complete)", text, re.I):
            return True
        if re.search(r"status:\s*(failed|error)", text, re.I):
            print(text.strip()[-400:])
            return False
        time.sleep(8)
    return False


def task_ready(slug: str, env: dict) -> bool:
    """True when the task already exists and finished creation."""
    proc = kaggle("benchmarks", "tasks", "status", slug, env=env, check=False)
    text = ((proc.stdout or "") + (proc.stderr or "")).lower()
    return proc.returncode == 0 and "status:" in text and "completed" in text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--kaggle-user", default=None)
    parser.add_argument("--kaggle-key", default=None)
    parser.add_argument("--slug", default=None)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-run", action="store_true", help="push the task only")
    parser.add_argument("--recreate", action="store_true",
                        help="build a fresh task version instead of reusing the task")
    parser.add_argument("--model", default="google/gemini-3-flash-preview",
                        help="model slug the benchmark run is scheduled against")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    slug = args.slug or f"paws-bridge-{slugify(args.username)}"
    task = build_task(args.url, args.key, args.username, slug)

    out = Path(args.out or f"/tmp/{slug}")
    out.mkdir(parents=True, exist_ok=True)
    task_file = out / f"{slug}.py"
    task_file.write_text(task)
    print(f"wrote {task_file} ({len(task.splitlines())} lines)")
    print(f"task: {slug}")

    if args.dry_run:
        print("dry run: not pushing")
        return

    env = os.environ.copy()
    if args.kaggle_key:
        if not args.kaggle_user:
            sys.exit("--kaggle-key needs --kaggle-user as well")
        env["KAGGLE_USERNAME"] = args.kaggle_user
        env["KAGGLE_KEY"] = args.kaggle_key

    if args.recreate:
        exists = False
    else:
        exists = task_ready(slug, env)
        if exists:
            print(f"reusing existing task: {slug}")

    if not exists:
        print("pushing benchmark task...")
        push = kaggle("benchmarks", "tasks", "push", slug, "-f", str(task_file),
                      env=env, check=False)
        output = (push.stdout or "") + (push.stderr or "")
        print(output.strip()[-1000:] or "(no output)")
        if push.returncode != 0:
            lowered = output.lower()
            if any(w in lowered for w in ("authenticat", "credential", "login", "api token")):
                sys.exit("Kaggle is not authenticated. Run `kaggle auth login` first.")
            sys.exit("task push failed; see the output above")

        if args.no_run:
            print("task created, not run. `kaggle b t run` when you want it.")
            return

        print("waiting for task creation...")
        if not wait_for_task(slug, env):
            print("task creation did not report complete in time; check `kaggle b t status`")

    if args.no_run:
        print("task exists, not started (--no-run).")
        return

    # The task run is what gets the benchmarks environment. Running the
    # backing notebook directly does not, so this is the only path that works.
    print("starting the task run...")
    run = kaggle("benchmarks", "tasks", "run", slug, "-m", args.model,
                 env=env, check=False)
    run_out = (run.stdout or "") + (run.stderr or "")
    print(run_out.strip()[-800:] or "(no output)")
    if run.returncode != 0:
        sys.exit("task run failed; see the output above")

    print(f"submitted: https://www.kaggle.com/benchmarks/tasks/{slug}")
    print("the task keeps serving until the session cap.")


if __name__ == "__main__":
    main()
