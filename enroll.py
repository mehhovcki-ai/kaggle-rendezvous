#!/usr/bin/env python3
"""Enroll a Kaggle account as a paws model-proxy benchmark task.

Kaggle gives benchmark tasks a real runtime; a plain kernel gets cut off in
seconds. So this generates a percent-format .py file with a @kbench.task
decorator, bakes the rendezvous URL and key into it, and pushes it with:

    kaggle benchmarks tasks push <task> -f <file>

The task body starts the local OpenAI-compatible proxy, dials the rendezvous
and then blocks, which is what keeps the session serving.

Usage (from a checkout of kaggle-rendezvous):

    python enroll.py --key beta --username kirill

Kaggle auth is whatever the `kaggle` CLI already uses; this shells out to it
instead of importing the package, because `uv run` gives the script its own
environment where `import kaggle` can fail while the CLI is on PATH.

Options:
    --key           rendezvous key the task presents on /connect
    --username      pawsapi username; used for the task slug
    --kaggle-user   Kaggle account name (rarely needed)
    --kaggle-key    legacy API key (sets KAGGLE_USERNAME/KAGGLE_KEY)
    --slug          task slug (default: paws-bridge-<username>)
    --url           rendezvous websocket URL
    --dry-run       write the task file locally, do not push
    --out           output directory (default: /tmp/<slug>)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_URL = "wss://rendezvous.paws.best/connect"
HERE = Path(__file__).resolve().parent

# Percent-format cells, in order. jupytext turns these into one notebook when
# the CLI pushes. The @task decorator is what makes Kaggle treat it as a
# benchmark instead of a short-lived kernel.
HEADER = """# %% [markdown]
# paws bridge for {username}
#
# Serves the paws rendezvous bridge. Started by enroll.py; the key and URL
# below are generated per account.

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
def {func_name}(llm) -> None:
    server = serve(port=8000)
    bridge = connect(port=8000).ready(timeout=90)
    print("connected as", bridge.notebook_id, "->", bridge.public_base_url)

    # No shutdown logic: the rendezvous stops routing new requests ten
    # minutes before the session cap. This just holds the session open.
    while True:
        time.sleep(60)


{func_name}.run(kbench.llm)
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
    func_name = slug.replace("-", "_")
    task_name = slug.replace("-", " ")
    return HEADER.format(
        username=username,
        url=url,
        key=key,
        proxy=read_source("proxy_cell.py").rstrip(),
        bridge=read_source("bridge_cell.py").rstrip(),
        task_name=task_name,
        func_name=func_name,
    )


def kaggle_exe() -> str:
    exe = shutil.which("kaggle")
    if not exe:
        sys.exit("the kaggle CLI is not on PATH; install it or fix PATH")
    return exe


def kaggle_env(env: dict, kaggle_user: str | None, kaggle_key: str | None) -> dict:
    out = env.copy()
    if kaggle_key:
        if not kaggle_user:
            sys.exit("--kaggle-key needs --kaggle-user as well")
        out["KAGGLE_USERNAME"] = kaggle_user
        out["KAGGLE_KEY"] = kaggle_key
    return out


def kaggle(*argv: str, env: dict, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [kaggle_exe(), *argv], capture_output=True, text=True, env=env
    )
    if check and proc.returncode != 0:
        print((proc.stdout or "") + (proc.stderr or ""), file=sys.stderr)
        sys.exit(f"kaggle {' '.join(argv[:2])} failed")
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True, help="rendezvous key")
    parser.add_argument("--username", required=True, help="pawsapi username")
    parser.add_argument("--kaggle-user", default=None)
    parser.add_argument("--kaggle-key", default=None)
    parser.add_argument("--slug", default=None)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--dry-run", action="store_true")
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

    env = kaggle_env(os.environ, args.kaggle_user, args.kaggle_key)
    print("pushing benchmark task...")
    push = kaggle(
        "benchmarks", "tasks", "push", slug, "-f", str(task_file),
        env=env, check=False,
    )
    output = (push.stdout or "") + (push.stderr or "")
    print(output.strip()[-1200:] or "(no output)")
    if push.returncode != 0:
        lowered = output.lower()
        if any(w in lowered for w in ("authenticat", "credential", "login", "api token")):
            sys.exit("Kaggle is not authenticated. Run `kaggle auth login` first.")
        sys.exit("push failed; see the output above")

    print(f"submitted: https://www.kaggle.com/benchmarks/tasks/{slug}")
    print("the task keeps serving until the session cap.")


if __name__ == "__main__":
    main()
