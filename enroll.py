#!/usr/bin/env python3
"""Enroll a Kaggle account as a paws model-proxy notebook.

Generates a single-cell notebook with the rendezvous URL and key baked in,
pushes it with the `kaggle` CLI and starts it. Re-running is safe: the kernel
id comes from the account name, so a second run updates and restarts the same
notebook instead of creating another one.

Usage (from a checkout of kaggle-rendezvous):

    python enroll.py --key beta --username kirill

Kaggle authentication is whatever the `kaggle` CLI already uses, which is why
this shell outs to it instead of importing the package: `uv run` gives the
script an isolated environment where `import kaggle` fails even though the
CLI is on PATH. If the CLI has no credentials you get:

    Kaggle is not authenticated. Run `kaggle auth login` first.

Options:
    --key           rendezvous key the notebook presents on /connect
    --username      pawsapi username; used for the kernel slug
    --kaggle-user   Kaggle account name (default: from `kaggle config view`)
    --kaggle-key    legacy API key (sets KAGGLE_USERNAME/KAGGLE_KEY for the CLI)
    --slug          kernel slug (default: paws-bridge-<username>)
    --url           rendezvous websocket URL
    --public        make the notebook public (default: private)
    --dry-run       write the notebook locally, do not push
    --out           output directory (default: /tmp/<slug>)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_URL = "wss://rendezvous.paws.best/connect"
HERE = Path(__file__).resolve().parent

# The notebook is one cell. Order matters: install, config, proxy, bridge,
# then the runner. Everything below is concatenated into that cell.
INSTALL = '%pip install -q "fastapi==0.135.1" "uvicorn==0.42.0" "websockets==15.0.1"\n'

CONFIG = '''import os

# Injected by enroll.py for this account. Not Kaggle secrets: the notebook is
# private and this only says which proxy it is.
os.environ["RENDEZVOUS_URL"] = {url!r}
os.environ["RENDEZVOUS_KEY"] = {key!r}
os.environ.setdefault("PROXY_API_KEY", "local-only")
print("rendezvous:", os.environ["RENDEZVOUS_URL"])


'''

RUNNER = '''

# --- runner ---------------------------------------------------------------
server = serve(port=8000)
bridge = connect(port=8000).ready(timeout=90)
print("connected as", bridge.notebook_id, "->", bridge.public_base_url)

# No shutdown logic: the rendezvous stops routing new requests ten minutes
# before the session cap, so this cell just holds the session open.
import time as _time

while True:
    _time.sleep(60)
'''


def read_source(name: str) -> str:
    path = HERE / name
    if not path.is_file():
        sys.exit(f"missing {name} next to enroll.py (run from the repo checkout)")
    return path.read_text()


def build_cell(url: str, key: str) -> str:
    proxy = read_source("proxy_cell.py")
    bridge = read_source("bridge_cell.py")
    return "".join([
        INSTALL,
        CONFIG.format(url=url, key=key),
        "# --- proxy ---------------------------------------------------------------\n",
        proxy,
        "\n\n# --- bridge --------------------------------------------------------------\n",
        bridge,
        RUNNER,
    ])


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def notebook(cell: str) -> dict:
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def kernel_metadata(owner: str, slug: str, private: bool) -> dict:
    return {
        "id": f"{owner}/{slug}",
        "title": slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": private,
        "enable_gpu": False,
        "enable_internet": True,
        "machine_shape": "None",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def kaggle_exe() -> str:
    exe = shutil.which("kaggle")
    if not exe:
        sys.exit("the kaggle CLI is not on PATH; install it or fix PATH")
    return exe


def kaggle_env(args: argparse.Namespace) -> dict:
    env = os.environ.copy()
    if args.kaggle_key:
        if not args.kaggle_user:
            sys.exit("--kaggle-key needs --kaggle-user as well")
        env["KAGGLE_USERNAME"] = args.kaggle_user
        env["KAGGLE_KEY"] = args.kaggle_key
    return env


def kaggle(*argv: str, env: dict, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [kaggle_exe(), *argv], capture_output=True, text=True, env=env
    )
    if check and proc.returncode != 0:
        output = (proc.stdout + proc.stderr).strip()
        lowered = output.lower()
        if "authenticate" in lowered or "credentials" in lowered or "login" in lowered:
            sys.exit("Kaggle is not authenticated. Run `kaggle auth login` first.")
        sys.exit(f"kaggle {argv[0]} {argv[1] if len(argv) > 1 else ''} failed:\n{output}")
    return proc


def kaggle_owner(args: argparse.Namespace, env: dict) -> str:
    if args.kaggle_user:
        return args.kaggle_user
    proc = kaggle("config", "view", env=env, check=False)
    text = (proc.stdout or "") + (proc.stderr or "")
    match = re.search(r"username[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9_-]+)", text)
    if match:
        return match.group(1)
    # the credentials file newer CLIs write
    try:
        data = json.loads((Path.home() / ".kaggle" / "credentials.json").read_text())
        if data.get("username"):
            return data["username"]
    except Exception:
        pass
    # last resort: an authenticated API call that names our own kernels.
    listing = kaggle("kernels", "list", "-m", "--page-size", "1", env=env, check=False)
    for line in ((listing.stdout or "") + (listing.stderr or "")).splitlines():
        match = re.match(r"([A-Za-z0-9_-]+)/[A-Za-z0-9_-]+", line.strip())
        if match:
            return match.group(1)
    sys.exit("could not determine your Kaggle username; pass --kaggle-user")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True, help="rendezvous key")
    parser.add_argument("--username", required=True, help="pawsapi username")
    parser.add_argument("--kaggle-user", default=None)
    parser.add_argument("--kaggle-key", default=None)
    parser.add_argument("--slug", default=None)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--public", dest="private", action="store_false", default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    slug = args.slug or f"paws-bridge-{slugify(args.username)}"
    cell = build_cell(args.url, args.key)

    out = Path(args.out or f"/tmp/{slug}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(notebook(cell), indent=1))

    env = kaggle_env(args)
    owner = args.kaggle_user or args.username if args.dry_run else kaggle_owner(args, env)

    (out / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata(owner, slug, args.private), indent=2)
    )
    print(f"wrote {out}/notebook.ipynb ({len(cell.splitlines())} lines, 1 cell)")
    print(f"kernel id: {owner}/{slug}  private={args.private}")

    if args.dry_run:
        print("dry run: not pushing")
        return

    # The push itself is the only reliable auth test. `kaggle config view`
    # exits non-zero on a machine that is perfectly well authenticated, so
    # probing with it produced false negatives.
    print("pushing kernel...")
    push = kaggle("kernels", "push", "-p", str(out), env=env, check=False)
    output = (push.stdout or "") + (push.stderr or "")
    print(output.strip()[-800:] or "(no output)")
    if push.returncode != 0:
        lowered = output.lower()
        if any(word in lowered for word in ("authenticat", "credential", "login", "api token")):
            sys.exit("Kaggle is not authenticated. Run `kaggle auth login` first.")
        sys.exit("push failed; see the output above")

    kernel = f"{owner}/{slug}"
    print(f"started: https://www.kaggle.com/code/{kernel}")
    for _ in range(10):
        status = kaggle("kernels", "status", kernel, env=env, check=False)
        line = ((status.stdout or "") + (status.stderr or "")).strip().splitlines()
        if line:
            print("status:", line[-1][:160])
        if "queued" not in " ".join(line).lower():
            break
        time.sleep(6)
    print("done. the notebook keeps serving until the session cap.")


if __name__ == "__main__":
    main()
