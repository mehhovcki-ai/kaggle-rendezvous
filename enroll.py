#!/usr/bin/env python3
"""Enroll a Kaggle account as a paws model-proxy notebook.

Generates a single-cell notebook with the rendezvous URL and key baked into
the cell, pushes it to Kaggle and starts it. Re-running is safe: the kernel
id is derived from the username, so a second run updates and restarts the
same notebook instead of creating another one.

Usage (from a checkout of kaggle-rendezvous):

    python enroll.py --key beta --username kirill

Kaggle authentication: reads ~/.kaggle/kaggle.json. If that file is missing
and no --kaggle-key was given, the script stops and tells you to authenticate
first. You can also pass credentials explicitly:

    python enroll.py --key beta --username kirill \
        --kaggle-user kirill --kaggle-key <kaggle api key>

Options:
    --key           rendezvous key the notebook presents on /connect
    --username      pawsapi/kaggle username; used for the kernel id
    --kaggle-user   Kaggle account name (defaults to KAGGLE_USERNAME)
    --kaggle-key    Kaggle API key (defaults to ~/.kaggle/kaggle.json)
    --slug          kernel slug (defaults to paws-bridge-<username>)
    --url           rendezvous websocket URL
    --private       keep the notebook private (default: on)
    --dry-run       write the notebook locally, do not push
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    """Load a source file from the checkout, with a clear error if missing."""
    path = HERE / name
    if not path.is_file():
        sys.exit(f"missing {name} next to enroll.py (run from the repo checkout)")
    return path.read_text()


def build_cell(url: str, key: str) -> str:
    """Assemble the single notebook cell: install, config, proxy, bridge, run."""
    proxy = read_source("proxy_cell.py")
    bridge = read_source("bridge_cell.py")
    parts = [
        INSTALL,
        CONFIG.format(url=url, key=key),
        "# --- proxy ---------------------------------------------------------------\n",
        proxy,
        "\n\n# --- bridge --------------------------------------------------------------\n",
        bridge,
        RUNNER,
    ]
    return "".join(parts)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def notebook(cell: str) -> dict:
    """A notebook with exactly one code cell."""
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


def ensure_kaggle_auth(args: argparse.Namespace) -> None:
    """Fail early with a human message when Kaggle has no credentials."""
    if args.kaggle_key:
        if not args.kaggle_user:
            sys.exit("--kaggle-key needs --kaggle-user as well")
        os.environ["KAGGLE_USERNAME"] = args.kaggle_user
        os.environ["KAGGLE_KEY"] = args.kaggle_key
        return
    config = Path.home() / ".kaggle" / "kaggle.json"
    if not config.is_file() and not (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    ):
        sys.exit(
            "Kaggle is not authenticated.\n"
            "Run `kaggle auth login` first, or put your API token in "
            "~/.kaggle/kaggle.json, or pass --kaggle-user/--kaggle-key."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True, help="rendezvous key")
    parser.add_argument("--username", required=True, help="pawsapi username")
    parser.add_argument("--kaggle-user", default=None)
    parser.add_argument("--kaggle-key", default=None)
    parser.add_argument("--slug", default=None)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--private", dest="private", action="store_true", default=True)
    parser.add_argument("--public", dest="private", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=None, help="output directory")
    args = parser.parse_args()

    slug = args.slug or f"paws-bridge-{slugify(args.username)}"
    cell = build_cell(args.url, args.key)

    out = Path(args.out or f"/tmp/{slug}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(notebook(cell), indent=1))

    owner = args.kaggle_user or os.environ.get("KAGGLE_USERNAME")
    if not owner and not args.dry_run:
        ensure_kaggle_auth(args)
        try:
            with open(Path.home() / ".kaggle" / "kaggle.json") as fh:
                owner = json.load(fh).get("username")
        except Exception:
            sys.exit("could not read the Kaggle username from ~/.kaggle/kaggle.json")
    owner = owner or args.username

    (out / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata(owner, slug, args.private), indent=2)
    )
    print(f"wrote {out}/notebook.ipynb ({len(cell.splitlines())} lines, 1 cell)")
    print(f"kernel id: {owner}/{slug}  private={args.private}")

    if args.dry_run:
        print("dry run: not pushing")
        return

    ensure_kaggle_auth(args)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print("pushing kernel...")
    api.kernels_push(str(out))

    kernel = f"{owner}/{slug}"
    url = f"https://www.kaggle.com/code/{kernel}"
    print(f"started: {url}")

    # Report the first status that is not queued, so the shell exit means
    # something. The run itself keeps going after this script ends.
    for _ in range(10):
        try:
            status = api.kernels_status(kernel)
            state = getattr(status, "status", None) or str(status)
            print(f"status: {state}")
            if state and "queued" not in str(state).lower():
                break
        except Exception as exc:
            print(f"status check failed: {exc}")
        time.sleep(6)

    print("done. the notebook keeps serving until the session cap.")


if __name__ == "__main__":
    main()
