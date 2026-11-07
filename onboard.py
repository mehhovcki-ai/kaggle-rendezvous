#!/usr/bin/env python3
"""Onboard a new paws Kaggle provider in one command.

Does everything except creating the Kaggle account and adding a phone number:

  1. builds a private venv under ~/.paws-bridge with the kaggle CLI
  2. checks that Kaggle is authenticated, and runs `kaggle auth login` if not
  3. asks for your pawsapi username and the key from the web UI
  4. hands both to enroll.py, which creates and starts the notebook

Usage (from a checkout of kaggle-rendezvous):

    python onboard.py

    python onboard.py --dry-run     set everything up, stop before login
    python onboard.py --key ... --username ...   skip the prompts

The venv exists so this works on a machine with any Python and no global
pip permissions. enroll.py shells out to `kaggle`, and this puts the venv's
bin directory first on PATH for that subprocess.
"""

from __future__ import annotations

import argparse
import getpass
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = Path.home() / ".paws-bridge"
VENV = HOME / "venv"
IS_WINDOWS = os.name == "nt"


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def venv_bin() -> Path:
    return VENV / ("Scripts" if IS_WINDOWS else "bin")


def kaggle_exe() -> Path:
    return venv_bin() / ("kaggle.exe" if IS_WINDOWS else "kaggle")


def env_with_venv() -> dict:
    env = os.environ.copy()
    env["PATH"] = str(venv_bin()) + os.pathsep + env.get("PATH", "")
    return env


def step(text: str) -> None:
    print(f"\n== {text}")


def ensure_venv() -> None:
    step("preparing the environment")
    if not venv_python().exists():
        print(f"creating venv at {VENV}")
        venv.EnvBuilder(with_pip=True).create(VENV)
    print("installing kaggle + jupytext (first run takes a minute)")
    subprocess.run(
        [str(venv_python()), "-m", "pip", "install", "-q", "--upgrade", "pip"],
        check=False,
    )
    result = subprocess.run(
        [str(venv_python()), "-m", "pip", "install", "-q", "kaggle", "jupytext"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print((result.stdout or "") + (result.stderr or ""), file=sys.stderr)
        sys.exit("could not install the kaggle CLI")
    print(f"kaggle: {kaggle_exe()}")


def authenticated(env: dict) -> bool:
    """A real API call is the only reliable check; `config view` lies."""
    result = subprocess.run(
        [str(kaggle_exe()), "kernels", "list", "-m", "--page-size", "1"],
        capture_output=True, text=True, env=env,
    )
    if result.returncode == 0:
        return True
    text = ((result.stdout or "") + (result.stderr or "")).lower()
    return "authenticat" not in text and "401" not in text and "login" not in text


def login(env: dict) -> None:
    step("kaggle login")
    if authenticated(env):
        print("already authenticated")
        return
    print("a browser window will open. sign in and approve the CLI.")
    result = subprocess.run(
        [str(kaggle_exe()), "auth", "login"], env=env,
    )
    if result.returncode != 0 or not authenticated(env):
        sys.exit(
            "kaggle login did not complete. run it manually:\n"
            f"  {kaggle_exe()} auth login"
        )
    print("authenticated")


def collect(args: argparse.Namespace) -> tuple[str, str]:
    step("pawsapi details")
    username = args.username or input("pawsapi username: ").strip()
    if not username:
        sys.exit("a pawsapi username is required")
    key = args.key or getpass.getpass(
        "pawsapi key (from the web UI, input hidden): "
    ).strip()
    if not key:
        sys.exit("a pawsapi key is required")
    return username, key


def run_enroll(username: str, key: str, env: dict, extra: list[str]) -> int:
    step("enrolling")
    enroll = HERE / "enroll.py"
    if not enroll.is_file():
        sys.exit(f"{enroll} not found; run this from the kaggle-rendezvous checkout")
    cmd = [
        sys.executable, str(enroll),
        "--key", key,
        "--username", username,
        *extra,
    ]
    return subprocess.run(cmd, env=env).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default=None)
    parser.add_argument("--key", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="stop before login; just prepare the environment")
    parser.add_argument("--no-run", action="store_true",
                        help="create the task but do not start the notebook")
    args = parser.parse_args()

    print(f"platform: {platform.system()} {platform.machine()}")
    ensure_venv()
    env = env_with_venv()

    if args.dry_run:
        print("\ndry run complete. rerun without --dry-run to log in and enroll.")
        return

    login(env)
    username, key = collect(args)
    extra = ["--no-run"] if args.no_run else []
    code = run_enroll(username, key, env, extra)
    if code != 0:
        sys.exit(code)
    print("\ndone. the notebook keeps serving until the Kaggle session cap.")


if __name__ == "__main__":
    main()
