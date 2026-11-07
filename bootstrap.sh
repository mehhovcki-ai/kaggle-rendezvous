#!/bin/sh
# paws Kaggle provider bootstrap (Linux + macOS).
#
# Run from a checkout of kaggle-rendezvous:
#   sh bootstrap.sh
#
# Installs Python 3 if it is missing, then hands over to onboard.py, which
# builds its own venv, installs the kaggle CLI, walks you through login and
# enrollment. Windows: use bootstrap.ps1 instead.

set -eu

say() { printf '\n== %s\n' "$1"; }

have() { command -v "$1" >/dev/null 2>&1; }

python_ok() {
    have python3 || return 1
    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null
}

install_python() {
    say "installing Python 3"
    os=$(uname -s 2>/dev/null || echo unknown)
    if [ "$os" = "Darwin" ]; then
        if have brew; then
            brew install python
        else
            echo "Homebrew is not installed. Install it from https://brew.sh and rerun." >&2
            exit 1
        fi
    elif have apt-get; then
        sudo apt-get update -y && sudo apt-get install -y python3 python3-venv
    elif have dnf; then
        sudo dnf install -y python3
    elif have pacman; then
        sudo pacman -Sy --noconfirm python
    elif have zypper; then
        sudo zypper install -y python3
    elif have apk; then
        sudo apk add --no-cache python3
    else
        echo "Could not detect a package manager. Install Python 3.9+ and rerun." >&2
        exit 1
    fi
}

if ! python_ok; then
    install_python
fi

if ! python_ok; then
    echo "Python 3.9+ still not available after install. Check your PATH." >&2
    exit 1
fi

if [ ! -f "./onboard.py" ]; then
    echo "onboard.py not found. Run this from the kaggle-rendezvous checkout." >&2
    exit 1
fi

say "starting onboarding"
exec python3 ./onboard.py "$@"
