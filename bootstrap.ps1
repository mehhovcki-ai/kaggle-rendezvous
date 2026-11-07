# paws Kaggle provider bootstrap (Windows).
#
# Run from a checkout of kaggle-rendezvous, in PowerShell:
#   powershell -ExecutionPolicy Bypass -File bootstrap.ps1
#
# Installs Python 3 with winget if it is missing, then hands over to
# onboard.py, which builds its own venv, installs the kaggle CLI, walks you
# through login and enrollment.

$ErrorActionPreference = "Stop"

function Say($text) { Write-Host "`n== $text" -ForegroundColor Cyan }

function Find-Python {
    foreach ($candidate in @("py", "python", "python3")) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) {
            try {
                & $candidate -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" 2>$null
                if ($LASTEXITCODE -eq 0) { return $candidate }
            } catch { }
        }
    }
    return $null
}

$python = Find-Python

if (-not $python) {
    Say "installing Python 3 with winget"
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "winget is not available. Install Python from https://python.org and rerun." -ForegroundColor Red
        exit 1
    }
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    # The installer updates the machine PATH, but this session keeps the old
    # one. Look in the per-user install location before giving up.
    $candidates = Get-ChildItem "$env:LOCALAPPDATA\Programs\Python" -Filter "python.exe" -Recurse -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName
    if ($candidates) {
        $python = $candidates[0]
    } else {
        $python = Find-Python
    }
}

if (-not $python) {
    Write-Host "Python 3.9+ is still not available. Open a new terminal and rerun." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "./onboard.py")) {
    Write-Host "onboard.py not found. Run this from the kaggle-rendezvous checkout." -ForegroundColor Red
    exit 1
}

Say "starting onboarding"
& $python ./onboard.py @args
exit $LASTEXITCODE
