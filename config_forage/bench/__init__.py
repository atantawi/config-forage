"""Benchmark harness: grid ground truth (cf-grid), method runs (cf-bench), analysis (cf-analyze)."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path


def git_state() -> dict:
    """{"git_sha", "dirty"} of the checkout holding this package: sha None outside git; dirty when a
    tracked file differs from HEAD (untracked files do not count)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        return {"git_sha": git("rev-parse", "HEAD") or None,
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "dirty": False}


def now() -> str:
    """UTC time, ISO 8601 to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")
