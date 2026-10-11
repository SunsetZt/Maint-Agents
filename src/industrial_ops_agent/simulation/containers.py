"""Shared container inspection for project experiment launchers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def container_exists(root: Path, name: str) -> bool:
    """Inspect one named container without starting or removing it."""
    completed = subprocess.run(
        ["docker", "container", "inspect", name],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0
