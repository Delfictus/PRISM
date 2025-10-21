"""
Utility package for PRISM-AI governance tooling.

This module exposes helpers used by the automated execution scripts.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence


def vault_root() -> Path:
    """Return the absolute path to the PRISM-AI unified vault root."""
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """Return the repository root that hosts the unified vault."""
    return vault_root().parent


def _git(args: Sequence[str]) -> Optional[str]:
    """Run a git command relative to the repository root."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root()),
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


@lru_cache(maxsize=1)
def worktree_context() -> Dict[str, object]:
    """Describe the active worktree so automation can tailor outputs."""
    root = repo_root()
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]) or "UNKNOWN"
    commit = _git(["rev-parse", "HEAD"]) or "UNKNOWN"
    dirty_status = _git(["status", "--short"])
    return {
        "path": str(root),
        "name": root.name,
        "branch": branch,
        "commit": commit,
        "dirty": bool(dirty_status),
    }


__all__ = ["vault_root", "repo_root", "worktree_context"]
