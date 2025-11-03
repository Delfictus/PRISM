"""
Utility package for PRISM-AI governance tooling.

This module exposes helpers used by the automated execution scripts.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache()
def vault_root() -> Path:
    """Return the absolute path to the PRISM-AI unified vault root."""
    return Path(__file__).resolve().parents[1]


@lru_cache()
def worktree_root() -> Path:
    """Return the git worktree root that contains the unified vault."""
    base = vault_root().parent.resolve()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=base,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        candidate = Path(result.stdout.strip()).resolve()
        if candidate.exists():
            return candidate
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return base


__all__ = ["vault_root", "worktree_root"]
