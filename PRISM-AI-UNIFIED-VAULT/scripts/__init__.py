"""
Utility package for PRISM-AI governance tooling.

This module exposes helpers used by the automated execution scripts.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _resolve_path(candidate: str) -> Path:
    """Expand and resolve a filesystem path override."""
    return Path(candidate).expanduser().resolve()


@lru_cache()
def vault_root() -> Path:
    """Return the absolute path to the PRISM-AI unified vault root."""
    override = os.environ.get("PRISM_VAULT_ROOT")
    if override:
        return _resolve_path(override)
    return Path(__file__).resolve().parents[1]


@lru_cache()
def repo_root() -> Path:
    """Return the repository root accommodating multi-worktree setups."""
    for key in ("PRISM_REPO_ROOT", "PRISM_WORKTREE_ROOT"):
        override = os.environ.get(key)
        if override:
            return _resolve_path(override)
    return vault_root().parent


__all__ = ["vault_root", "repo_root"]
