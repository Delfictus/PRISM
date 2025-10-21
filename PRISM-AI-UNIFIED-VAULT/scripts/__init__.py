"""
Utility package for PRISM-AI governance tooling.

This module exposes helpers used by the automated execution scripts.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "PRISM_VAULT_ROOT"
_DEFAULT_VAULT_ROOT = Path(__file__).resolve().parents[1]


def vault_root() -> Path:
    """Return the absolute path to the PRISM-AI unified vault root."""
    override = os.environ.get(_ENV_VAR)
    if override:
        candidate = Path(override).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(
                f"Configured PRISM vault override at {_ENV_VAR}={override} does not exist."
            )
        return candidate.resolve()
    return _DEFAULT_VAULT_ROOT


__all__ = ["vault_root"]
