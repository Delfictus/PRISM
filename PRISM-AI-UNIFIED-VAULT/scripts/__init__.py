"""
Utility package for PRISM-AI governance tooling.

This module exposes helpers used by the automated execution scripts.
"""

from pathlib import Path


def vault_root() -> Path:
    """Return the absolute path to the PRISM-AI unified vault root."""
    return Path(__file__).resolve().parents[1]


__all__ = ["vault_root"]
