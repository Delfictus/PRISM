#!/usr/bin/env python3
"""
Ledger audit helper for cognitive blockchain entries.

Validates whether a given block or thought hash is present in the current
ledger snapshot. This stub reads from a future `artifacts/merkle` directory
and prints placeholder output until live integration is complete.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

try:
    from . import vault_root  # type: ignore
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import vault_root  # type: ignore


DEFAULT_MERKLE_ROOT = vault_root() / "artifacts" / "merkle"


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit cognitive ledger entries.")
    parser.add_argument("--thought", metavar="HASH", help="Thought DAG hash to verify.")
    parser.add_argument("--block", metavar="HASH", help="Cognitive block hash to verify.")
    parser.add_argument("--merkle-root", metavar="PATH", default=str(DEFAULT_MERKLE_ROOT), help="Ledger merkle anchor directory.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    merkle_root = Path(args.merkle_root)
    print("🧾 Cognitive Ledger Audit")
    print(f"Merkle anchor directory: {merkle_root}")
    if not merkle_root.exists():
        print("No merkle anchors present yet. Populate during Phase M0+.")
        return 1
    if args.thought:
        print(f"Thought hash {args.thought} verification: pending (stub).")
    if args.block:
        print(f"Block hash {args.block} verification: pending (stub).")
    if not args.thought and not args.block:
        print("Provide --thought or --block to audit specific entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
