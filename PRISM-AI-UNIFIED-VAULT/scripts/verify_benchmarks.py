#!/usr/bin/env python3
"""Verify benchmark artifacts against the manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def verify(manifest_path: Path) -> bool:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    ok = True
    for name, entry in artifacts.items():
        path = Path(entry["path"]).expanduser()
        expected = entry.get("sha256")
        if not path.exists():
            print(f"❌ Missing benchmark artifact: {path}")
            ok = False
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            print(f"❌ Hash mismatch for {name}\n   expected: {expected}\n   actual:   {digest}")
            ok = False
        else:
            print(f"✅ Verified {name}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify benchmark artifacts")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    return 0 if verify(args.manifest) else 1


if __name__ == "__main__":
    sys.exit(main())
