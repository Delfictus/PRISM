#!/usr/bin/env python3
"""
Placeholder governance dashboard generator.

This stub keeps the automation contract satisfied until Phase M6 implements
the full dashboard. The script prints a simple status summary derived from the
task monitor, ensuring CI hooks that reference this file do not fail.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

try:
    from . import vault_root  # type: ignore
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import vault_root  # type: ignore


VAULT_ROOT = vault_root()


def load_metrics() -> Dict[str, object]:
    tasks_path = VAULT_ROOT / "05-PROJECT-PLAN" / "tasks.json"
    if not tasks_path.exists():
        return {}
    with tasks_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "phase_count": len(data.get("phases", [])),
        "tasks_total": sum(len(phase.get("tasks", [])) for phase in data.get("phases", [])),
    }


def render() -> str:
    metrics = load_metrics()
    return (
        "# Governance Dashboard (Placeholder)\n"
        f"- phases tracked: {metrics.get('phase_count', 0)}\n"
        f"- total tasks: {metrics.get('tasks_total', 0)}\n"
        "\n"
        "Full dashboard implementation arrives in Phase M6."
    )


def main() -> None:
    print(render())


if __name__ == "__main__":
    main()
