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
    from . import vault_root, worktree_context  # type: ignore
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import vault_root, worktree_context  # type: ignore
def load_metrics() -> Dict[str, object]:
    tasks_path = vault_root() / "05-PROJECT-PLAN" / "tasks.json"
    metrics: Dict[str, object] = {}
    if tasks_path.exists():
        with tasks_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        metrics.update(
            {
                "phase_count": len(data.get("phases", [])),
                "tasks_total": sum(len(phase.get("tasks", [])) for phase in data.get("phases", [])),
            }
        )
    lattice_path = vault_root() / "artifacts" / "mec" / "M3" / "lattice_report.json"
    if lattice_path.exists():
        try:
            report = json.loads(lattice_path.read_text(encoding="utf-8"))
            snapshot = report.get("snapshot", {})
            metrics["lattice_mode"] = snapshot.get("state", {}).get("mode", "unknown")
            metrics["lattice_stability"] = snapshot.get("stability")
        except json.JSONDecodeError:
            metrics.setdefault("lattice_mode", "unavailable")
    return metrics


def render() -> str:
    metrics = load_metrics()
    ctx = worktree_context()
    return (
        "# Governance Dashboard (Placeholder)\n"
        f"- worktree: {ctx.get('name', 'unknown')} ({ctx.get('branch', 'unknown')})\n"
        f"- phases tracked: {metrics.get('phase_count', 0)}\n"
        f"- total tasks: {metrics.get('tasks_total', 0)}\n"
        f"- reflexive mode: {metrics.get('lattice_mode', 'n/a')}\n"
        f"- lattice stability: {metrics.get('lattice_stability', 'n/a')}\n"
        "\n"
        "Full dashboard implementation arrives in Phase M6."
    )


def main() -> None:
    print(render())


if __name__ == "__main__":
    main()
