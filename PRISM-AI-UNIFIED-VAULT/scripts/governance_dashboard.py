#!/usr/bin/env python3
"""
Placeholder governance dashboard generator.

This stub keeps the automation contract satisfied until Phase M6 implements
the full dashboard. The script prints a simple status summary derived from the
task monitor, ensuring CI hooks that reference this file do not fail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

try:
    from . import vault_root  # type: ignore
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import vault_root  # type: ignore


def load_metrics() -> Dict[str, object]:
    tasks_path = vault_root() / "05-PROJECT-PLAN" / "tasks.json"
    if not tasks_path.exists():
        return {}
    with tasks_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "phase_count": len(data.get("phases", [])),
        "tasks_total": sum(len(phase.get("tasks", [])) for phase in data.get("phases", [])),
    }


def load_reflexive_summary() -> Dict[str, object]:
    lattice_path = vault_root() / "artifacts" / "mec" / "M3" / "lattice_report.json"
    if not lattice_path.exists():
        return {}
    try:
        payload = json.loads(lattice_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"mode": "unavailable", "alerts": ["invalid_lattice_report"]}

    return {
        "mode": payload.get("mode"),
        "entropy": payload.get("entropy"),
        "divergence": payload.get("divergence"),
        "temperature": payload.get("effective_temperature"),
        "alerts": payload.get("alerts", []),
    }


def render() -> str:
    metrics = load_metrics()
    reflexive = load_reflexive_summary()

    lines = [
        "# Governance Dashboard (Placeholder)",
        f"- phases tracked: {metrics.get('phase_count', 0)}",
        f"- total tasks: {metrics.get('tasks_total', 0)}",
    ]

    if reflexive:
        mode = reflexive.get("mode", "unknown")
        entropy = reflexive.get("entropy")
        divergence = reflexive.get("divergence")
        temperature = reflexive.get("temperature")
        lines.append(
            f"- reflexive mode: {mode}"
            + (
                f" (entropy {entropy:.3f}, divergence {divergence:.3f}, temp {temperature:.2f})"
                if isinstance(entropy, (int, float)) and isinstance(divergence, (int, float))
                else ""
            )
        )
        alerts = reflexive.get("alerts") or []
        if alerts:
            lines.append(f"  alerts: {', '.join(alerts)}")

    lines.append("")
    lines.append("Full dashboard implementation arrives in Phase M6.")
    return "\n".join(lines)


def main() -> None:
    print(render())


if __name__ == "__main__":
    main()
