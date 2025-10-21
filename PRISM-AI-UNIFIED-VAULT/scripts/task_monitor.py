#!/usr/bin/env python3
"""
Task monitor for the PRISM-AI multi-phase implementation roadmap.

Features:
  * Reads 05-PROJECT-PLAN/tasks.json and prints phase/status summaries.
  * Optional compliance validation (invokes scripts/compliance_validator.py).
  * Optional watch mode for continuous reporting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from . import vault_root  # type: ignore
    from . import worktree_root  # type: ignore
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import vault_root  # type: ignore
    from scripts import worktree_root  # type: ignore


VAULT_ROOT = vault_root()
WORKTREE_ROOT = worktree_root()
TASKS_PATH = VAULT_ROOT / "05-PROJECT-PLAN" / "tasks.json"
COMPLIANCE_SCRIPT = VAULT_ROOT / "scripts" / "compliance_validator.py"
ALLOWED_STATUSES = {"pending", "in_progress", "blocked", "done"}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_tasks() -> List[Dict[str, object]]:
    if not TASKS_PATH.exists():
        raise FileNotFoundError(f"Task manifest missing: {TASKS_PATH}")
    with TASKS_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("phases", [])


def filter_tasks(phases: Iterable[Dict[str, object]], phase_filter: Optional[str], status_filter: Optional[str]) -> List[Dict[str, object]]:
    filtered = []
    for phase in phases:
        if phase_filter and phase["id"] != phase_filter and phase["name"] != phase_filter:
            continue
        phase_tasks = []
        for task in phase.get("tasks", []):
            status = task.get("status", "pending")
            if status not in ALLOWED_STATUSES:
                task["status"] = "pending"
                status = "pending"
            if status_filter and status != status_filter:
                continue
            phase_tasks.append(task)
        if phase_tasks:
            filtered.append({**phase, "tasks": phase_tasks})
    return filtered


def summarize(phases: Iterable[Dict[str, object]]) -> str:
    lines: List[str] = []
    for phase in phases:
        tasks = phase.get("tasks", [])
        counts = {status: 0 for status in ALLOWED_STATUSES}
        for task in tasks:
            counts[task["status"]] = counts.get(task["status"], 0) + 1

        total = len(tasks)
        lines.append(f"{phase['id']} · {phase['name']} ({total} tasks)")
        lines.append(f"  pending: {counts['pending']} | in_progress: {counts['in_progress']} | blocked: {counts['blocked']} | done: {counts['done']}")
        for task in tasks:
            lines.append(f"    [{task['status']}] {task['id']} – {task['title']} ({task['source']})")
        lines.append("")
    return "\n".join(lines).rstrip()


def run_compliance(strict: bool) -> int:
    cmd = ["python3", str(COMPLIANCE_SCRIPT)]
    if strict:
        cmd.append("--strict")
    result = subprocess.run(cmd, cwd=WORKTREE_ROOT, text=True)
    return result.returncode


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor PRISM-AI project tasks and compliance.")
    parser.add_argument("--phase", help="Filter by phase id or name.")
    parser.add_argument("--status", choices=sorted(ALLOWED_STATUSES), help="Filter by task status.")
    parser.add_argument("--run-compliance", action="store_true", help="Execute compliance validator after summary.")
    parser.add_argument("--strict", action="store_true", help="Use strict mode when running compliance validator.")
    parser.add_argument("--watch", type=int, metavar="SECONDS", help="Continuously display status every N seconds.")
    parser.add_argument("--once", action="store_true", help="Print a single summary (default).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    watch_interval = args.watch if args.watch and args.watch > 0 else None

    def loop_body() -> int:
        phases = load_tasks()
        filtered = filter_tasks(phases, args.phase, args.status)
        if not filtered:
            print("No tasks match the selected filters.")
        else:
            print(f"[{utc_timestamp()}] Task Summary")
            print("=" * 72)
            print(summarize(filtered))
        if args.run_compliance:
            print("\nRunning compliance validator…")
            code = run_compliance(args.strict)
            if code == 0:
                print("Compliance validator passed.")
            else:
                print(f"Compliance validator failed (exit code {code}).")
            return code
        return 0

    if watch_interval:
        try:
            while True:
                exit_code = loop_body()
                if exit_code != 0:
                    return exit_code
                time.sleep(watch_interval)
        except KeyboardInterrupt:
            print("\nStopped task monitor.")
            return 0
    else:
        return loop_body()


if __name__ == "__main__":
    sys.exit(main())
