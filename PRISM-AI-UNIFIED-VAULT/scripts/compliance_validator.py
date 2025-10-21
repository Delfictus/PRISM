#!/usr/bin/env python3
"""
Compliance validator for the PRISM-AI unified vault.

This tool enforces the Advanced Definition of Done (A-DoD) contract by
inspecting documentation, governance manifests, and execution artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from . import vault_root  # type: ignore
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import vault_root  # type: ignore


TASKS_PATH = vault_root() / "05-PROJECT-PLAN" / "tasks.json"
PROJECT_OVERVIEW_PATH = vault_root() / "PROJECT-OVERVIEW.md"
ALLOWED_STATUSES = {"pending", "in_progress", "blocked", "done"}

ADVANCED_KEYWORDS: Dict[str, Sequence[str]] = {
    "constitution": [
        "ADVANCED DELIVERY CONTRACT",
        "Advanced, Not Simplified",
        "Persistent kernels + work stealing",
        "Module-Specific Advanced Directions",
    ],
    "governance": [
        "AdvancedDoDGate",
        "GpuPatternGate",
        "RooflineGate",
        "AblationProofGate",
        "DeviceGuardGate",
    ],
    "implementation": [
        "ADVANCED IMPLEMENTATION BLUEPRINT",
        "Sparse Coloring Kernel Requirements",
        "Dense Path with WMMA/Tensor Cores",
        "Numerics & Reproducibility Hooks",
    ],
    "automation": [
        "ADVANCED A-DoD EXECUTION FLOW",
        "OUTPUT ARTIFACT CHECKLIST",
        "GOVERNANCE INTEGRATION",
    ],
}

ARTIFACT_PATHS: Dict[str, Path] = {
    "advanced_manifest": Path("artifacts/advanced_manifest.json"),
    "roofline": Path("reports/roofline.json"),
    "determinism": Path("reports/determinism_replay.json"),
    "ablation": Path("artifacts/ablation_report.json"),
    "protein": Path("reports/protein_auroc.json"),
    "device_caps": Path("device_caps.json"),
    "path_decision": Path("path_decision.json"),
    "feasibility": Path("feasibility.log"),
    "graph_capture": Path("reports/graph_capture.json"),
    "graph_exec": Path("reports/graph_exec.bin"),
    "determinism_manifest": Path("artifacts/determinism_manifest.json"),
    "selection_report": Path("artifacts/mec/M1/selection_report.json"),
}

CRITICAL_ARTIFACTS = {
    "advanced_manifest",
    "roofline",
    "determinism",
    "graph_capture",
    "graph_exec",
    "determinism_manifest",
    "selection_report",
}

META_REGISTRY_PATH = Path("meta/meta_flags.json")
META_SCHEMA_PATH = Path("telemetry/schema/meta_v1.json")
REQUIRED_META_FLAGS = {
    "meta_generation",
    "ontology_bridge",
    "free_energy_snapshots",
    "semantic_plasticity",
    "federated_meta",
    "meta_prod",
}

PLACEHOLDER_RULES: Dict[str, Dict[str, object]] = {
    "kernel_residency": {
        "tasks": ["P2-03", "P2-04"],
        "flag": "PRISM_OVERRIDE_KERNEL_RESIDENCY",
    },
    "performance_metrics": {
        "tasks": ["P2-07"],
        "flag": "PRISM_OVERRIDE_PERFORMANCE_METRICS",
    },
    "advanced_tactics": {
        "tasks": ["P2-04"],
        "flag": "PRISM_OVERRIDE_ADVANCED_TACTICS",
    },
    "algorithmic_advantage": {
        "tasks": ["P2-05", "M1-02"],
        "flag": "PRISM_OVERRIDE_ALGO_ADVANTAGE",
    },
    "determinism_replay": {
        "tasks": ["P1-01", "M1-03"],
        "flag": "PRISM_OVERRIDE_DETERMINISM_REPLAY",
    },
    "device_guards": {
        "tasks": ["P1-03"],
        "flag": "PRISM_OVERRIDE_DEVICE_GUARDS",
    },
    "telemetry_cuda_graph": {
        "tasks": ["P0-02", "P2-03"],
        "flag": "PRISM_OVERRIDE_CUDA_GRAPH_TELEMETRY",
    },
    "telemetry_persistent_kernel": {
        "tasks": ["P0-02", "P2-04"],
        "flag": "PRISM_OVERRIDE_PERSISTENT_KERNEL_TELEMETRY",
    },
    "telemetry_mixed_precision": {
        "tasks": ["P0-02"],
        "flag": "PRISM_OVERRIDE_MIXED_PRECISION_TELEMETRY",
    },
    "tactic_bitmap": {
        "tasks": ["P2-04"],
        "flag": "PRISM_OVERRIDE_TACTIC_BITMAP",
    },
    "ablation_transparency": {
        "tasks": ["P2-02"],
        "flag": "PRISM_OVERRIDE_ABLATION_TRANSPARENCY",
    },
}


@dataclass
class Finding:
    """Represents a compliance finding."""

    item: str
    status: str
    message: str
    severity: str = "INFO"
    evidence: Optional[str] = None


@dataclass
class ComplianceReport:
    """Aggregated compliance report."""

    timestamp: str
    strict: bool
    findings: List[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    @property
    def passed(self) -> bool:
        severities = {"BLOCKER": 3, "CRITICAL": 2, "WARNING": 1, "INFO": 0}
        return all(severities[f.severity] < 2 for f in self.findings if f.status != "PASS")

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "strict": self.strict,
            "passed": self.passed,
            "findings": [asdict(f) for f in self.findings],
        }


def load_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


_TASK_STATUS_CACHE: Optional[Dict[str, str]] = None


def task_status_map() -> Dict[str, str]:
    global _TASK_STATUS_CACHE
    if _TASK_STATUS_CACHE is not None:
        return _TASK_STATUS_CACHE

    data = load_json(TASKS_PATH)
    statuses: Dict[str, str] = {}
    if isinstance(data, dict):
        for phase in data.get("phases", []):
            for task in phase.get("tasks", []):
                task_id = task.get("id")
                if not task_id:
                    continue
                status = task.get("status", "pending")
                statuses[task_id] = str(status)
    _TASK_STATUS_CACHE = statuses
    return statuses


def module_state(module_key: str) -> Tuple[str, Dict[str, str]]:
    rule = PLACEHOLDER_RULES.get(module_key)
    if not rule:
        return "unknown", {}

    task_ids = rule.get("tasks", [])
    if not task_ids:
        return "unknown", {}

    statuses = task_status_map()
    state_map = {task_id: statuses.get(task_id, "missing") for task_id in task_ids}
    if state_map and all(status == "done" for status in state_map.values()):
        return "implemented", state_map

    return "pending", state_map


def add_placeholder_finding(
    report: ComplianceReport,
    module_key: str,
    item: str,
    message: str,
    default_severity: str,
) -> None:
    rule = PLACEHOLDER_RULES.get(module_key)
    if not rule:
        report.add(
            Finding(item=item, status="FAIL", severity=default_severity, message=message)
        )
        return

    state, task_map = module_state(module_key)
    if state != "implemented":
        pending_descriptions = ", ".join(
            f"{task_id}={status}" for task_id, status in task_map.items()
        ) or "untracked"
        report.add(
            Finding(
                item=item,
                status="WARN",
                severity="WARNING",
                message=f"{message} (placeholder accepted; pending tasks: {pending_descriptions})",
            )
        )
        return

    override_flag = rule.get("flag")
    if isinstance(override_flag, str):
        override_value = os.environ.get(override_flag)
        if override_value:
            report.add(
                Finding(
                    item=item,
                    status="OVERRIDE",
                    severity="WARNING",
                    message=(
                        f"{message} — TEMPORARY OVERRIDE VIA {override_flag}="
                        f"{override_value!r}. REMOVE THIS FLAG IMMEDIATELY AFTER RESTORING FULL CHECKS."
                    ),
                )
            )
            return

    report.add(
        Finding(item=item, status="FAIL", severity=default_severity, message=message)
    )


def check_keywords(report: ComplianceReport, name: str, path: Path, keywords: Sequence[str]) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        report.add(
            Finding(
                item=f"documentation:{name}",
                status="FAIL",
                severity="BLOCKER" if report.strict else "WARNING",
                message=f"Required document missing: {path}",
            )
        )
        return

    missing = [kw for kw in keywords if kw not in content]
    if missing:
        report.add(
            Finding(
                item=f"documentation:{name}",
                status="FAIL",
                severity="CRITICAL",
                message=f"Missing required sections: {', '.join(missing)}",
            )
        )
    else:
        report.add(
            Finding(
                item=f"documentation:{name}",
                status="PASS",
                severity="INFO",
                message="All advanced governance keywords present.",
            )
        )


def evaluate_advanced_manifest(report: ComplianceReport, path: Path) -> None:
    data = load_json(path)
    if data is None:
        report.add(
            Finding(
                item="artifacts:advanced_manifest",
                status="FAIL",
                severity="BLOCKER" if report.strict else "WARNING",
                message="Advanced manifest missing.",
            )
        )
        return

    # Kernel residency
    residency = data.get("kernel_residency", {})
    if not residency.get("gpu_resident", False):
        add_placeholder_finding(
            report,
            "kernel_residency",
            "a-dod:kernel_residency",
            "Kernel residency check failed (GPU residency not guaranteed).",
            "BLOCKER",
        )

    # Performance thresholds
    performance = data.get("performance", {})
    thresholds = {
        "occupancy": (0.60, "BLOCKER"),
        "sm_efficiency": (0.70, "BLOCKER"),
        "bandwidth": (0.60, "CRITICAL"),
        "flops": (0.40, "CRITICAL"),
        "p95_variance": (0.10, "CRITICAL"),
    }

    for key, (threshold, severity) in thresholds.items():
        value = performance.get(key)
        if value is None:
            add_placeholder_finding(
                report,
                "performance_metrics",
                f"a-dod:performance:{key}",
                f"Performance metric '{key}' missing.",
                "CRITICAL",
            )
            continue

        if key == "p95_variance":
            if value > threshold:
                report.add(
                    Finding(
                        item=f"a-dod:performance:{key}",
                        status="FAIL",
                        severity=severity,
                        message=f"P95 runtime variance {value:.3f} exceeds {threshold:.3f}.",
                    )
                )
        else:
            if value < threshold:
                report.add(
                    Finding(
                        item=f"a-dod:performance:{key}",
                        status="FAIL",
                        severity=severity,
                        message=f"{key} {value:.3f} below threshold {threshold:.3f}.",
                    )
                )

    # Complexity evidence
    tactics = data.get("advanced_tactics", [])
    if len(tactics) < 2:
        add_placeholder_finding(
            report,
            "advanced_tactics",
            "a-dod:advanced_tactics",
            f"Advanced tactic count {len(tactics)} < 2.",
            "CRITICAL",
        )

    # Algorithmic advantage
    algorithmic = data.get("algorithmic", {})
    if not algorithmic.get("improves_speed", False) or not algorithmic.get("improves_quality", False):
        add_placeholder_finding(
            report,
            "algorithmic_advantage",
            "a-dod:algorithmic_advantage",
            "Algorithmic advantage not demonstrated.",
            "CRITICAL",
        )

    # Determinism
    determinism = data.get("determinism", {})
    if not determinism.get("replay_passed", False):
        add_placeholder_finding(
            report,
            "determinism_replay",
            "a-dod:determinism_replay",
            "Determinism replay gate failed.",
            "BLOCKER",
        )

    # Device guards
    device = data.get("device", {})
    if not device.get("guard_passed", False):
        add_placeholder_finding(
            report,
            "device_guards",
            "a-dod:device_guards",
            "Device guard checks did not pass.",
            "CRITICAL",
        )

    telemetry = data.get("telemetry", {})
    if not telemetry.get("cuda_graph_captured", False):
        add_placeholder_finding(
            report,
            "telemetry_cuda_graph",
            "a-dod:telemetry:cuda_graph",
            "CUDA Graph capture telemetry not confirmed.",
            "CRITICAL",
        )
    if not telemetry.get("persistent_kernel_used", False):
        add_placeholder_finding(
            report,
            "telemetry_persistent_kernel",
            "a-dod:telemetry:persistent_kernel",
            "Persistent kernel telemetry not confirmed.",
            "CRITICAL",
        )
    if not telemetry.get("mixed_precision_policy", False):
        add_placeholder_finding(
            report,
            "telemetry_mixed_precision",
            "a-dod:telemetry:mixed_precision",
            "Mixed precision policy telemetry not confirmed.",
            "CRITICAL",
        )

    bitmap = data.get("tactic_bitmap", {})
    if not bitmap:
        add_placeholder_finding(
            report,
            "tactic_bitmap",
            "a-dod:tactic_bitmap",
            "Advanced tactic bitmap missing.",
            "CRITICAL",
        )
    else:
        missing = [name for name, enabled in bitmap.items() if not enabled]
        if missing:
            add_placeholder_finding(
                report,
                "tactic_bitmap",
                "a-dod:tactic_bitmap",
                f"Advanced tactics disabled or unreported: {', '.join(missing)}",
                "CRITICAL",
            )


def evaluate_artifact_presence(report: ComplianceReport, base: Path, allow_missing: bool) -> None:
    for name, rel_path in ARTIFACT_PATHS.items():
        path = base / rel_path
        if not path.exists():
            report.add(
                Finding(
                    item=f"artifact:{name}",
                    status="FAIL",
                    severity="BLOCKER" if (not allow_missing and name in CRITICAL_ARTIFACTS) else "WARNING",
                    message=f"Expected artifact missing: {path}",
                )
            )
        else:
            report.add(
                Finding(
                    item=f"artifact:{name}",
                    status="PASS",
                    severity="INFO",
                    message=f"Artifact present: {path}",
                )
            )


def compute_meta_merkle(records: Sequence[Dict[str, object]]) -> str:
    if not records:
        return hashlib.sha256(b"meta-empty").hexdigest()

    leaves = []
    for record in records:
        payload = json.dumps(record, separators=(",", ":")).encode("utf-8")
        leaves.append(hashlib.sha256(payload).digest())
    leaves.sort()

    working = leaves
    while len(working) > 1:
        next_level = []
        for idx in range(0, len(working), 2):
            left = working[idx]
            right = working[idx + 1] if idx + 1 < len(working) else working[idx]
            next_level.append(hashlib.sha256(left + right).digest())
        working = next_level

    return working[0].hex()


def evaluate_meta_contract(report: ComplianceReport, base: Path) -> None:
    schema_path = base / META_SCHEMA_PATH
    if not schema_path.exists():
        report.add(
            Finding(
                item="meta:schema",
                status="FAIL",
                severity="CRITICAL",
                message=f"Meta telemetry schema missing: {schema_path}",
            )
        )
    else:
        report.add(
            Finding(
                item="meta:schema",
                status="PASS",
                severity="INFO",
                message="Meta telemetry schema present.",
            )
        )

    registry_path = base / META_REGISTRY_PATH
    if not registry_path.exists():
        report.add(
            Finding(
                item="meta:registry",
                status="FAIL",
                severity="CRITICAL",
                message=f"Meta feature registry missing: {registry_path}",
            )
        )
        return

    data = load_json(registry_path)
    if data is None:
        report.add(
            Finding(
                item="meta:registry",
                status="FAIL",
                severity="CRITICAL",
                message="Meta feature registry unreadable.",
            )
        )
        return

    records = data.get("records")
    if not isinstance(records, list):
        report.add(
            Finding(
                item="meta:registry",
                status="FAIL",
                severity="CRITICAL",
                message="Meta feature registry malformed (records missing).",
            )
        )
        return

    observed_flags = {record.get("id") for record in records}
    missing = REQUIRED_META_FLAGS - observed_flags
    if missing:
        report.add(
            Finding(
                item="meta:registry_flags",
                status="FAIL",
                severity="CRITICAL",
                message=f"Meta registry missing flags: {', '.join(sorted(missing))}",
            )
        )
    else:
        report.add(
            Finding(
                item="meta:registry_flags",
                status="PASS",
                severity="INFO",
                message="All required meta feature flags present.",
            )
        )

    expected_root = compute_meta_merkle(records)
    actual_root = data.get("merkle_root")
    if actual_root != expected_root:
        report.add(
            Finding(
                item="meta:merkle",
                status="FAIL",
                severity="BLOCKER",
                message=f"Meta registry Merkle mismatch (expected {expected_root}, got {actual_root}).",
            )
        )
    else:
        report.add(
            Finding(
                item="meta:merkle",
                status="PASS",
                severity="INFO",
                message="Meta registry Merkle root verified.",
            )
        )

    invariants = data.get("invariant_snapshot", {})
    entropy = invariants.get("generation_entropy")
    if not entropy:
        report.add(
            Finding(
                item="meta:invariants",
                status="FAIL",
                severity="WARNING",
                message="Meta invariant snapshot missing entropy hash.",
            )
        )
    else:
        report.add(
            Finding(
                item="meta:invariants",
                status="PASS",
                severity="INFO",
                message="Meta invariant snapshot present.",
            )
        )

def evaluate_task_manifest(report: ComplianceReport) -> None:
    if not TASKS_PATH.exists():
        report.add(
            Finding(
                item="tasks:manifest",
                status="FAIL",
                severity="BLOCKER",
                message=f"Task manifest missing: {TASKS_PATH}",
            )
        )
        return

    try:
        data = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.add(
            Finding(
                item="tasks:manifest",
                status="FAIL",
                severity="BLOCKER",
                message=f"Invalid JSON in tasks manifest: {exc}",
            )
        )
        return

    phases = data.get("phases", [])
    if not phases:
        report.add(
            Finding(
                item="tasks:phases",
                status="FAIL",
                severity="CRITICAL",
                message="No phases defined in tasks manifest.",
            )
        )
        return

    invalid_status = []
    empty_phases = []
    for phase in phases:
        tasks = phase.get("tasks", [])
        if not tasks:
            empty_phases.append(phase.get("id", "<unknown>"))
            continue
        for task in tasks:
            status = task.get("status", "pending")
            if status not in ALLOWED_STATUSES:
                invalid_status.append((task.get("id", "<unknown>"), status))

    if empty_phases:
        report.add(
            Finding(
                item="tasks:empty_phases",
                status="FAIL",
                severity="WARNING",
                message=f"Phases missing tasks: {', '.join(empty_phases)}",
            )
        )

    if invalid_status:
        details = ", ".join(f"{tid}={status}" for tid, status in invalid_status)
        report.add(
            Finding(
                item="tasks:status",
                status="FAIL",
                severity="CRITICAL",
                message=f"Invalid task statuses detected: {details}",
            )
        )
    else:
        report.add(
            Finding(
                item="tasks:manifest",
                status="PASS",
                severity="INFO",
                message="Task manifest present with valid statuses.",
            )
        )


def check_project_overview(report: ComplianceReport) -> None:
    if not PROJECT_OVERVIEW_PATH.exists():
        report.add(
            Finding(
                item="documentation:project_overview",
                status="FAIL",
                severity="BLOCKER",
                message="PROJECT-OVERVIEW.md missing.",
            )
        )
        return

    content = PROJECT_OVERVIEW_PATH.read_text(encoding="utf-8")
    required_sections = ["PRISM-AI PROJECT OVERVIEW", "Phase Snapshot", "How to Track Progress"]
    missing = [section for section in required_sections if section not in content]

    if missing:
        report.add(
            Finding(
                item="documentation:project_overview",
                status="FAIL",
                severity="CRITICAL",
                message=f"Project overview missing sections: {', '.join(missing)}",
            )
        )
    else:
        report.add(
            Finding(
                item="documentation:project_overview",
                status="PASS",
                severity="INFO",
                message="Project overview present with required sections.",
            )
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate PRISM-AI A-DoD compliance.")
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail on any missing or insufficient artifact (default: False).",
    )
    parser.add_argument(
        "--allow-missing-artifacts",
        action="store_true",
        default=False,
        help="Downgrade missing artifacts to warnings (labs / bring-up).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON report.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    base = vault_root()
    report = ComplianceReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        strict=args.strict and not args.allow_missing_artifacts,
    )

    # Documentation keywords
    check_keywords(
        report,
        "constitution",
        base / "00-CONSTITUTION" / "IMPLEMENTATION-CONSTITUTION.md",
        ADVANCED_KEYWORDS["constitution"],
    )
    check_keywords(
        report,
        "governance",
        base / "01-GOVERNANCE" / "AUTOMATED-GOVERNANCE-ENGINE.md",
        ADVANCED_KEYWORDS["governance"],
    )
    check_keywords(
        report,
        "implementation",
        base / "02-IMPLEMENTATION" / "MODULE-INTEGRATION.md",
        ADVANCED_KEYWORDS["implementation"],
    )
    check_keywords(
        report,
        "automation",
        base / "03-AUTOMATION" / "AUTOMATED-EXECUTION.md",
        ADVANCED_KEYWORDS["automation"],
    )

    check_project_overview(report)
    evaluate_task_manifest(report)

    evaluate_artifact_presence(report, base, args.allow_missing_artifacts)
    evaluate_advanced_manifest(report, base / ARTIFACT_PATHS["advanced_manifest"])
    evaluate_meta_contract(report, base)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if report.passed:
        print("✅ A-DoD compliance checks passed (informational issues only).")
        return 0

    print("❌ A-DoD compliance checks failed. See findings below:\n")
    for finding in report.findings:
        status = f"[{finding.status}]"
        severity = f"({finding.severity})"
        print(f"- {status:<8} {severity:<12} {finding.item}: {finding.message}")
        if finding.evidence:
            print(f"    evidence: {finding.evidence}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
