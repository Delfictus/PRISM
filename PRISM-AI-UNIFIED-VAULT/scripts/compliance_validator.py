#!/usr/bin/env python3
"""
Compliance validator for the PRISM-AI unified vault.

This tool enforces the Advanced Definition of Done (A-DoD) contract by
inspecting documentation, governance manifests, and execution artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
    "lattice_report": Path("artifacts/mec/M3/lattice_report.json"),
    "lattice_manifest": Path("determinism/meta/lattice_manifest.json"),
}

CRITICAL_ARTIFACTS = {
    "advanced_manifest",
    "roofline",
    "determinism",
    "graph_capture",
    "graph_exec",
    "determinism_manifest",
    "lattice_report",
    "lattice_manifest",
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


def evaluate_advanced_manifest(
    report: ComplianceReport, path: Path, lattice_snapshot: Optional[Dict[str, object]] = None
) -> None:
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
        report.add(
            Finding(
                item="a-dod:kernel_residency",
                status="FAIL",
                severity="BLOCKER",
                message="Kernel residency check failed (GPU residency not guaranteed).",
            )
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
            report.add(
                Finding(
                    item=f"a-dod:performance:{key}",
                    status="FAIL",
                    severity="CRITICAL" if report.strict else "WARNING",
                    message=f"Performance metric '{key}' missing.",
                )
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
        report.add(
            Finding(
                item="a-dod:advanced_tactics",
                status="FAIL",
                severity="CRITICAL",
                message=f"Advanced tactic count {len(tactics)} < 2.",
            )
        )

    # Algorithmic advantage
    algorithmic = data.get("algorithmic", {})
    if not algorithmic.get("improves_speed", False) or not algorithmic.get("improves_quality", False):
        report.add(
            Finding(
                item="a-dod:algorithmic_advantage",
                status="FAIL",
                severity="CRITICAL",
                message="Algorithmic advantage not demonstrated."
            )
        )

    # Determinism
    determinism = data.get("determinism", {})
    if not determinism.get("replay_passed", False):
        report.add(
            Finding(
                item="a-dod:determinism_replay",
                status="FAIL",
                severity="BLOCKER",
                message="Determinism replay gate failed.",
            )
        )

    # Device guards
    device = data.get("device", {})
    if not device.get("guard_passed", False):
        report.add(
            Finding(
                item="a-dod:device_guards",
                status="FAIL",
                severity="CRITICAL",
                message="Device guard checks did not pass.",
            )
        )

    telemetry = data.get("telemetry", {})
    if not telemetry.get("cuda_graph_captured", False):
        report.add(
            Finding(
                item="a-dod:telemetry:cuda_graph",
                status="FAIL",
                severity="CRITICAL",
                message="CUDA Graph capture telemetry not confirmed.",
            )
        )
    if not telemetry.get("persistent_kernel_used", False):
        report.add(
            Finding(
                item="a-dod:telemetry:persistent_kernel",
                status="FAIL",
                severity="CRITICAL",
                message="Persistent kernel telemetry not confirmed.",
            )
        )
    if not telemetry.get("mixed_precision_policy", False):
        report.add(
            Finding(
                item="a-dod:telemetry:mixed_precision",
                status="FAIL",
                severity="CRITICAL",
                message="Mixed precision policy telemetry not confirmed.",
            )
        )

    bitmap = data.get("tactic_bitmap", {})
    if not bitmap:
        report.add(
            Finding(
                item="a-dod:tactic_bitmap",
                status="FAIL",
                severity="CRITICAL",
                message="Advanced tactic bitmap missing.",
            )
        )
    else:
        missing = [name for name, enabled in bitmap.items() if not enabled]
        if missing:
            report.add(
                Finding(
                    item="a-dod:tactic_bitmap",
                    status="FAIL",
                    severity="CRITICAL",
                    message=f"Advanced tactics disabled or unreported: {', '.join(missing)}",
                )
            )

    reflexive = data.get("reflexive", {})
    if not isinstance(reflexive, dict) or not reflexive:
        report.add(
            Finding(
                item="a-dod:reflexive",
                status="FAIL",
                severity="CRITICAL",
                message="Reflexive snapshot summary missing from advanced manifest.",
            )
        )
        return

    mode = reflexive.get("mode")
    if mode not in {"strict", "recovery", "exploration"}:
        report.add(
            Finding(
                item="a-dod:reflexive:mode",
                status="FAIL",
                severity="CRITICAL",
                message=f"Invalid reflexive mode reported: {mode!r}.",
            )
        )

    entropy = reflexive.get("entropy")
    divergence = reflexive.get("divergence")
    fingerprint = reflexive.get("lattice_fingerprint")

    if fingerprint is None:
        report.add(
            Finding(
                item="a-dod:reflexive:fingerprint",
                status="FAIL",
                severity="BLOCKER",
                message="Reflexive lattice fingerprint missing from advanced manifest.",
            )
        )
    else:
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            report.add(
                Finding(
                    item="a-dod:reflexive:fingerprint",
                    status="FAIL",
                    severity="BLOCKER",
                    message="Reflexive lattice fingerprint must be a 64-hex digest.",
                )
            )
        if lattice_snapshot and lattice_snapshot.get("fingerprint") != fingerprint:
            report.add(
                Finding(
                    item="a-dod:reflexive:fingerprint",
                    status="FAIL",
                    severity="BLOCKER",
                    message="Advanced manifest fingerprint does not match lattice snapshot.",
                )
            )

    if mode == "exploration" and divergence is not None and divergence >= 0.18:
        report.add(
            Finding(
                item="a-dod:reflexive:divergence",
                status="FAIL",
                severity="CRITICAL",
                message=f"Exploration mode divergence {divergence:.3f} exceeds strict cap 0.18.",
            )
        )
    if mode == "exploration" and entropy is not None and entropy <= 1.05:
        report.add(
            Finding(
                item="a-dod:reflexive:entropy",
                status="FAIL",
                severity="CRITICAL",
                message=f"Exploration mode entropy {entropy:.3f} below guardrail floor 1.05.",
            )
        )


def evaluate_lattice_snapshot(report: ComplianceReport, path: Path) -> Optional[Dict[str, object]]:
    data = load_json(path)
    if data is None:
        report.add(
            Finding(
                item="lattice:snapshot",
                status="FAIL",
                severity="BLOCKER",
                message=f"Lattice snapshot missing or unreadable: {path}",
            )
        )
        return None

    required = [
        "timestamp",
        "mode",
        "entropy",
        "divergence",
        "effective_temperature",
        "lattice_edge",
        "lattice",
        "fingerprint",
    ]
    missing = [field for field in required if field not in data]
    if missing:
        report.add(
            Finding(
                item="lattice:snapshot",
                status="FAIL",
                severity="BLOCKER",
                message=f"Lattice snapshot missing fields: {', '.join(missing)}",
            )
        )
        return None

    issues = False

    mode = data.get("mode")
    if mode not in {"strict", "recovery", "exploration"}:
        report.add(
            Finding(
                item="lattice:mode",
                status="FAIL",
                severity="CRITICAL",
                message=f"Invalid lattice mode {mode!r}.",
            )
        )
        issues = True

    lattice = data.get("lattice")
    edge = data.get("lattice_edge")
    if not isinstance(lattice, list) or not all(isinstance(row, list) for row in lattice):
        report.add(
            Finding(
                item="lattice:structure",
                status="FAIL",
                severity="CRITICAL",
                message="Lattice grid must be a 2D list.",
            )
        )
    else:
        if len(lattice) != edge or any(len(row) != edge for row in lattice):
            report.add(
                Finding(
                    item="lattice:structure",
                    status="FAIL",
                    severity="CRITICAL",
                    message=f"Lattice grid shape mismatch (edge={edge}, actual_rows={len(lattice)}).",
                )
            )
        issues = True

    fingerprint = data.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        report.add(
            Finding(
                item="lattice:fingerprint",
                status="FAIL",
                severity="BLOCKER",
                message="Lattice fingerprint must be 64 hex characters.",
            )
        )
        issues = True
    else:
        payload = {k: v for k, v in data.items() if k != "fingerprint"}
        computed = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if computed != fingerprint:
            report.add(
                Finding(
                    item="lattice:fingerprint",
                    status="FAIL",
                    severity="BLOCKER",
                    message="Lattice fingerprint mismatch (file contents tampered?).",
                )
            )
            issues = True

    alerts = data.get("alerts", [])
    blocking_alerts = {"reflexive_disabled", "divergence_cap_exceeded", "entropy_below_floor"}
    target_alerts = [alert for alert in alerts if str(alert).startswith("target_mode")]
    streak_alerts = [alert for alert in alerts if str(alert).startswith("strict_streak")]

    if any(alert in blocking_alerts for alert in alerts):
        report.add(
            Finding(
                item="lattice:alerts",
                status="FAIL",
                severity="BLOCKER",
                message=f"Blocking reflexive alert present: {alerts}",
            )
        )
        issues = True

    if target_alerts:
        report.add(
            Finding(
                item="lattice:alerts",
                status="FAIL",
                severity="BLOCKER",
                message=f"Target mode not achieved after retries: {', '.join(target_alerts)}",
            )
        )
        issues = True

    if streak_alerts:
        report.add(
            Finding(
                item="lattice:alerts",
                status="FAIL",
                severity="CRITICAL",
                message=f"Reflexive controller strict streak detected: {', '.join(streak_alerts)}",
            )
        )
        issues = True

    weights = data.get("distribution_weights")
    if isinstance(weights, list) and weights:
        total = sum(weights)
        if abs(total - 1.0) > 1e-3:
            report.add(
                Finding(
                    item="lattice:weights",
                    status="FAIL",
                    severity="WARNING",
                    message=f"Lattice distribution weights sum to {total:.4f} (expected ≈ 1.0).",
                )
            )
            issues = True

    if not issues:
        report.add(
            Finding(
                item="lattice:snapshot",
                status="PASS",
                severity="INFO",
                message="Lattice snapshot validated.",
            )
        )
    return data


def evaluate_lattice_manifest(
    report: ComplianceReport, path: Path, snapshot: Optional[Dict[str, object]]
) -> None:
    data = load_json(path)
    if data is None:
        report.add(
            Finding(
                item="lattice:manifest",
                status="FAIL",
                severity="BLOCKER",
                message=f"Lattice manifest missing or unreadable: {path}",
            )
        )
        return

    entries = data.get("snapshots")
    if not isinstance(entries, list) or not entries:
        report.add(
            Finding(
                item="lattice:manifest:snapshots",
                status="FAIL",
                severity="BLOCKER",
                message="Lattice manifest contains no snapshots.",
            )
        )
        return

    fingerprints = [
        entry.get("fingerprint")
        for entry in entries
        if isinstance(entry, dict) and entry.get("fingerprint")
    ]
    issues = False
    computed = hashlib.sha256("".join(sorted(fingerprints)).encode("utf-8")).hexdigest()
    if data.get("hash") != computed:
        report.add(
            Finding(
                item="lattice:manifest:hash",
                status="FAIL",
                severity="CRITICAL",
                message="Lattice manifest hash does not match snapshot fingerprints.",
            )
        )
        issues = True

    if snapshot:
        expected = snapshot.get("fingerprint")
        match = next(
            (entry for entry in entries if entry.get("fingerprint") == expected), None
        )
        if match is None:
            report.add(
                Finding(
                    item="lattice:manifest:fingerprint",
                    status="FAIL",
                    severity="BLOCKER",
                    message=f"Lattice manifest missing fingerprint {expected}.",
                )
            )
            issues = True
        else:
            if match.get("mode") != snapshot.get("mode"):
                report.add(
                    Finding(
                        item="lattice:manifest:mode",
                        status="FAIL",
                        severity="CRITICAL",
                        message="Lattice manifest mode disagrees with snapshot.",
                    )
                )
                issues = True

    if not issues:
        report.add(
            Finding(
                item="lattice:manifest",
                status="PASS",
                severity="INFO",
                message="Lattice manifest validated.",
            )
        )


def evaluate_determinism_manifest(
    report: ComplianceReport, path: Path, snapshot: Optional[Dict[str, object]]
) -> None:
    data = load_json(path)
    if data is None:
        report.add(
            Finding(
                item="determinism:manifest",
                status="FAIL",
                severity="BLOCKER",
                message=f"Determinism manifest missing or unreadable: {path}",
            )
        )
        return

    issues = False

    reflexive = data.get("reflexive")
    if not isinstance(reflexive, dict):
        report.add(
            Finding(
                item="determinism:manifest:reflexive",
                status="FAIL",
                severity="CRITICAL",
                message="Determinism manifest missing reflexive section.",
            )
        )
        issues = True
    else:
        mode = reflexive.get("mode")
        if mode not in {"strict", "recovery", "exploration"}:
            report.add(
                Finding(
                    item="determinism:manifest:mode",
                    status="FAIL",
                    severity="CRITICAL",
                    message=f"Invalid reflexive mode recorded in determinism manifest: {mode!r}.",
                )
            )
            issues = True
        fingerprint = reflexive.get("lattice_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            report.add(
                Finding(
                    item="determinism:manifest:fingerprint",
                    status="FAIL",
                    severity="BLOCKER",
                    message="Determinism manifest lattice fingerprint invalid.",
                )
            )
            issues = True
        elif snapshot and fingerprint != snapshot.get("fingerprint"):
            report.add(
                Finding(
                    item="determinism:manifest:fingerprint",
                    status="FAIL",
                    severity="BLOCKER",
                    message="Determinism manifest fingerprint does not match lattice snapshot.",
                )
            )
            issues = True

        alerts = reflexive.get("alerts", [])
        if any(alert in {"reflexive_disabled", "divergence_cap_exceeded"} for alert in alerts):
            report.add(
                Finding(
                    item="determinism:manifest:alerts",
                    status="FAIL",
                    severity="BLOCKER",
                    message=f"Determinism manifest recorded blocking reflexive alerts: {alerts}",
                )
            )
            issues = True

        if any(str(alert).startswith("target_mode") for alert in alerts):
            report.add(
                Finding(
                    item="determinism:manifest:alerts",
                    status="FAIL",
                    severity="BLOCKER",
                    message=f"Reflexive controller failed to reach configured mode: {alerts}",
                )
            )
            issues = True

        if any(str(alert).startswith("strict_streak") for alert in alerts):
            report.add(
                Finding(
                    item="determinism:manifest:alerts",
                    status="FAIL",
                    severity="CRITICAL",
                    message=f"Repeated strict-mode reflexive detections: {alerts}",
                )
            )
            issues = True

    if not issues:
        report.add(
            Finding(
                item="determinism:manifest",
                status="PASS",
                severity="INFO",
                message="Determinism manifest validated.",
            )
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
    lattice_snapshot = evaluate_lattice_snapshot(report, base / ARTIFACT_PATHS["lattice_report"])
    evaluate_advanced_manifest(
        report, base / ARTIFACT_PATHS["advanced_manifest"], lattice_snapshot
    )
    evaluate_lattice_manifest(
        report, base / ARTIFACT_PATHS["lattice_manifest"], lattice_snapshot
    )
    evaluate_determinism_manifest(
        report, base / ARTIFACT_PATHS["determinism_manifest"], lattice_snapshot
    )
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
