#!/usr/bin/env python3
"""
Compliance validator for the PRISM-AI unified vault.

This tool enforces the Advanced Definition of Done (A-DoD) contract by
inspecting documentation, governance manifests, and execution artifacts.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
PROJECT_OVERVIEW_PATH = VAULT_ROOT / "PROJECT-OVERVIEW.md"
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
}

CRITICAL_ARTIFACTS = {
    "advanced_manifest",
    "roofline",
    "determinism",
    "graph_capture",
    "graph_exec",
    "determinism_manifest",
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

FEDERATION_PLAN_PATH = Path("artifacts/mec/M5/federated_plan.md")
FEDERATION_SUMMARY_PATH = Path("artifacts/mec/M5/simulations/epoch_summary.json")
FEDERATION_LEDGER_DIR = Path("artifacts/mec/M5/ledger")
FEDERATION_SCENARIO_DIR = Path("artifacts/mec/M5/scenarios")
FEDERATION_HMAC_KEY = b"PRISM-FEDERATED-HMAC-KEY-123456"


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
    worktree_root: str
    vault_root: str
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
            "worktree_root": self.worktree_root,
            "vault_root": self.vault_root,
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


def fnv1a64(value: str) -> int:
    offset = 0xCBF29CE484222325
    prime = 0x100000001B3
    hash_val = offset
    for byte in value.encode("utf-8"):
        hash_val ^= byte
        hash_val = (hash_val * prime) & 0xFFFFFFFFFFFFFFFF
    return hash_val


def compute_ledger_merkle(entries: Sequence[Dict[str, object]]) -> str:
    if not entries:
        return f"{fnv1a64('ledger-empty'):016x}"

    leaves = []
    for entry in entries:
        node = entry.get("node_id", "")
        anchor = entry.get("anchor_hash", "")
        leaves.append(f"{fnv1a64(f'{node}:{anchor}'):016x}")
    leaves.sort()

    level = leaves
    while len(level) > 1:
        next_level: List[str] = []
        for idx in range(0, len(level), 2):
            left = level[idx]
            right = level[idx + 1] if idx + 1 < len(level) else level[idx]
            next_level.append(f"{fnv1a64(left + right):016x}")
        level = next_level
    return level[0]


def compute_summary_digest(roots: Sequence[str]) -> str:
    if not roots:
        return f"{fnv1a64('summary-empty'):016x}"

    level = sorted(f"{fnv1a64(root):016x}" for root in roots)
    while len(level) > 1:
        next_level: List[str] = []
        for idx in range(0, len(level), 2):
            left = level[idx]
            right = level[idx + 1] if idx + 1 < len(level) else level[idx]
            next_level.append(f"{fnv1a64(left + right):016x}")
        level = next_level
    return level[0]


def sign_digest(digest: str) -> str:
    mac = hmac.new(FEDERATION_HMAC_KEY, digest.encode("utf-8"), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode("ascii")


def verify_signature(digest: str, signature: str) -> bool:
    expected = sign_digest(digest)
    return hmac.compare_digest(expected, signature)


def verify_summary_signature(roots: Sequence[str], signature: str) -> bool:
    digest = compute_summary_digest(roots)
    return verify_signature(digest, signature)


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


def evaluate_federated_artifacts(report: ComplianceReport, base: Path) -> None:
    plan_path = base / FEDERATION_PLAN_PATH
    if not plan_path.exists():
        report.add(
            Finding(
                item="federation:plan",
                status="FAIL",
                severity="CRITICAL",
                message=f"Federation execution plan missing: {plan_path}",
            )
        )
    else:
        report.add(
            Finding(
                item="federation:plan",
                status="PASS",
                severity="INFO",
                message="Federation plan present.",
            )
        )

    summary_path = base / FEDERATION_SUMMARY_PATH
    summary = load_json(summary_path)
    if summary is None:
        report.add(
            Finding(
                item="federation:summary",
                status="FAIL",
                severity="CRITICAL",
                message=f"Federated simulation summary missing or invalid: {summary_path}",
            )
        )
        return

    epochs = summary.get("epochs")
    epoch_count = summary.get("epoch_count")
    declared_merkle: Dict[int, str] = {}
    merkle_roots: List[str] = []
    if not isinstance(epochs, list) or not epochs:
        report.add(
            Finding(
                item="federation:summary:epochs",
                status="FAIL",
                severity="CRITICAL",
                message="Federated summary must include a non-empty list of epochs.",
            )
        )
    else:
        if epoch_count is not None and epoch_count != len(epochs):
            report.add(
                Finding(
                    item="federation:summary:epoch_count",
                    status="FAIL",
                    severity="CRITICAL",
                    message=f"Epoch count mismatch (declared {epoch_count}, actual {len(epochs)}).",
                )
            )
        else:
            report.add(
                Finding(
                    item="federation:summary:epoch_count",
                    status="PASS",
                    severity="INFO",
                    message=f"Federated summary reports {len(epochs)} epochs.",
                )
            )

        for entry in epochs:
            if not isinstance(entry, dict):
                continue
            epoch_value = entry.get("epoch")
            merkle_value = entry.get("ledger_merkle")
            signature_value = entry.get("signature")
            if not isinstance(epoch_value, int) or merkle_value is None:
                report.add(
                    Finding(
                        item="federation:summary:ledger_merkle",
                        status="FAIL",
                        severity="CRITICAL",
                        message="Federated summary must provide ledger_merkle per epoch.",
                    )
                )
                declared_merkle.clear()
                merkle_roots.clear()
                break
            merkle_str = str(merkle_value)
            declared_merkle[epoch_value] = merkle_str
            merkle_roots.append(merkle_str)
            if signature_value is None or str(signature_value) != merkle_str:
                report.add(
                    Finding(
                        item="federation:summary:signature",
                        status="FAIL",
                        severity="CRITICAL",
                        message=f"Summary signature mismatch for epoch {epoch_value}.",
                    )
                )

    if merkle_roots:
        declared_summary_sig = summary.get("summary_signature")
        if not isinstance(declared_summary_sig, str):
            report.add(
                Finding(
                    item="federation:summary:signature",
                    status="FAIL",
                    severity="CRITICAL",
                    message="Summary signature missing or invalid.",
                )
            )
        elif not verify_summary_signature(merkle_roots, declared_summary_sig):
            report.add(
                Finding(
                    item="federation:summary:signature",
                    status="FAIL",
                    severity="CRITICAL",
                    message="Summary signature mismatch.",
                )
            )
        else:
            report.add(
                Finding(
                    item="federation:summary:signature",
                    status="PASS",
                    severity="INFO",
                    message="Summary signature verified.",
                )
            )

    ledger_dir = base / FEDERATION_LEDGER_DIR
    if not ledger_dir.exists():
        report.add(
            Finding(
                item="federation:ledger",
                status="FAIL",
                severity="CRITICAL",
                message=f"Federated ledger directory missing: {ledger_dir}",
            )
        )
        return

    ledger_files = sorted(p for p in ledger_dir.rglob("epoch_*.json") if p.is_file())
    if not ledger_files:
        report.add(
            Finding(
                item="federation:ledger",
                status="FAIL",
                severity="CRITICAL",
                message="Federated ledger directory contains no epoch files.",
            )
        )
        return

    expected_epochs = set(declared_merkle.keys())
    if not expected_epochs and isinstance(epochs, list):
        for entry in epochs:
            if isinstance(entry, dict):
                val = entry.get("epoch")
                if isinstance(val, int):
                    expected_epochs.add(val)

    found_epochs = set()
    recomputed_merkle: Dict[int, str] = {}
    for path in ledger_files:
        parts = path.stem.split("_")
        try:
            epoch = int(parts[-1])
            found_epochs.add(epoch)
        except ValueError:
            report.add(
                Finding(
                    item="federation:ledger:naming",
                    status="FAIL",
                    severity="WARNING",
                    message=f"Ledger file has unexpected name format: {path.name}",
                )
            )
            continue

        data = load_json(path)
        if data is None:
            report.add(
                Finding(
                    item="federation:ledger:read",
                    status="FAIL",
                    severity="CRITICAL",
                    message=f"Ledger file unreadable: {path}",
                )
            )
            continue
        entries = data.get("entries", [])
        if not isinstance(entries, list) or not entries:
            report.add(
                Finding(
                    item="federation:ledger:entries",
                    status="FAIL",
                    severity="CRITICAL",
                    message=f"Ledger file missing entries: {path}",
                )
            )
            continue
        computed_root = compute_ledger_merkle(entries)
        file_root = data.get("merkle_root")
        if file_root != computed_root:
            report.add(
                Finding(
                    item="federation:ledger:merkle",
                    status="FAIL",
                    severity="CRITICAL",
                    message=f"Ledger merkle mismatch for epoch {epoch}: expected {computed_root}, found {file_root}",
                )
            )
        recomputed_merkle[epoch] = computed_root

    if expected_epochs:
        missing_epochs = sorted(expected_epochs - found_epochs)
        if missing_epochs:
            report.add(
                Finding(
                    item="federation:ledger:coverage",
                    status="FAIL",
                    severity="CRITICAL",
                    message=f"Ledger missing epochs: {', '.join(map(str, missing_epochs))}",
                )
            )
        else:
            report.add(
                Finding(
                    item="federation:ledger:coverage",
                    status="PASS",
                    severity="INFO",
                    message="Ledger contains entries for all simulated epochs.",
                )
            )
    else:
        report.add(
            Finding(
                item="federation:ledger:coverage",
                status="PASS",
                severity="INFO",
                message="Ledger files present; summary did not declare explicit epochs.",
            )
        )

    scenario_dir = base / FEDERATION_SCENARIO_DIR
    if scenario_dir.exists():
        for scenario in sorted(p for p in scenario_dir.glob('*.json')):
            label = scenario.stem
            summary_ok = True
            if label == 'baseline':
                summary_path = base / FEDERATION_SUMMARY_PATH
                ledger_path = ledger_dir
                expected_label = 'default'
            else:
                summary_path = (base / FEDERATION_SUMMARY_PATH).parent / f'epoch_summary_{label}.json'
                ledger_path = ledger_dir / label
                expected_label = label

            summary_data = load_json(summary_path) if summary_path.exists() else None
            if summary_data is None:
                report.add(
                    Finding(
                        item=f'federation:scenario:{label}:summary',
                        status='FAIL',
                        severity='CRITICAL',
                        message=f'Missing or invalid federated summary for scenario {label}: {summary_path}',
                    )
                )
                summary_ok = False
            else:
                reported_label = summary_data.get('label')
                if reported_label != expected_label:
                    report.add(
                        Finding(
                            item=f'federation:scenario:{label}:label',
                            status='FAIL',
                            severity='CRITICAL',
                            message=(
                                'Summary label mismatch for scenario '
                                f"{label}: expected {expected_label}, found {reported_label}"
                            ),
                        )
                    )
                    summary_ok = False

                epochs_data = summary_data.get('epochs', [])
                scenario_roots: List[str] = []
                if isinstance(epochs_data, list):
                    for entry in epochs_data:
                        if not isinstance(entry, dict):
                            continue
                        merkle = entry.get('ledger_merkle')
                        signature = entry.get('signature')
                        if merkle is None or signature is None or not verify_signature(str(merkle), str(signature)):
                            report.add(
                                Finding(
                                    item=f'federation:scenario:{label}:epoch_signature',
                                    status='FAIL',
                                    severity='CRITICAL',
                                    message=f'Scenario {label} epoch entry missing signature alignment.',
                                )
                            )
                            scenario_roots.clear()
                            summary_ok = False
                            break
                        scenario_roots.append(str(merkle))

                if scenario_roots:
                    declared_sig = summary_data.get('summary_signature')
                    if not isinstance(declared_sig, str) or not verify_summary_signature(scenario_roots, declared_sig):
                        report.add(
                            Finding(
                                item=f'federation:scenario:{label}:summary_signature',
                                status='FAIL',
                                severity='CRITICAL',
                                message=f'Scenario {label} summary signature mismatch.',
                            )
                        )
                        summary_ok = False
                    else:
                        report.add(
                            Finding(
                                item=f'federation:scenario:{label}:summary_signature',
                                status='PASS',
                                severity='INFO',
                                message=f'Scenario {label} summary signature verified.',
                            )
                        )

                if summary_ok:
                    report.add(
                        Finding(
                            item=f'federation:scenario:{label}:summary',
                            status='PASS',
                            severity='INFO',
                            message=f'Federated summary present for scenario {label}.',
                        )
                    )

            if not ledger_path.exists():
                report.add(
                    Finding(
                        item=f'federation:scenario:{label}:ledger',
                        status='FAIL',
                        severity='CRITICAL',
                        message=f'Missing federated ledger directory for scenario {label}: {ledger_path}',
                    )
                )
            else:
                report.add(
                    Finding(
                        item=f'federation:scenario:{label}:ledger',
                        status='PASS',
                        severity='INFO',
                        message=f'Federated ledger present for scenario {label}.',
                    )
                )

    if declared_merkle and recomputed_merkle:
        mismatched = {
            epoch: (declared_merkle.get(epoch), recomputed_merkle.get(epoch))
            for epoch in declared_merkle
            if declared_merkle.get(epoch) != recomputed_merkle.get(epoch)
        }
        if mismatched:
            detail = ", ".join(
                f"{epoch}: declared={declared} actual={actual}"
                for epoch, (declared, actual) in mismatched.items()
            )
            report.add(
                Finding(
                    item="federation:merkle_consistency",
                    status="FAIL",
                    severity="CRITICAL",
                    message=f"Summary merkle mismatch detected: {detail}",
                )
            )
        else:
            report.add(
                Finding(
                    item="federation:merkle_consistency",
                    status="PASS",
                    severity="INFO",
                    message="Federated merkle roots consistent between summary and ledger.",
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

    base = VAULT_ROOT
    report = ComplianceReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        strict=args.strict and not args.allow_missing_artifacts,
        worktree_root=str(WORKTREE_ROOT),
        vault_root=str(VAULT_ROOT),
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
    evaluate_federated_artifacts(report, base)
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
