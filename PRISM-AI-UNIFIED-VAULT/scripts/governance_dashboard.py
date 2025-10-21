#!/usr/bin/env python3
"""
Governance dashboard and alarm surface for the Meta Evolution Cycle (MEC).

Features
--------
* Aggregates compliance, performance, determinism, and feature flag data.
* Emits alarm payloads when guardrail thresholds are violated.
* Supports Markdown or JSON output, HTML snapshot export, and HTTP serving.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import socketserver
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PhaseId = str

try:
    from . import vault_root  # type: ignore
except ImportError:
    def vault_root() -> Path:
        return Path(__file__).resolve().parents[1]


# Thresholds derived from A-DoD governance contract
OCCUPANCY_MIN = 0.60
SM_EFFICIENCY_MIN = 0.70
VARIANCE_MAX = 0.08
DETERMINISM_STATUS_REQUIRED = {"PASS", "STABLE"}
META_PROD_ALLOWED = {"disabled", "shadow"}

BACKUP_TARGET_PHASE = "M6"
META_REGISTRY_PATH = Path("meta/meta_flags.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        return {"__error__": f"invalid json: {exc}"}


def compute_meta_merkle(records: Sequence[Dict[str, Any]]) -> str:
    if not records:
        return hashlib.sha256(b"meta-empty").hexdigest()

    leaves: List[bytes] = []
    for record in records:
        payload = json.dumps(record, separators=(",", ":")).encode("utf-8")
        leaves.append(hashlib.sha256(payload).digest())
    leaves.sort()

    working = leaves
    while len(working) > 1:
        next_level: List[bytes] = []
        for idx in range(0, len(working), 2):
            left = working[idx]
            right = working[idx + 1] if idx + 1 < len(working) else working[idx]
            next_level.append(hashlib.sha256(left + right).digest())
        working = next_level
    return working[0].hex()


def summarise_meta_registry(base: Path) -> Dict[str, Any]:
    registry_path = base / META_REGISTRY_PATH
    data = load_json(registry_path)
    if data is None:
        return {
            "status": "missing",
            "merkle_root": None,
            "features": [],
            "enforced": [],
            "path": str(registry_path),
        }
    if "__error__" in data:
        return {
            "status": "corrupt",
            "error": data["__error__"],
            "merkle_root": None,
            "features": [],
            "enforced": [],
            "path": str(registry_path),
        }

    records = data.get("records", [])
    merkle_root = data.get("merkle_root")
    expected_root = compute_meta_merkle(records if isinstance(records, list) else [])
    status = "valid" if merkle_root == expected_root else "mismatch"

    features: List[Dict[str, Any]] = []
    observed: set[str] = set()
    if isinstance(records, list):
        for record in records:
            feature_id = record.get("id", "unknown")
            observed.add(feature_id)
            state = record.get("state", {})
            features.append(
                {
                    "id": feature_id,
                    "state": state,
                    "updated_at": record.get("updated_at"),
                    "updated_by": record.get("updated_by"),
                    "rationale": record.get("rationale"),
                    "evidence_uri": record.get("evidence_uri"),
                    "approvals": record.get("approvals", []),
                    "controls": record.get("controls", {}),
                }
            )

    invariants = data.get("invariant_snapshot", {})

    return {
        "status": status,
        "merkle_root": merkle_root,
        "expected_root": expected_root,
        "features": features,
        "enforced": invariants.get("enforced", []),
        "entropy": invariants.get("generation_entropy"),
        "missing_flags": sorted(
            list(
                {"meta_generation", "ontology_bridge", "free_energy_snapshots", "semantic_plasticity", "federated_meta", "meta_prod"}
                - observed
            )
        ),
        "path": str(registry_path),
    }


def collect_backups(base: Path, phase: PhaseId) -> Dict[str, Any]:
    backups_dir = base / "artifacts" / "mec" / phase / "backups"
    entries: List[Dict[str, Any]] = []
    if backups_dir.exists():
        for candidate in sorted(backups_dir.iterdir()):
            if not candidate.is_dir():
                continue
            manifest = load_json(candidate / "manifest.json")
            entries.append(
                {
                    "path": str(candidate),
                    "manifest": manifest,
                    "timestamp": candidate.name,
                }
            )

    latest = entries[-1] if entries else None
    return {
        "phase": phase,
        "count": len(entries),
        "latest": latest,
        "entries": entries,
    }


def collect_dashboard(base: Path) -> Dict[str, Any]:
    advanced = load_json(base / "artifacts" / "advanced_manifest.json")
    roofline = load_json(base / "reports" / "roofline.json")
    determinism = load_json(base / "reports" / "determinism_replay.json")
    protein = load_json(base / "reports" / "protein_auroc.json")
    graph_capture = load_json(base / "reports" / "graph_capture.json")
    device_caps = load_json(base / "device_caps.json")
    path_decision = load_json(base / "path_decision.json")
    feasibility = (base / "feasibility.log").read_text(encoding="utf-8").strip() if (base / "feasibility.log").exists() else None

    meta_registry = summarise_meta_registry(base)
    backups = collect_backups(base, BACKUP_TARGET_PHASE)

    alerts = compute_alerts(roofline, determinism, meta_registry, backups)
    health = "critical" if any(a["severity"] in {"critical", "blocker"} for a in alerts) else ("warning" if alerts else "pass")

    return {
        "generated_at": utc_now(),
        "health": health,
        "alerts": alerts,
        "advanced_manifest": advanced,
        "roofline": roofline,
        "determinism": determinism,
        "protein": protein,
        "graph_capture": graph_capture,
        "device_caps": device_caps,
        "path_decision": path_decision,
        "feasibility": feasibility,
        "meta_registry": meta_registry,
        "backups": backups,
    }


def compute_alerts(
    roofline: Optional[Dict[str, Any]],
    determinism: Optional[Dict[str, Any]],
    meta_registry: Dict[str, Any],
    backups: Dict[str, Any],
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []

    if roofline and "__error__" not in roofline:
        occupancy = roofline.get("occupancy")
        sm_eff = roofline.get("sm_efficiency")
        bandwidth = roofline.get("achieved_bandwidth")
        if occupancy is not None and occupancy < OCCUPANCY_MIN:
            alerts.append(
                {
                    "severity": "critical",
                    "category": "performance",
                    "message": f"Occupancy {occupancy:.2f} below minimum {OCCUPANCY_MIN:.2f}",
                }
            )
        if sm_eff is not None and sm_eff < SM_EFFICIENCY_MIN:
            alerts.append(
                {
                    "severity": "warning",
                    "category": "performance",
                    "message": f"SM efficiency {sm_eff:.2f} below target {SM_EFFICIENCY_MIN:.2f}",
                }
            )
        if bandwidth is not None and bandwidth < 0.60:
            alerts.append(
                {
                    "severity": "warning",
                    "category": "performance",
                    "message": f"Achieved bandwidth {bandwidth:.2f} below 0.60 roofline target",
                }
            )
    else:
        alerts.append(
            {
                "severity": "critical",
                "category": "observability",
                "message": "Roofline report missing or corrupt.",
            }
        )

    if determinism and "__error__" not in determinism:
        variance = determinism.get("variance")
        status = (determinism.get("status") or determinism.get("status", "")).upper()
        if variance is not None and variance > VARIANCE_MAX:
            alerts.append(
                {
                    "severity": "critical",
                    "category": "determinism",
                    "message": f"Determinism variance {variance:.2f} exceeds {VARIANCE_MAX:.2f}",
                }
            )
        if status and status not in DETERMINISM_STATUS_REQUIRED:
            alerts.append(
                {
                    "severity": "critical",
                    "category": "determinism",
                    "message": f"Determinism status {status} outside required {sorted(DETERMINISM_STATUS_REQUIRED)}",
                }
            )
    else:
        alerts.append(
            {
                "severity": "critical",
                "category": "determinism",
                "message": "Determinism replay report missing or corrupt.",
            }
        )

    if meta_registry["status"] != "valid":
        alerts.append(
            {
                "severity": "blocker",
                "category": "feature_flags",
                "message": f"Meta registry integrity issue: {meta_registry['status']}",
            }
        )
    else:
        meta_prod = next((f for f in meta_registry["features"] if f["id"] == "meta_prod"), None)
        if meta_prod:
            state = meta_prod.get("state", {})
            mode = state.get("mode")
            if mode not in META_PROD_ALLOWED:
                alerts.append(
                    {
                        "severity": "critical",
                        "category": "feature_flags",
                        "message": f"`meta_prod` unexpectedly in mode '{mode}'.",
                    }
                )
            approvals = meta_prod.get("approvals") or []
            approved = [entry for entry in approvals if entry.get("status") == "approved"]
            if len(approved) < 2:
                alerts.append(
                    {
                        "severity": "warning",
                        "category": "feature_flags",
                        "message": "`meta_prod` has fewer than two recorded approvals.",
                    }
                )
        missing = meta_registry.get("missing_flags", [])
        if missing:
            alerts.append(
                {
                    "severity": "critical",
                    "category": "feature_flags",
                    "message": f"Meta registry missing flags: {', '.join(missing)}",
                }
            )

    if backups["count"] == 0:
        alerts.append(
            {
                "severity": "warning",
                "category": "rollback",
                "message": f"No backups found for phase {backups['phase']}.",
            }
        )

    return alerts


def render_markdown(data: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# MEC Governance Dashboard")
    lines.append(f"Generated: {data['generated_at']}  ")
    lines.append(f"Overall Health: **{data['health'].upper()}**")
    lines.append("")

    alerts = data["alerts"]
    if alerts:
        lines.append("## Alerts")
        for alert in alerts:
            lines.append(f"- `{alert['severity'].upper()}` · **{alert['category']}** – {alert['message']}")
        lines.append("")
    else:
        lines.append("## Alerts")
        lines.append("- ✅ No active alerts.")
        lines.append("")

    roofline = data.get("roofline") or {}
    determinism = data.get("determinism") or {}
    meta_registry = data.get("meta_registry") or {}

    lines.append("## Performance")
    if roofline and "__error__" not in roofline:
        lines.append(
            f"- Occupancy: {roofline.get('occupancy', 'n/a')}  •  SM Eff: {roofline.get('sm_efficiency', 'n/a')}  •  Bandwidth: {roofline.get('achieved_bandwidth', 'n/a')}"
        )
        lines.append(f"- Notes: {roofline.get('notes', '—')}")
    else:
        lines.append("- ❌ Roofline data unavailable.")
    lines.append("")

    lines.append("## Determinism")
    if determinism and "__error__" not in determinism:
        lines.append(
            f"- Status: {determinism.get('status', 'UNKNOWN')}  •  Variance: {determinism.get('variance', 'n/a')}  •  Hash A/B: {determinism.get('hash_a')} / {determinism.get('hash_b')}"
        )
        lines.append(f"- Notes: {determinism.get('notes', '—')}")
    else:
        lines.append("- ❌ Determinism report unavailable.")
    lines.append("")

    lines.append("## Feature Flags")
    if meta_registry.get("features"):
        lines.append(f"- Merkle Root: `{meta_registry.get('merkle_root')}` (expected `{meta_registry.get('expected_root')}`)")
        lines.append(f"- Enforced invariants: {meta_registry.get('enforced') or 'none'}")
        for record in meta_registry["features"]:
            state = record.get("state", {})
            approvals = record.get("approvals") or []
            lines.append(
                f"  - `{record['id']}` → {state.get('mode', 'unknown')} (updated {record.get('updated_at')}, approvals={len(approvals)})"
            )
    else:
        lines.append("- ❌ Meta registry unreadable.")
    lines.append("")

    backups = data.get("backups") or {}
    lines.append("## Backups")
    if backups.get("count", 0) > 0:
        latest = backups.get("latest")
        lines.append(f"- Snapshots: {backups['count']} (latest: {latest.get('timestamp') if latest else 'n/a'})")
        if latest and latest.get("manifest"):
            digest = latest["manifest"].get("composite_hash")
            lines.append(f"- Latest composite hash: `{digest}`")
    else:
        lines.append("- ❌ No rollback snapshots located.")

    return "\n".join(lines)


def render_html(data: Dict[str, Any]) -> str:
    markdown = render_markdown(data)
    escaped = markdown.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>MEC Governance Dashboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; margin: 2rem; }}
    a {{ color: #93c5fd; }}
    pre {{ background: #1e293b; padding: 1.5rem; border-radius: 0.75rem; overflow-x: auto; }}
    h1, h2 {{ color: #38bdf8; }}
  </style>
</head>
<body>
  <h1>MEC Governance Dashboard</h1>
  <p>Generated: {data['generated_at']}</p>
  <pre>{escaped}</pre>
</body>
</html>
"""


def write_snapshot(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(data), encoding="utf-8")


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # type: ignore[override]
        try:
            payload = collect_dashboard(vault_root())
            if self.path.endswith(".json") or self.path.startswith("/api"):
                body = json.dumps(payload, indent=2)
                content_type = "application/json; charset=utf-8"
            else:
                body = render_html(payload)
                content_type = "text/html; charset=utf-8"

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"dashboard error: {exc}\n".encode("utf-8"))

    def log_message(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        """Silence default request logging."""
        return


def serve_dashboard(port: int, refresh: float) -> None:
    base = vault_root()

    def background_refresh() -> None:
        while True:
            collect_dashboard(base)
            time.sleep(refresh)

    refresher = threading.Thread(target=background_refresh, daemon=True)
    refresher.start()

    with socketserver.TCPServer(("", port), DashboardHandler) as httpd:
        print(f"Serving MEC dashboard on http://127.0.0.1:{port} (refresh interval: {refresh}s)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping dashboard server...")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MEC governance dashboard outputs.")
    parser.add_argument("--format", choices={"markdown", "json"}, default="markdown", help="Output format for stdout.")
    parser.add_argument("--snapshot", type=Path, help="Optional path to write an HTML snapshot.")
    parser.add_argument("--serve", action="store_true", help="Start an HTTP server instead of printing once.")
    parser.add_argument("--port", type=int, default=8088, help="Port for --serve (default: 8088).")
    parser.add_argument("--refresh", type=float, default=30.0, help="Refresh interval in seconds for --serve.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    base = vault_root()

    if args.serve:
        serve_dashboard(args.port, max(args.refresh, 5.0))
        return 0

    data = collect_dashboard(base)
    if args.format == "json":
        print(json.dumps(data, indent=2))
    else:
        print(render_markdown(data))

    if args.snapshot:
        write_snapshot(args.snapshot, data)

    # Alarm payload for automation hooks
    if any(alert["severity"] in {"critical", "blocker"} for alert in data["alerts"]):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
