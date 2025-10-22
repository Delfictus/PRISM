#!/usr/bin/env python3
"""
Extract federation updates from telemetry JSONL and produce dataset.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract federation dataset from telemetry JSONL files.")
    parser.add_argument("--input", nargs="+", required=True, help="Telemetry JSONL files to process.")
    parser.add_argument("--output", required=True, help="Output dataset path (JSON).")
    parser.add_argument("--policy-hash", default="federation-policy-hash", help="Policy hash identifier.")
    parser.add_argument("--epoch", type=int, default=1, help="Federated epoch number.")
    return parser.parse_args()


def load_events(paths: List[str]) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    for path in paths:
        file_path = Path(path)
        if not file_path.exists():
            continue
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def build_dataset(events: List[Dict[str, object]], epoch: int, policy_hash: str) -> Dict[str, object]:
    nodes: Dict[str, Dict[str, object]] = {}
    updates: Dict[str, Dict[str, object]] = {}

    for event in events:
        if event.get("component") != "federation":
            continue
        payload = event.get("payload", {})
        node_id = payload.get("node")
        if not node_id:
            continue
        node_entry = nodes.setdefault(
            node_id,
            {
                "fingerprint": node_id,
                "software_version": payload.get("software_version", "unknown"),
                "capabilities": payload.get("capabilities", []),
                "stake_weight": float(payload.get("stake_weight", 1.0)),
            },
        )

        # Update stake weight if telemetry provides a newer value
        node_entry["stake_weight"] = float(payload.get("stake_weight", node_entry["stake_weight"]))

        update_hash = payload.get("update_hash")
        ledger_block_id = payload.get("ledger_block_id")
        payload_hash = payload.get("payload_hash")

        if not (update_hash and ledger_block_id and payload_hash):
            continue

        updates[node_id] = {
            "node": node_id,
            "epoch": payload.get("epoch", epoch),
            "update_hash": update_hash,
            "ledger_block_id": ledger_block_id,
            "payload_hash": payload_hash,
            "stake_weight": float(payload.get("stake_weight", node_entry["stake_weight"])),
            "timestamp_ms": event.get("timestamp_ms"),
        }

    return {
        "policy_hash": policy_hash,
        "epoch": epoch,
        "nodes": list(nodes.values()),
        "updates": list(updates.values()),
    }


def main() -> None:
    args = parse_args()
    events = load_events(args.input)
    dataset = build_dataset(events, args.epoch, args.policy_hash)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(
        f"Wrote federation dataset with {len(dataset['nodes'])} nodes and {len(dataset['updates'])} updates -> {output_path}"
    )


if __name__ == "__main__":
    main()
