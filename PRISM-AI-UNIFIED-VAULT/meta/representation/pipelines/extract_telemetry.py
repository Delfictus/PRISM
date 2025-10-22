#!/usr/bin/env python3
"""Convert semantic plasticity telemetry JSONL into representation dataset JSON.

This script reads one or more telemetry streams (JSON lines files) containing
semantic plasticity events, aggregates observations per concept, and emits a
dataset compatible with the Phase M4 representation adapter.

Usage:
    python extract_telemetry.py --input telemetry/*.jsonl --output dataset.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract semantic plasticity dataset from telemetry")
    parser.add_argument("--input", nargs="+", required=True, help="Telemetry JSONL files to scan")
    parser.add_argument("--output", required=True, help="Destination dataset JSON path")
    parser.add_argument("--dimension", type=int, default=4, help="Embedding dimension (default: 4)")
    parser.add_argument(
        "--adapter-id",
        default="semantic_plasticity",
        help="Adapter identifier to embed in the dataset (default: semantic_plasticity)",
    )
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


def extract_semantic_events(events: List[Dict[str, object]]) -> List[Dict[str, object]]:
    extracted: List[Dict[str, object]] = []
    for event in events:
        component = event.get("component")
        if component != "semantic_plasticity":
            continue
        payload = event.get("payload", {})
        concept_id = payload.get("concept_id")
        embedding = payload.get("embedding")
        if not concept_id or not isinstance(embedding, list):
            continue
        extracted.append(
            {
                "concept_id": concept_id,
                "embedding": embedding,
                "timestamp_ms": event.get("timestamp_ms") or event.get("timestamp_us", 0) // 1000,
                "source": event.get("source"),
                "notes": payload.get("notes"),
            }
        )
    return extracted


def build_dataset(args: argparse.Namespace, events: List[Dict[str, object]]) -> Dict[str, object]:
    observations = extract_semantic_events(events)
    concepts = {}  # concept_id -> embedding
    for obs in observations:
        concept_id = obs["concept_id"]
        if concept_id not in concepts:
            embedding = obs["embedding"]
            if len(embedding) == args.dimension:
                concepts[concept_id] = embedding

    return {
        "adapter_id": args.adapter_id,
        "dimension": args.dimension,
        "concepts": [
            {
                "id": concept_id,
                "description": f"Telemetry-derived concept {concept_id}",
                "attributes": {},
                "related": [],
                "embedding": embedding,
            }
            for concept_id, embedding in sorted(concepts.items())
        ],
        "observations": observations,
    }


def main() -> None:
    args = parse_args()
    events = load_events(args.input)
    dataset = build_dataset(args, events)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Wrote dataset with {len(dataset['concepts'])} concepts and {len(dataset['observations'])} observations -> {output_path}")


if __name__ == "__main__":
    main()
