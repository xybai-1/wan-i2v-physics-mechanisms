#!/usr/bin/env python3
"""Validate the pre-registered PhyGround mechanism-study manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ALLOWED_STRATA = {
    "visual_dominance",
    "simple_motion",
    "collision_momentum",
    "state_transition",
    "multi_stage_causal",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise SystemExit("unsupported schema_version")
    if payload.get("benchmark") != "PhyGround":
        raise SystemExit("benchmark must be PhyGround")
    if args.require_frozen and not payload.get("selection_frozen"):
        raise SystemExit("case selection is not frozen")
    if not payload.get("model_revision"):
        raise SystemExit("model_revision must be pinned")
    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)):
        raise SystemExit("seeds must be a non-empty unique list")

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("cases must be a non-empty list")
    ids: set[str] = set()
    for index, case in enumerate(cases):
        prefix = f"cases[{index}]"
        video_id = case.get("video_id")
        if not video_id or video_id in ids:
            raise SystemExit(f"{prefix}: missing or duplicate video_id")
        ids.add(video_id)
        if case.get("stratum") not in ALLOWED_STRATA:
            raise SystemExit(f"{prefix}: invalid stratum")
        for field in (
            "prompt",
            "first_image",
            "first_image_url",
            "first_image_sha256",
            "entity_tokens",
            "events",
        ):
            if not case.get(field):
                raise SystemExit(f"{prefix}: missing {field}")
        if len(case["first_image_sha256"]) != 64:
            raise SystemExit(f"{prefix}: first_image_sha256 must be a SHA-256 hex digest")
        event_ids = [event.get("id") for event in case["events"]]
        if any(not event_id for event_id in event_ids) or len(event_ids) != len(set(event_ids)):
            raise SystemExit(f"{prefix}: event IDs must be non-empty and unique")
        order = case.get("expected_event_order", [])
        if set(order) - set(event_ids):
            raise SystemExit(f"{prefix}: expected_event_order references unknown event")
        targets = case.get("attention_targets")
        if not isinstance(targets, dict) or len(targets) < 3:
            raise SystemExit(f"{prefix}: attention_targets must define two entities and one event")
        prompt_lower = case["prompt"].lower()
        for label, phrase in targets.items():
            if not isinstance(phrase, str) or phrase.lower() not in prompt_lower:
                raise SystemExit(f"{prefix}: attention target {label!r} is not a prompt substring")

    pair_ids = [case.get("pair_id") for case in cases if case.get("pair_id")]
    if pair_ids:
        from collections import Counter

        pair_counts = Counter(pair_ids)
        if any(count != 2 for count in pair_counts.values()):
            raise SystemExit("every declared pair_id must occur exactly twice")

    print(f"valid manifest: {len(cases)} cases, {len(seeds)} seeds")


if __name__ == "__main__":
    main()

