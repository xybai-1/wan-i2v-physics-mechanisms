#!/usr/bin/env python3
"""Validate the frozen matrix cardinality and protocol invariants."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    assert payload["selection_frozen"] is True
    assert len(payload["models"]) == 3
    assert len(payload["cases"]) == 9
    assert payload["protocol"]["seeds"] == [42, 123, 777]
    assert payload["protocol"]["arms"] == ["baseline", "phaselock"]
    assert payload["protocol"]["decode_steps_one_based"] == [1, 2, 4, 8, 25, 50]
    assert payload["protocol"]["num_inference_steps"] == 50
    expected_complexities = {
        "single_entity_motion", "two_entity_contact", "multi_stage_causal_chain"
    }
    counts = Counter((case["benchmark"], case["complexity"]) for case in payload["cases"])
    assert set(case["benchmark"] for case in payload["cases"]) == {
        "PhyGround", "PhyGenBench", "Physics-IQ Verified"
    }
    assert set(case["complexity"] for case in payload["cases"]) == expected_complexities
    assert all(value == 1 for value in counts.values()) and len(counts) == 9
    assert len({case["case_id"] for case in payload["cases"]}) == 9
    cells = len(payload["models"]) * len(payload["cases"]) * len(payload["protocol"]["seeds"]) * len(payload["protocol"]["arms"])
    assert cells == 162
    print(json.dumps({"valid": True, "models": 3, "cases": 9, "cells": cells}, indent=2))


if __name__ == "__main__":
    main()
