#!/usr/bin/env python3
"""Validate completed mechanism-matrix cells before advancing a gate."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def validate_video(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty video: {path}")
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "null", "-"]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise ValueError(f"video does not decode: {path}: {completed.stderr[-1000:]}")


def validate_run(run_path: Path) -> dict:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    root = run_path.parent
    outputs = run["outputs"]
    videos = [outputs["few_step"], outputs["final"], *outputs["checkpoints"]]
    for relative in videos:
        validate_video(root / relative)
    trace = json.loads((root / outputs["latent_delta_trace"]).read_text(encoding="utf-8"))
    if len(trace) != 50:
        raise ValueError(f"expected 50 trace records, found {len(trace)}: {run_path}")
    for record in trace:
        for key in ("timestep", "cosine_to_two_step_prior_before_guidance", "cosine_to_two_step_prior_after_guidance"):
            if not math.isfinite(float(record[key])):
                raise ValueError(f"non-finite {key}: {run_path}")
    metadata = json.loads((root / outputs["attention_metadata"]).read_text(encoding="utf-8"))
    if metadata["record_count"] <= 0 or metadata["module_count"] <= 0:
        raise ValueError(f"empty attention statistics: {run_path}")
    with np.load(root / outputs["attention_stats"]) as arrays:
        if not arrays.files or any(not np.isfinite(arrays[key]).all() for key in arrays.files):
            raise ValueError(f"invalid attention arrays: {run_path}")
    forbidden = list(root.glob("**/latent*.pt")) + list(root.glob("**/attention_maps.npz"))
    if forbidden:
        raise ValueError(f"forbidden raw artifact(s): {forbidden}")
    return {
        "model_key": run["model_key"], "case_id": run["case"]["case_id"],
        "seed": run["seed"], "arm": run["arm"], "attention_records": metadata["record_count"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--expected-cells", type=int, required=True)
    args = parser.parse_args()
    paths = sorted(args.output_root.glob("*/*/seed_*/*/run.json"))
    if len(paths) != args.expected_cells:
        raise SystemExit(f"expected {args.expected_cells} run.json files, found {len(paths)}")
    records = [validate_run(path) for path in paths]
    print(json.dumps({"valid": True, "cell_count": len(records), "cells": records}, indent=2))


if __name__ == "__main__":
    main()
