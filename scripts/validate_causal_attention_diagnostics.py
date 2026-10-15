#!/usr/bin/env python3
"""Validate causal-attention diagnostic outputs and protocol invariants."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def video_decodes(path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    streams = json.loads(result.stdout).get("streams", [])
    return bool(streams and streams[0].get("codec_name"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-key", action="append")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    diagnostic = json.loads(args.diagnostic_config.read_text(encoding="utf-8"))
    models = args.model_key or list(diagnostic["models"])
    cases = (
        ["phyground_pair_balls_collide"]
        if args.smoke
        else [case["case_id"] for case in diagnostic["cases"]]
    )
    arms = ["baseline"] if args.smoke else diagnostic["protocol"]["arms"]
    seed = diagnostic["protocol"]["seed"]
    checked: list[dict] = []
    errors: list[str] = []

    for model_key in models:
        model_plan = diagnostic["models"][model_key]
        for case_id in cases:
            targets = next(
                case["attention_targets"]
                for case in diagnostic["cases"]
                if case["case_id"] == case_id
            )
            expected_arrays = len(diagnostic["protocol"]["steps_zero_based"]) * (
                len(model_plan["entity_map_layers"]) * len(targets)
                + len(model_plan["self_source_layer_pairs"]) * len(targets) ** 2
            )
            expected_events = len(diagnostic["protocol"]["steps_zero_based"]) * (
                len(model_plan["entity_map_layers"])
                + len(model_plan["self_source_layer_pairs"])
            )
            for arm in arms:
                run_dir = args.output_root / model_key / case_id / f"seed_{seed}" / arm
                run_path = run_dir / "run.json"
                if not run_path.exists():
                    errors.append(f"missing run.json: {run_dir}")
                    continue
                run = json.loads(run_path.read_text(encoding="utf-8"))
                if run["model_key"] != model_key or run["case"]["case_id"] != case_id:
                    errors.append(f"run identity mismatch: {run_dir}")
                checks = run["noninvasive_video_check"]
                if not all(check["identical"] for check in checks.values()):
                    errors.append(f"non-invasive hash failure: {run_dir}")
                for video_name in ("few_step.mp4", "final.mp4"):
                    video = run_dir / video_name
                    if not video.exists() or video.stat().st_size == 0 or not video_decodes(video):
                        errors.append(f"invalid video: {video}")
                metadata = json.loads(
                    (run_dir / "attention" / "attention_metadata.json").read_text(
                        encoding="utf-8"
                    )
                )
                if len(metadata["events"]) != expected_events:
                    errors.append(
                        f"event count {len(metadata['events'])}!={expected_events}: {run_dir}"
                    )
                with np.load(run_dir / "attention" / "attention_maps.npz") as arrays:
                    if len(arrays.files) != expected_arrays:
                        errors.append(
                            f"array count {len(arrays.files)}!={expected_arrays}: {run_dir}"
                        )
                    for name in arrays.files:
                        value = arrays[name]
                        if not np.isfinite(value).all():
                            errors.append(f"non-finite array {name}: {run_dir}")
                        if name.startswith("entity_qk_") and tuple(value.shape) != tuple(run["grid_thw"]):
                            errors.append(f"entity shape {value.shape}: {run_dir}/{name}")
                        if name.startswith("self_") and value.shape != (
                            run["grid_thw"][0], run["grid_thw"][0]
                        ):
                            errors.append(f"self shape {value.shape}: {run_dir}/{name}")
                checked.append(
                    {
                        "model": model_key,
                        "case": case_id,
                        "arm": arm,
                        "arrays": expected_arrays,
                        "events": expected_events,
                    }
                )
    result = {
        "valid": not errors,
        "checked_runs": len(checked),
        "expected_runs": len(models) * len(cases) * len(arms),
        "errors": errors,
        "runs": checked,
    }
    print(json.dumps(result, indent=2))
    if errors or len(checked) != result["expected_runs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
