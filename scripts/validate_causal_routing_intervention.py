#!/usr/bin/env python3
"""Validate Wan causal-routing intervention outputs and control invariants."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def video_decodes(path: Path) -> bool:
    reader = None
    try:
        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        metadata = next(reader)
        frame = next(reader)
        return bool(metadata.get("size") and frame)
    except (OSError, RuntimeError, StopIteration):
        return False
    finally:
        if reader is not None:
            reader.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-config", type=Path, required=True)
    parser.add_argument("--intervention-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    diagnostic = json.loads(args.diagnostic_config.read_text(encoding="utf-8"))
    config = json.loads(args.intervention_config.read_text(encoding="utf-8"))
    model_key = config["scope"]["model_key"]
    seed = int(config["scope"]["seed"])
    variants = config["scope"]["variants"]
    cases = (
        ["phygenbench_chain_fire_paper"]
        if args.smoke
        else config["scope"]["cases"]
    )
    protocol = config["protocol"]
    expected_events = len(protocol["steps_zero_based"]) * (
        len(protocol["entity_map_layers"])
        + len(protocol["self_source_layer_pairs"])
    )
    expected_interventions = len(protocol["steps_zero_based"]) * len(
        protocol["self_layers"]
    )
    errors: list[str] = []
    checked: list[dict] = []

    for case_id in cases:
        targets = next(
            case["attention_targets"]
            for case in diagnostic["cases"]
            if case["case_id"] == case_id
        )
        expected_arrays = len(protocol["steps_zero_based"]) * (
            len(protocol["entity_map_layers"]) * len(targets)
            + len(protocol["self_source_layer_pairs"]) * len(targets) ** 2
        )
        final_hashes: dict[str, str] = {}
        for variant in variants:
            run_dir = args.output_root / model_key / case_id / f"seed_{seed}" / variant
            run_path = run_dir / "run.json"
            if not run_path.exists():
                errors.append(f"missing run.json: {run_dir}")
                continue
            run = json.loads(run_path.read_text(encoding="utf-8"))
            if run["variant"] != variant or run["case"]["case_id"] != case_id:
                errors.append(f"run identity mismatch: {run_dir}")
            checks = run["reference_check"]
            if not checks["few_step"]["identical"]:
                errors.append(f"two-step prior mismatch: {run_dir}")
            if checks["final"]["identical"]:
                errors.append(f"final video unchanged from baseline: {run_dir}")
            final_hashes[variant] = checks["final"]["intervention_sha256"]
            for video_name in ("few_step.mp4", "final.mp4"):
                video = run_dir / video_name
                if not video.exists() or video.stat().st_size == 0 or not video_decodes(video):
                    errors.append(f"invalid video: {video}")
            trace = json.loads(
                (run_dir / "latent_delta_trace.json").read_text(encoding="utf-8")
            )
            if len(trace) != 50 or not all(
                math.isfinite(float(item["cosine_to_two_step_prior"])) for item in trace
            ):
                errors.append(f"invalid latent trace: {run_dir}")
            metadata = json.loads(
                (run_dir / "attention" / "attention_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            if not metadata["generation_path_modified"]:
                errors.append(f"metadata says intervention disabled: {run_dir}")
            if metadata["intervention_plan"]["mode"] != variant:
                errors.append(f"intervention mode mismatch: {run_dir}")
            if len(metadata["events"]) != expected_events:
                errors.append(f"diagnostic event count mismatch: {run_dir}")
            intervention_events = metadata["intervention_events"]
            if len(intervention_events) != expected_interventions:
                errors.append(f"intervention event count mismatch: {run_dir}")
            if not all(
                event["mode"] == variant
                and math.isfinite(float(event["delta_to_reference_rms"]))
                and float(event["delta_to_reference_rms"]) > 0
                for event in intervention_events
            ):
                errors.append(f"invalid intervention statistics: {run_dir}")
            with np.load(run_dir / "attention" / "attention_maps.npz") as arrays:
                if len(arrays.files) != expected_arrays:
                    errors.append(f"array count mismatch: {run_dir}")
                for name in arrays.files:
                    value = arrays[name]
                    if not np.isfinite(value).all():
                        errors.append(f"non-finite attention array: {run_dir}/{name}")
                    if name.startswith("entity_qk_") and tuple(value.shape) != tuple(
                        run["grid_thw"]
                    ):
                        errors.append(f"entity shape mismatch: {run_dir}/{name}")
                    if name.startswith("self_") and value.shape != (
                        run["grid_thw"][0],
                        run["grid_thw"][0],
                    ):
                        errors.append(f"self shape mismatch: {run_dir}/{name}")
            checked.append(
                {
                    "case": case_id,
                    "variant": variant,
                    "arrays": expected_arrays,
                    "diagnostic_events": expected_events,
                    "intervention_events": expected_interventions,
                }
            )
        if len(final_hashes) == len(variants) and len(set(final_hashes.values())) != len(
            variants
        ):
            errors.append(f"control variants produced duplicate final hashes: {case_id}")

    result = {
        "valid": not errors,
        "checked_runs": len(checked),
        "expected_runs": len(cases) * len(variants),
        "errors": errors,
        "runs": checked,
    }
    print(json.dumps(result, indent=2))
    if errors or len(checked) != result["expected_runs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
