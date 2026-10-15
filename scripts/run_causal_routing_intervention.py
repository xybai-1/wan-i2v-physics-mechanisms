#!/usr/bin/env python3
"""Run Wan causal-routing interventions with equal-mass temporal controls."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from diffusers.utils import export_to_video, load_image

from causal_attention_probe import CausalAttentionProbe, locate_text_spans
from run_causal_attention_diagnostic import diagnostic_case, transformer_grid
from run_mechanism_matrix import ModelAdapter, cosine_to_prior, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-config", type=Path, required=True)
    parser.add_argument("--intervention-config", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--phaselock-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def run_one(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    diagnostic: dict[str, Any],
    intervention: dict[str, Any],
    lock: dict[str, Any],
    adapter: ModelAdapter,
    case_id: str,
    variant: str,
) -> None:
    model_key = intervention["scope"]["model_key"]
    seed = int(intervention["scope"]["seed"])
    case = next((item for item in manifest["cases"] if item["case_id"] == case_id), None)
    if case is None:
        raise SystemExit(f"case is not registered in parent manifest: {case_id}")
    diag_case = diagnostic_case(diagnostic, case_id)
    if variant not in intervention["scope"]["variants"]:
        raise SystemExit(f"variant is not registered: {variant}")
    if case_id not in intervention["scope"]["cases"]:
        raise SystemExit(f"case is not registered in intervention scope: {case_id}")

    protocol = manifest["protocol"]
    model_config = manifest["models"][model_key]
    route_protocol = intervention["protocol"]
    image_path = args.data_root / case["first_image"]
    expected_hash = lock["files"][case_id]["sha256"]
    actual_hash = file_sha256(image_path)
    if actual_hash != expected_hash:
        raise SystemExit(f"conditioning image checksum mismatch: {image_path}")

    run_dir = args.output_root / model_key / case_id / f"seed_{seed}" / variant
    if (run_dir / "run.json").exists():
        print(f"already complete: {model_key} {case_id} seed={seed} {variant}")
        return
    if run_dir.exists():
        raise SystemExit(f"partial output requires inspection; refusing overwrite: {run_dir}")
    run_dir.mkdir(parents=True)

    common = adapter.call_kwargs(case["prompt"], load_image(str(image_path)), protocol)
    few_frames = adapter.pipe(
        **common,
        num_inference_steps=protocol["few_step_count"],
        generator=adapter.generator(seed),
    ).frames[0]
    few_path = run_dir / "few_step.mp4"
    export_to_video(few_frames, str(few_path), fps=model_config["fps"])

    sys.path.insert(0, str(args.phaselock_code))
    from phaselock.guidance import extract_motion_prior

    few_latents = adapter.encode_generated_frames(few_frames)
    motion_prior = extract_motion_prior(few_latents).detach()
    del few_frames, few_latents
    gc.collect()
    torch.cuda.empty_cache()

    positions, token_metadata = locate_text_spans(
        adapter.pipe.tokenizer,
        case["prompt"],
        diag_case["attention_targets"],
        512,
    )
    plan = {
        "mode": variant,
        "steps_zero_based": route_protocol["steps_zero_based"],
        "self_layers": route_protocol["self_layers"],
        "support_topk_per_frame": route_protocol["support_topk_per_frame"],
        "temporal_lag_frames": route_protocol["temporal_lag_frames"],
        "total_strength": route_protocol["total_strength"],
        "edges": intervention["case_edges"][case_id],
    }
    probe = CausalAttentionProbe(
        adapter.pipe.transformer,
        model_kind="wan",
        token_positions=positions,
        token_metadata=token_metadata,
        grid_thw=transformer_grid(adapter),
        steps=route_protocol["steps_zero_based"],
        entity_map_layers=route_protocol["entity_map_layers"],
        self_source_layer_pairs=route_protocol["self_source_layer_pairs"],
        calls_per_step=model_config["transformer_calls_per_step"],
        cfg_layout=model_config["cfg_layout"],
        query_topk_per_frame=route_protocol["query_topk_per_frame"],
        support_topk_per_frame=route_protocol["support_topk_per_frame"],
        query_chunk_size=route_protocol["query_chunk_size"],
        cross_query_chunk_size=route_protocol["cross_query_chunk_size"],
        intervention_plan=plan,
    ).install()
    trace: list[dict[str, Any]] = []

    def callback(active_pipe, step_index, timestep, callback_kwargs):
        del active_pipe
        latents = callback_kwargs["latents"]
        trace.append(
            {
                "step_one_based": int(step_index + 1),
                "timestep": float(timestep.detach().float().mean().cpu()),
                "cosine_to_two_step_prior": cosine_to_prior(
                    adapter, latents, motion_prior
                ),
            }
        )
        return callback_kwargs

    try:
        final_frames = adapter.pipe(
            **common,
            num_inference_steps=protocol["num_inference_steps"],
            generator=adapter.generator(seed),
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        ).frames[0]
    finally:
        probe.remove()
    final_path = run_dir / "final.mp4"
    export_to_video(final_frames, str(final_path), fps=model_config["fps"])
    del final_frames
    (run_dir / "latent_delta_trace.json").write_text(
        json.dumps(trace, indent=2) + "\n", encoding="utf-8"
    )
    probe.save(run_dir / "attention")

    reference_dir = (
        args.reference_root / model_key / case_id / f"seed_{seed}" / "baseline"
    )
    few_reference = reference_dir / "few_step.mp4"
    final_reference = reference_dir / "final.mp4"
    reference_check = {
        "few_step": {
            "reference": str(few_reference),
            "reference_sha256": file_sha256(few_reference),
            "intervention_sha256": file_sha256(few_path),
        },
        "final": {
            "reference": str(final_reference),
            "reference_sha256": file_sha256(final_reference),
            "intervention_sha256": file_sha256(final_path),
        },
    }
    reference_check["few_step"]["identical"] = (
        reference_check["few_step"]["reference_sha256"]
        == reference_check["few_step"]["intervention_sha256"]
    )
    reference_check["final"]["identical"] = (
        reference_check["final"]["reference_sha256"]
        == reference_check["final"]["intervention_sha256"]
    )
    (run_dir / "reference_check.json").write_text(
        json.dumps(reference_check, indent=2) + "\n", encoding="utf-8"
    )
    if not reference_check["few_step"]["identical"]:
        raise RuntimeError("intervention changed the frozen two-step prior")
    if reference_check["final"]["identical"]:
        raise RuntimeError("intervention produced a byte-identical final video")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "study_revision": os.environ.get("STUDY_REVISION"),
        "study_id": intervention["study_id"],
        "parent_study_id": intervention["parent_study_id"],
        "model_key": model_key,
        "model": model_config,
        "case": case,
        "diagnostic_case": diag_case,
        "seed": seed,
        "variant": variant,
        "conditioning_image": str(image_path),
        "conditioning_sha256": actual_hash,
        "parent_protocol": protocol,
        "intervention_protocol": route_protocol,
        "intervention_plan": plan,
        "grid_thw": list(transformer_grid(adapter)),
        "reference_check": reference_check,
        "torch_version": torch.__version__,
        "diffusers_version": __import__("diffusers").__version__,
        "outputs": {
            "few_step": "few_step.mp4",
            "final": "final.mp4",
            "latent_delta_trace": "latent_delta_trace.json",
            "attention_maps": "attention/attention_maps.npz",
            "attention_metadata": "attention/attention_metadata.json",
            "reference_check": "reference_check.json",
        },
        "intervention_enabled": True,
        "raw_latents_saved": False,
    }
    (run_dir / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    diagnostic = json.loads(args.diagnostic_config.read_text(encoding="utf-8"))
    intervention = json.loads(args.intervention_config.read_text(encoding="utf-8"))
    lock = json.loads(args.input_lock.read_text(encoding="utf-8"))
    model_key = intervention["scope"]["model_key"]
    if model_key != "wan21_i2v":
        raise SystemExit("causal routing intervention v1 is intentionally Wan-only")
    adapter = ModelAdapter(
        model_key,
        manifest["models"][model_key],
        args.models_root / manifest["models"][model_key]["local_dir"],
    )
    for case_id in args.case_id:
        for variant in args.variant:
            run_one(
                args,
                manifest,
                diagnostic,
                intervention,
                lock,
                adapter,
                case_id,
                variant,
            )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
