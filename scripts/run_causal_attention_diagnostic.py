#!/usr/bin/env python3
"""Rerun frozen matrix cells with read-only causal-attention diagnostics."""

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
from run_mechanism_matrix import ModelAdapter, cosine_to_prior, file_sha256, git_revision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-config", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--arm", action="append", choices=("baseline", "phaselock"), required=True
    )
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--phaselock-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def tokenizer_length(model_kind: str) -> int:
    return {"wan": 512, "cogvideox": 226, "ltx": 128}[model_kind]


def transformer_grid(adapter: ModelAdapter) -> tuple[int, int, int]:
    temporal, height, width = adapter.latent_shape
    if adapter.kind == "wan":
        patch = adapter.pipe.transformer.config.patch_size
        return temporal // patch[0], height // patch[1], width // patch[2]
    if adapter.kind == "cogvideox":
        patch = adapter.pipe.transformer.config.patch_size
        return temporal, height // patch, width // patch
    return (
        temporal // adapter.pipe.transformer_temporal_patch_size,
        height // adapter.pipe.transformer_spatial_patch_size,
        width // adapter.pipe.transformer_spatial_patch_size,
    )


def diagnostic_case(config: dict[str, Any], case_id: str) -> dict[str, Any]:
    match = next((case for case in config["cases"] if case["case_id"] == case_id), None)
    if match is None:
        raise SystemExit(f"case is not registered in diagnostic config: {case_id}")
    return match


def run_one(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    diagnostic: dict[str, Any],
    lock: dict[str, Any],
    adapter: ModelAdapter,
    case_id: str,
    arm: str,
) -> None:
    case = next((case for case in manifest["cases"] if case["case_id"] == case_id), None)
    if case is None:
        raise SystemExit(f"case is not registered in parent manifest: {case_id}")
    diag_case = diagnostic_case(diagnostic, case_id)
    protocol = manifest["protocol"]
    model_config = manifest["models"][args.model_key]
    probe_config = diagnostic["models"][args.model_key]
    probe_protocol = diagnostic["protocol"]
    if args.seed not in protocol["seeds"]:
        raise SystemExit(f"seed is not registered in parent protocol: {args.seed}")

    image_path = args.data_root / case["first_image"]
    expected_image_hash = lock["files"][case_id]["sha256"]
    actual_image_hash = file_sha256(image_path)
    if actual_image_hash != expected_image_hash:
        raise SystemExit(f"conditioning image checksum mismatch: {image_path}")

    run_dir = args.output_root / args.model_key / case_id / f"seed_{args.seed}" / arm
    if (run_dir / "run.json").exists():
        print(f"already complete: {args.model_key} {case_id} {args.seed} {arm}")
        return
    if run_dir.exists():
        raise SystemExit(f"partial output requires inspection; refusing overwrite: {run_dir}")
    run_dir.mkdir(parents=True)

    common = adapter.call_kwargs(case["prompt"], load_image(str(image_path)), protocol)
    few_frames = adapter.pipe(
        **common,
        num_inference_steps=protocol["few_step_count"],
        generator=adapter.generator(args.seed),
    ).frames[0]
    few_path = run_dir / "few_step.mp4"
    export_to_video(few_frames, str(few_path), fps=model_config["fps"])

    sys.path.insert(0, str(args.phaselock_code))
    from phaselock.guidance import LatentDeltaGuidance, extract_motion_prior

    few_latents = adapter.encode_generated_frames(few_frames)
    motion_prior = extract_motion_prior(few_latents).detach()
    guide = LatentDeltaGuidance(
        motion_prior=motion_prior,
        guidance_strength=protocol["phaselock_guidance_strength"],
        guide_start=protocol["phaselock_guide_start"],
        guide_end=protocol["phaselock_guide_end"],
        total_steps=protocol["num_inference_steps"],
    )
    del few_frames, few_latents
    gc.collect()
    torch.cuda.empty_cache()

    positions, token_metadata = locate_text_spans(
        adapter.pipe.tokenizer,
        case["prompt"],
        diag_case["attention_targets"],
        tokenizer_length(adapter.kind),
    )
    probe = CausalAttentionProbe(
        adapter.pipe.transformer,
        model_kind=adapter.kind,
        token_positions=positions,
        token_metadata=token_metadata,
        grid_thw=transformer_grid(adapter),
        steps=probe_protocol["steps_zero_based"],
        entity_map_layers=probe_config["entity_map_layers"],
        self_source_layer_pairs=probe_config["self_source_layer_pairs"],
        calls_per_step=model_config["transformer_calls_per_step"],
        cfg_layout=model_config["cfg_layout"],
        query_topk_per_frame=probe_protocol["query_topk_per_frame"],
        support_topk_per_frame=probe_protocol["support_topk_per_frame"],
        query_chunk_size=probe_protocol["query_chunk_size"],
        cross_query_chunk_size=probe_protocol["cross_query_chunk_size"],
    ).install()
    trace: list[dict[str, Any]] = []

    def callback(active_pipe, step_index, timestep, callback_kwargs):
        del active_pipe
        latents = callback_kwargs["latents"]
        before = cosine_to_prior(adapter, latents, motion_prior)
        strength = guide.compute_schedule(step_index) if arm == "phaselock" else 0.0
        if arm == "phaselock":
            phase_layout = adapter.to_btc(latents)
            phase_layout = guide(
                adapter.pipe, step_index, timestep, {"latents": phase_layout}
            )["latents"]
            latents = adapter.from_btc(phase_layout)
            callback_kwargs["latents"] = latents
        trace.append(
            {
                "step_one_based": int(step_index + 1),
                "timestep": float(timestep.detach().float().mean().cpu()),
                "guidance_strength": float(strength),
                "cosine_to_two_step_prior_before_guidance": before,
                "cosine_to_two_step_prior_after_guidance": cosine_to_prior(
                    adapter, latents, motion_prior
                ),
            }
        )
        return callback_kwargs

    try:
        final_frames = adapter.pipe(
            **common,
            num_inference_steps=protocol["num_inference_steps"],
            generator=adapter.generator(args.seed),
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
        args.reference_root / args.model_key / case_id / f"seed_{args.seed}" / arm
    )
    checks: dict[str, Any] = {}
    for name, produced in (("few_step", few_path), ("final", final_path)):
        reference = reference_dir / f"{name}.mp4"
        if not reference.exists():
            raise RuntimeError(f"missing parent-matrix reference: {reference}")
        reference_hash = file_sha256(reference)
        produced_hash = file_sha256(produced)
        checks[name] = {
            "reference": str(reference),
            "reference_sha256": reference_hash,
            "diagnostic_sha256": produced_hash,
            "identical": reference_hash == produced_hash,
        }
    (run_dir / "noninvasive_check.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
    )
    if not all(check["identical"] for check in checks.values()):
        raise RuntimeError(f"instrumented generation differs from frozen reference: {checks}")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "study_revision": os.environ.get("STUDY_REVISION"),
        "study_id": diagnostic["study_id"],
        "parent_study_id": diagnostic["parent_study_id"],
        "model_key": args.model_key,
        "model": model_config,
        "case": case,
        "diagnostic_case": diag_case,
        "seed": args.seed,
        "arm": arm,
        "conditioning_image": str(image_path),
        "conditioning_sha256": actual_image_hash,
        "parent_protocol": protocol,
        "diagnostic_protocol": probe_protocol,
        "probe_model_config": probe_config,
        "grid_thw": list(transformer_grid(adapter)),
        "phaselock_revision_expected": manifest["source_revisions"]["phaselock"],
        "phaselock_revision_actual": git_revision(args.phaselock_code.parent),
        "noninvasive_video_check": checks,
        "torch_version": torch.__version__,
        "diffusers_version": __import__("diffusers").__version__,
        "outputs": {
            "few_step": "few_step.mp4",
            "final": "final.mp4",
            "latent_delta_trace": "latent_delta_trace.json",
            "attention_maps": "attention/attention_maps.npz",
            "attention_metadata": "attention/attention_metadata.json",
            "noninvasive_check": "noninvasive_check.json",
        },
        "intervention_enabled": False,
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
    lock = json.loads(args.input_lock.read_text(encoding="utf-8"))
    if args.model_key not in manifest["models"]:
        raise SystemExit(f"unknown model key: {args.model_key}")
    if args.model_key not in diagnostic["models"]:
        raise SystemExit(f"model has no diagnostic plan: {args.model_key}")
    model_config = manifest["models"][args.model_key]
    adapter = ModelAdapter(
        args.model_key,
        model_config,
        args.models_root / model_config["local_dir"],
    )
    for case_id in args.case_id:
        for arm in args.arm:
            run_one(args, manifest, diagnostic, lock, adapter, case_id, arm)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
