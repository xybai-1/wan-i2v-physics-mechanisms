#!/usr/bin/env python3
"""Rerun one frozen smoke arm with sparse Wan attention instrumentation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

from attention_probe import ProbePlan, WanAttentionProbeState, install_wan_attention_probes, locate_text_spans
from run_smoke import cosine_to_prior, encode_wan_video_to_latents, file_sha256, git_revision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("baseline", "phaselock"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--phaselock-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    case = next(item for item in payload["cases"] if item["video_id"] == args.case_id)
    generation = payload["generation"]
    image_path = args.data_root / case["first_image"]
    if file_sha256(image_path) != case["first_image_sha256"]:
        raise SystemExit(f"conditioning image checksum mismatch: {image_path}")

    run_dir = args.output_root / args.case_id / f"seed_{args.seed}" / args.arm
    run_dir.mkdir(parents=True, exist_ok=False)
    pipe = WanImageToVideoPipeline.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    )
    pipe.image_encoder.to(dtype=torch.float32)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    common = {
        "prompt": case["prompt"],
        "image": load_image(str(image_path)),
        "height": generation["height"],
        "width": generation["width"],
        "num_frames": generation["num_frames"],
        "guidance_scale": generation["guidance_scale"],
    }

    def seeded_generator() -> torch.Generator:
        return torch.Generator(device="cuda").manual_seed(args.seed)

    few_result = pipe(
        **common, num_inference_steps=generation["few_step_count"], generator=seeded_generator()
    ).frames[0]
    export_to_video(few_result, str(run_dir / "few_step.mp4"), fps=generation["fps"])
    sys.path.insert(0, str(args.phaselock_code))
    from phaselock.guidance import LatentDeltaGuidance, extract_motion_prior

    few_latents = encode_wan_video_to_latents(pipe, few_result)
    motion_prior = extract_motion_prior(few_latents).detach()
    guide = LatentDeltaGuidance(
        motion_prior=motion_prior,
        guidance_strength=generation["phaselock_guidance_strength"],
        guide_start=generation["phaselock_guide_start"],
        guide_end=generation["phaselock_guide_end"],
        total_steps=generation["num_inference_steps"],
    )
    del few_result, few_latents
    gc.collect()
    torch.cuda.empty_cache()

    targets = case.get(
        "attention_targets", {"ball": "bowling ball", "pins": "pins", "impact": "hitting"}
    )
    positions, token_metadata = locate_text_spans(pipe.tokenizer, case["prompt"], targets)
    latent_t = (generation["num_frames"] - 1) // pipe.vae_scale_factor_temporal + 1
    grid = (
        latent_t,
        generation["height"] // pipe.vae_scale_factor_spatial // pipe.transformer.config.patch_size[1],
        generation["width"] // pipe.vae_scale_factor_spatial // pipe.transformer.config.patch_size[2],
    )
    plan = ProbePlan()
    probe = WanAttentionProbeState(positions, token_metadata, grid, plan)
    install_wan_attention_probes(pipe.transformer, probe)

    trace: list[dict] = []
    snapshots: dict[str, torch.Tensor] = {}

    def callback(active_pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        before = cosine_to_prior(latents, motion_prior)
        strength = guide.compute_schedule(step_index) if args.arm == "phaselock" else 0.0
        if args.arm == "phaselock":
            guided = guide(
                active_pipe, step_index, timestep, {"latents": latents.permute(0, 2, 1, 3, 4)}
            )["latents"]
            latents = guided.permute(0, 2, 1, 3, 4)
            callback_kwargs["latents"] = latents
        if step_index in plan.steps:
            snapshots[f"step_{step_index:02d}"] = latents.detach().to("cpu", torch.float16)
        trace.append(
            {
                "step_index": int(step_index),
                "timestep": float(timestep.detach().float().cpu()),
                "guidance_strength": float(strength),
                "cosine_to_two_step_prior_before_guidance": before,
                "cosine_to_two_step_prior_after_guidance": cosine_to_prior(latents, motion_prior),
            }
        )
        return callback_kwargs

    frames = pipe(
        **common,
        num_inference_steps=generation["num_inference_steps"],
        generator=seeded_generator(),
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    ).frames[0]
    final_path = run_dir / "final.mp4"
    export_to_video(frames, str(final_path), fps=generation["fps"])
    (run_dir / "latent_delta_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    torch.save(snapshots, run_dir / "latent_snapshots.pt")
    probe.save(run_dir / "attention")

    reference_path = args.reference_root / args.case_id / f"seed_{args.seed}" / args.arm / "final.mp4"
    reference_sha = file_sha256(reference_path) if reference_path.exists() else None
    output_sha = file_sha256(final_path)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "study_revision": os.environ.get("STUDY_REVISION"),
        "arm": args.arm,
        "benchmark": payload["benchmark"],
        "case_id": args.case_id,
        "seed": args.seed,
        "model_revision": payload["model_revision"],
        "phaselock_revision_actual": git_revision(args.phaselock_code.parent),
        "probe_plan": probe.token_metadata | {
            "steps": list(plan.steps), "cross_layers": list(plan.cross_layers), "self_layers": list(plan.self_layers)
        },
        "noninvasive_video_check": {
            "reference": str(reference_path),
            "reference_sha256": reference_sha,
            "probe_sha256": output_sha,
            "identical": reference_sha == output_sha if reference_sha else None,
        },
        "outputs": {
            "final_video": "final.mp4",
            "attention": "attention/attention_maps.npz",
            "attention_metadata": "attention/attention_metadata.json",
            "latent_snapshots": "latent_snapshots.pt",
        },
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
