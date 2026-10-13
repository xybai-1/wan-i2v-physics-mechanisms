#!/usr/bin/env python3
"""Run one matched Wan-I2V baseline or PhaseLock smoke arm.

Both arms independently recreate the same deterministic two-step prior. The
baseline records its denoising trajectory without modifying it; PhaseLock
records the same measurements and applies the published latent-delta update.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image


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
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def cosine_to_prior(latents_bcthw: torch.Tensor, prior_tchw: torch.Tensor) -> float:
    current = latents_bcthw.permute(0, 2, 1, 3, 4)
    delta = current[:, 1:] - current[:, :-1]
    prior = prior_tchw.unsqueeze(0).to(delta.device, delta.dtype)
    if delta.shape != prior.shape:
        raise RuntimeError(f"delta/prior mismatch: {tuple(delta.shape)} vs {tuple(prior.shape)}")
    return float(F.cosine_similarity(delta.float().flatten(1), prior.float().flatten(1)).mean())


def encode_wan_video_to_latents(pipe, frames) -> torch.Tensor:
    """Encode generated frames with Wan's native latent normalization.

    PhaseLock's released helper assumes a Stable-Diffusion-style scalar
    ``scaling_factor``. AutoencoderKLWan instead defines one mean and standard
    deviation per latent channel. This follows WanImageToVideoPipeline's own
    encode/decode convention and returns PhaseLock's expected T,C,H,W layout.
    Posterior mode is used so the same generated prior has one deterministic
    representation.
    """

    frame_tensors = torch.stack([TVF.to_tensor(frame) for frame in frames])
    video = (2.0 * frame_tensors - 1.0).permute(1, 0, 2, 3).unsqueeze(0)
    video = video.to(device=pipe._execution_device, dtype=pipe.vae.dtype)
    with torch.no_grad():
        raw = pipe.vae.encode(video).latent_dist.mode()
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    std = torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    mean = mean.to(raw.device, raw.dtype)
    std = std.to(raw.device, raw.dtype)
    normalized = (raw - mean) / std
    return normalized.squeeze(0).permute(1, 0, 2, 3)


def main() -> None:
    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    case = next((item for item in payload["cases"] if item["video_id"] == args.case_id), None)
    if case is None:
        raise SystemExit(f"unknown case: {args.case_id}")
    if args.seed not in payload["seeds"]:
        raise SystemExit(f"seed {args.seed} is not pre-registered")

    generation = payload["generation"]
    image_path = args.data_root / case["first_image"]
    if file_sha256(image_path) != case["first_image_sha256"]:
        raise SystemExit(f"conditioning image checksum mismatch: {image_path}")

    run_dir = args.output_root / args.case_id / f"seed_{args.seed}" / args.arm
    run_dir.mkdir(parents=True, exist_ok=False)
    trace_path = run_dir / "latent_delta_trace.json"
    metadata_path = run_dir / "run.json"

    pipe = WanImageToVideoPipeline.from_pretrained(
        str(args.model),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    # Diffusers 0.35.2 preprocesses Wan's CLIP pixels as FP32. Loading every
    # component with the global BF16 dtype also converts visual_projection,
    # which makes CLIP's FP32 pooled output fail at the final linear layer.
    # Keep the small image encoder in its native FP32 while the transformer and
    # VAE remain BF16. The returned hidden states are cast to transformer_dtype
    # by WanImageToVideoPipeline before denoising.
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
        **common,
        num_inference_steps=generation["few_step_count"],
        generator=seeded_generator(),
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

    trace: list[dict] = []

    def callback(active_pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        before = cosine_to_prior(latents, motion_prior)
        strength = guide.compute_schedule(step_index) if args.arm == "phaselock" else 0.0
        if args.arm == "phaselock":
            as_btc = latents.permute(0, 2, 1, 3, 4)
            guided = guide(active_pipe, step_index, timestep, {"latents": as_btc})["latents"]
            latents = guided.permute(0, 2, 1, 3, 4)
            callback_kwargs["latents"] = latents
        trace.append(
            {
                "step_index": int(step_index),
                "timestep": float(timestep.detach().float().cpu()),
                "guidance_strength": float(strength),
                "cosine_to_two_step_prior_before_guidance": before,
                "cosine_to_two_step_prior_after_guidance": cosine_to_prior(latents, motion_prior),
                "latent_shape_bcthw": list(latents.shape),
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
    export_to_video(frames, str(run_dir / "final.mp4"), fps=generation["fps"])
    trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "study_revision": os.environ.get("STUDY_REVISION"),
        "arm": args.arm,
        "benchmark": payload["benchmark"],
        "case_id": args.case_id,
        "prompt": case["prompt"],
        "seed": args.seed,
        "conditioning_image": str(image_path),
        "conditioning_sha256": case["first_image_sha256"],
        "model_id": payload["model_id"],
        "model_revision": payload["model_revision"],
        "phaselock_revision_expected": payload["phaselock_revision"],
        "phaselock_revision_actual": git_revision(args.phaselock_code.parent),
        "generation": generation,
        "wan_phaselock_adapter": {
            "latent_layout": "Wan B,C,T,H,W <-> PhaseLock B,T,C,H,W",
            "vae_normalization": "(z - latents_mean) / latents_std",
            "vae_posterior": "mode"
        },
        "torch_version": torch.__version__,
        "diffusers_version": __import__("diffusers").__version__,
        "outputs": {
            "few_step_video": "few_step.mp4",
            "final_video": "final.mp4",
            "latent_delta_trace": trace_path.name,
        },
        "attention_exports": "not_enabled_in_smoke_generation; instrumentation follows only after generation/artifact validation",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
