#!/usr/bin/env python3
"""Run one frozen model/case/seed/arm mechanism-matrix cell."""

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
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import CogVideoXImageToVideoPipeline, LTXImageToVideoPipeline, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

from aggregate_attention_probe import AggregateAttentionProbe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", choices=("baseline", "phaselock"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--phaselock-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def posterior_mode(encoded: Any) -> torch.Tensor:
    if hasattr(encoded, "latent_dist"):
        return encoded.latent_dist.mode()
    if hasattr(encoded, "latents"):
        return encoded.latents
    if isinstance(encoded, tuple):
        return encoded[0]
    raise TypeError(f"cannot retrieve latent posterior from {type(encoded)!r}")


class ModelAdapter:
    def __init__(self, model_key: str, config: dict[str, Any], model_path: Path) -> None:
        self.model_key = model_key
        self.config = config
        self.kind = config["pipeline"]
        classes = {
            "wan": WanImageToVideoPipeline,
            "cogvideox": CogVideoXImageToVideoPipeline,
            "ltx": LTXImageToVideoPipeline,
        }
        self.pipe = classes[self.kind].from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
        )
        if self.kind == "wan":
            # Wan's CLIP projection receives FP32 pooled features in Diffusers 0.35.2.
            self.pipe.image_encoder.to(dtype=torch.float32)
        self.pipe.enable_model_cpu_offload()
        if hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()
        if hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        frames = (self.config["num_frames"] - 1) // self.temporal_scale + 1
        height = self.config["height"] // self.spatial_scale
        width = self.config["width"] // self.spatial_scale
        return frames, height, width

    @property
    def temporal_scale(self) -> int:
        if self.kind == "ltx":
            return self.pipe.vae_temporal_compression_ratio
        return self.pipe.vae_scale_factor_temporal

    @property
    def spatial_scale(self) -> int:
        if self.kind == "ltx":
            return self.pipe.vae_spatial_compression_ratio
        return self.pipe.vae_scale_factor_spatial

    def generator(self, seed: int) -> torch.Generator:
        return torch.Generator(device="cuda").manual_seed(seed)

    def call_kwargs(self, prompt: str, image, protocol: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": protocol["negative_prompt"],
            "image": image,
            "height": self.config["height"],
            "width": self.config["width"],
            "num_frames": self.config["num_frames"],
            "guidance_scale": self.config["guidance_scale"],
        }
        if self.kind == "ltx":
            kwargs["frame_rate"] = self.config["fps"]
        return kwargs

    @torch.no_grad()
    def encode_generated_frames(self, frames) -> torch.Tensor:
        """Return normalized PhaseLock layout T,C,H,W using posterior mode."""
        video = self.pipe.video_processor.preprocess_video(
            frames, height=self.config["height"], width=self.config["width"]
        ).to(device=self.pipe._execution_device, dtype=self.pipe.vae.dtype)
        raw = posterior_mode(self.pipe.vae.encode(video))
        if self.kind == "wan":
            mean = torch.tensor(self.pipe.vae.config.latents_mean, device=raw.device, dtype=raw.dtype)
            std = torch.tensor(self.pipe.vae.config.latents_std, device=raw.device, dtype=raw.dtype)
            normalized = (raw - mean.view(1, -1, 1, 1, 1)) / std.view(1, -1, 1, 1, 1)
        elif self.kind == "cogvideox":
            scale = self.pipe.vae_scaling_factor_image
            normalized = raw / scale if self.pipe.vae.config.invert_scale_latents else raw * scale
        else:
            normalized = self.pipe._normalize_latents(raw, self.pipe.vae.latents_mean, self.pipe.vae.latents_std)
        return normalized.squeeze(0).permute(1, 0, 2, 3)

    def to_btc(self, latents: torch.Tensor) -> torch.Tensor:
        if self.kind == "wan":
            return latents.permute(0, 2, 1, 3, 4)
        if self.kind == "cogvideox":
            return latents
        frames, height, width = self.latent_shape
        unpacked = self.pipe._unpack_latents(
            latents,
            frames,
            height,
            width,
            self.pipe.transformer_spatial_patch_size,
            self.pipe.transformer_temporal_patch_size,
        )
        return unpacked.permute(0, 2, 1, 3, 4)

    def from_btc(self, latents: torch.Tensor) -> torch.Tensor:
        if self.kind == "wan":
            return latents.permute(0, 2, 1, 3, 4)
        if self.kind == "cogvideox":
            return latents
        unpacked = latents.permute(0, 2, 1, 3, 4)
        return self.pipe._pack_latents(
            unpacked,
            self.pipe.transformer_spatial_patch_size,
            self.pipe.transformer_temporal_patch_size,
        )

    @torch.no_grad()
    def decode_latents(self, packed_latents: torch.Tensor):
        latents = packed_latents.to(device=self.pipe._execution_device, dtype=self.pipe.vae.dtype)
        if self.kind == "wan":
            mean = torch.tensor(self.pipe.vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
            reciprocal_std = 1.0 / torch.tensor(
                self.pipe.vae.config.latents_std, device=latents.device, dtype=latents.dtype
            )
            latents = latents / reciprocal_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
            video = self.pipe.vae.decode(latents, return_dict=False)[0]
        elif self.kind == "cogvideox":
            video = self.pipe.decode_latents(latents)
        else:
            frames, height, width = self.latent_shape
            latents = self.pipe._unpack_latents(
                latents,
                frames,
                height,
                width,
                self.pipe.transformer_spatial_patch_size,
                self.pipe.transformer_temporal_patch_size,
            )
            latents = self.pipe._denormalize_latents(
                latents,
                self.pipe.vae.latents_mean,
                self.pipe.vae.latents_std,
                self.pipe.vae.config.scaling_factor,
            )
            timestep = None
            if self.pipe.vae.config.timestep_conditioning:
                timestep = torch.zeros(latents.shape[0], device=latents.device, dtype=latents.dtype)
            video = self.pipe.vae.decode(latents, timestep, return_dict=False)[0]
        return self.pipe.video_processor.postprocess_video(video, output_type="pil")[0]


def cosine_to_prior(adapter: ModelAdapter, latents: torch.Tensor, prior: torch.Tensor) -> float:
    current = adapter.to_btc(latents)
    delta = current[:, 1:] - current[:, :-1]
    expected = prior.unsqueeze(0).to(delta.device, delta.dtype)
    if delta.shape != expected.shape:
        raise RuntimeError(f"delta/prior mismatch: {tuple(delta.shape)} vs {tuple(expected.shape)}")
    return float(F.cosine_similarity(delta.float().flatten(1), expected.float().flatten(1)).mean())


def run_one(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    lock: dict[str, Any],
    adapter: ModelAdapter,
    case_id: str,
    seed: int,
    arm: str,
) -> None:
    case = next((item for item in manifest["cases"] if item["case_id"] == case_id), None)
    if case is None:
        raise SystemExit(f"unknown case: {case_id}")
    protocol = manifest["protocol"]
    if seed not in protocol["seeds"]:
        raise SystemExit(f"seed {seed} is not registered")
    image_path = args.data_root / case["first_image"]
    locked_hash = lock["files"][case["case_id"]]["sha256"]
    actual_hash = file_sha256(image_path)
    if actual_hash != locked_hash:
        raise SystemExit(f"conditioning image checksum mismatch: {image_path}")

    model_config = manifest["models"][args.model_key]
    run_dir = args.output_root / args.model_key / case_id / f"seed_{seed}" / arm
    run_dir.mkdir(parents=True, exist_ok=False)
    common = adapter.call_kwargs(case["prompt"], load_image(str(image_path)), protocol)

    few_frames = adapter.pipe(
        **common,
        num_inference_steps=protocol["few_step_count"],
        generator=adapter.generator(seed),
    ).frames[0]
    export_to_video(few_frames, str(run_dir / "few_step.mp4"), fps=model_config["fps"])

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

    probe = AggregateAttentionProbe(
        adapter.pipe.transformer,
        calls_per_step=model_config["transformer_calls_per_step"],
        cfg_layout=model_config["cfg_layout"],
    ).install()
    checkpoint_indices = {step - 1: step for step in protocol["decode_steps_one_based"]}
    snapshots: dict[int, torch.Tensor] = {}
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
        if step_index in checkpoint_indices:
            snapshots[checkpoint_indices[step_index]] = latents.detach().to("cpu", torch.float16)
        trace.append(
            {
                "step_one_based": int(step_index + 1),
                "timestep": float(timestep.detach().float().mean().cpu()),
                "guidance_strength": float(strength),
                "cosine_to_two_step_prior_before_guidance": before,
                "cosine_to_two_step_prior_after_guidance": cosine_to_prior(adapter, latents, motion_prior),
                "pipeline_latent_shape": list(latents.shape),
            }
        )
        return callback_kwargs

    final_frames = adapter.pipe(
        **common,
        num_inference_steps=protocol["num_inference_steps"],
        generator=adapter.generator(seed),
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    ).frames[0]
    probe.remove()
    export_to_video(final_frames, str(run_dir / "final.mp4"), fps=model_config["fps"])
    del final_frames
    for step in protocol["decode_steps_one_based"]:
        if step not in snapshots:
            raise RuntimeError(f"missing requested checkpoint step {step}")
        checkpoint_frames = adapter.decode_latents(snapshots.pop(step))
        export_to_video(
            checkpoint_frames,
            str(run_dir / f"checkpoint_step_{step:03d}.mp4"),
            fps=model_config["fps"],
        )
        del checkpoint_frames
        gc.collect()
        torch.cuda.empty_cache()

    (run_dir / "latent_delta_trace.json").write_text(
        json.dumps(trace, indent=2) + "\n", encoding="utf-8"
    )
    probe.save(run_dir / "attention")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "study_revision": os.environ.get("STUDY_REVISION"),
        "study_id": manifest["study_id"],
        "model_key": args.model_key,
        "model": model_config,
        "case": case,
        "seed": seed,
        "arm": arm,
        "conditioning_image": str(image_path),
        "conditioning_sha256": actual_hash,
        "protocol": protocol,
        "phaselock_revision_expected": manifest["source_revisions"]["phaselock"],
        "phaselock_revision_actual": git_revision(args.phaselock_code.parent),
        "vae_posterior_for_motion_prior": "mode (deterministic)",
        "torch_version": torch.__version__,
        "diffusers_version": __import__("diffusers").__version__,
        "outputs": {
            "few_step": "few_step.mp4",
            "final": "final.mp4",
            "checkpoints": [f"checkpoint_step_{step:03d}.mp4" for step in protocol["decode_steps_one_based"]],
            "latent_delta_trace": "latent_delta_trace.json",
            "attention_stats": "attention/attention_activation_stats.npz",
            "attention_metadata": "attention/attention_activation_metadata.json"
        },
        "raw_latents_or_attention_matrices_saved": False,
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    lock = json.loads(args.input_lock.read_text(encoding="utf-8"))
    if args.model_key not in manifest["models"]:
        raise SystemExit(f"unknown model key: {args.model_key}")
    model_config = manifest["models"][args.model_key]
    model_path = args.models_root / model_config["local_dir"]
    adapter = ModelAdapter(args.model_key, model_config, model_path)
    for case_id in args.case_id:
        for seed in args.seed:
            for arm in args.arm:
                run_dir = args.output_root / args.model_key / case_id / f"seed_{seed}" / arm
                if (run_dir / "run.json").exists():
                    print(f"already complete: {args.model_key} {case_id} {seed} {arm}")
                    continue
                if run_dir.exists():
                    raise SystemExit(f"partial output requires inspection; refusing overwrite: {run_dir}")
                run_one(args, manifest, lock, adapter, case_id, seed, arm)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
