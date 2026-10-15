#!/usr/bin/env python3
"""Validate all completed mechanism-matrix output cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import imageio.v3 as iio


CHECKPOINTS = (1, 2, 4, 8, 25, 50)
ARMS = ("baseline", "phaselock")
SEEDS = (42, 123, 777)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_video(path: Path, ffprobe: str) -> dict:
    if shutil.which(ffprobe) is None:
        frames = 0
        width = height = 0
        for array in iio.imiter(path, plugin="FFMPEG"):
            frames += 1
            height, width = array.shape[:2]
        if frames <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"invalid decoded video: {path}")
        return {"codec": "decoded_by_imageio", "width": width, "height": height, "frames": frames}
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,width,height,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    if not payload.get("streams"):
        raise ValueError(f"no decodable video stream: {path}")
    stream = payload["streams"][0]
    frames = int(stream.get("nb_read_frames") or 0)
    if frames <= 0 or int(stream["width"]) <= 0 or int(stream["height"]) <= 0:
        raise ValueError(f"invalid decoded video metadata: {path}: {stream}")
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": frames,
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    model_keys = list(manifest["models"])
    case_ids = [row["case_id"] for row in manifest["cases"]]
    expected = {
        (model, case, seed, arm)
        for model in model_keys
        for case in case_ids
        for seed in SEEDS
        for arm in ARMS
    }

    runs: dict[tuple[str, str, int, str], tuple[Path, dict]] = {}
    study_revisions = Counter()
    slurm_jobs = Counter()
    video_summaries = Counter()
    attention_record_counts = Counter()
    module_counts = Counter()
    errors: list[str] = []

    for run_file in sorted(args.output_root.glob("*/*/seed_*/*/run.json")):
        try:
            metadata = json.loads(run_file.read_text(encoding="utf-8"))
            key = (
                metadata["model_key"],
                metadata["case"]["case_id"],
                int(metadata["seed"]),
                metadata["arm"],
            )
            if key in runs:
                raise ValueError(f"duplicate cell {key}")
            runs[key] = (run_file.parent, metadata)
            study_revisions[metadata["study_revision"]] += 1
            slurm_jobs[str(metadata.get("slurm_job_id", "unknown"))] += 1
        except Exception as exc:  # keep auditing remaining cells
            errors.append(f"{run_file}: {exc}")

    missing = sorted(expected - set(runs))
    unexpected = sorted(set(runs) - expected)
    if missing:
        errors.append(f"missing cells: {missing}")
    if unexpected:
        errors.append(f"unexpected cells: {unexpected}")

    pairs: dict[tuple[str, str, int], dict[str, Path]] = defaultdict(dict)
    conditioning_hashes: dict[str, set[str]] = defaultdict(set)
    for key, (run_dir, metadata) in sorted(runs.items()):
        model, case, seed, arm = key
        pairs[(model, case, seed)][arm] = run_dir
        conditioning_hashes[case].add(metadata["conditioning_sha256"])
        try:
            if metadata["raw_latents_or_attention_matrices_saved"] is not False:
                raise ValueError("raw tensors unexpectedly saved")
            for step in CHECKPOINTS:
                checkpoint = run_dir / f"checkpoint_step_{step:03d}.mp4"
                if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                    raise ValueError(f"missing/empty checkpoint {checkpoint.name}")

            for name in ("few_step.mp4", "final.mp4"):
                path = run_dir / name
                if not path.is_file() or path.stat().st_size <= 0:
                    raise ValueError(f"missing/empty video {name}")
                summary = probe_video(path, args.ffprobe)
                video_summaries[(model, name, summary["frames"], summary["width"], summary["height"])] += 1

            trace = json.loads((run_dir / "latent_delta_trace.json").read_text(encoding="utf-8"))
            if len(trace) != 50:
                raise ValueError(f"latent trace has {len(trace)} records")
            required_trace = {
                "cosine_to_two_step_prior_before_guidance",
                "cosine_to_two_step_prior_after_guidance",
            }
            for index, row in enumerate(trace):
                if not required_trace.issubset(row):
                    raise ValueError(f"trace record {index} missing fields")
                step = row.get("step_one_based")
                if step is None and "step_index_zero_based" in row:
                    step = int(row["step_index_zero_based"]) + 1
                if step != index + 1:
                    raise ValueError(
                        f"trace record {index} has unexpected step value {step}"
                    )
                values = [row[field] for field in required_trace]
                if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
                    raise ValueError(f"trace record {index} is non-finite")

            stats_path = run_dir / "attention" / "attention_activation_stats.npz"
            metadata_path = run_dir / "attention" / "attention_activation_metadata.json"
            stats = np.load(stats_path)
            attention_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            required_arrays = {
                "step", "call", "module", "branch", "rms", "residual_rms", "input_output_cosine"
            }
            if not required_arrays.issubset(stats.files):
                raise ValueError(f"attention stats missing arrays: {required_arrays - set(stats.files)}")
            lengths = {name: len(stats[name]) for name in required_arrays}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"unaligned attention arrays: {lengths}")
            record_count = next(iter(lengths.values()))
            if record_count <= 0:
                raise ValueError("empty attention stats")
            for name in ("rms", "residual_rms", "input_output_cosine"):
                if not np.isfinite(stats[name]).all():
                    raise ValueError(f"non-finite attention values in {name}")
            if int(stats["step"].min()) != 0 or int(stats["step"].max()) != 49:
                raise ValueError("attention stats do not span all 50 steps")
            attention_record_counts[(model, record_count)] += 1
            module_counts[(model, len(attention_meta["module_names"]))] += 1
        except Exception as exc:
            errors.append(f"{key}: {exc}")

    identical_prior_pairs = 0
    for pair_key, arms in sorted(pairs.items()):
        if set(arms) != set(ARMS):
            errors.append(f"unpaired cell {pair_key}: {sorted(arms)}")
            continue
        baseline_hash = sha256(arms["baseline"] / "few_step.mp4")
        phaselock_hash = sha256(arms["phaselock"] / "few_step.mp4")
        if baseline_hash != phaselock_hash:
            errors.append(f"few-step prior mismatch: {pair_key}")
        else:
            identical_prior_pairs += 1

    for case, hashes in sorted(conditioning_hashes.items()):
        if len(hashes) != 1:
            errors.append(f"conditioning image mismatch for {case}: {sorted(hashes)}")
    if len(study_revisions) != 1:
        errors.append(f"multiple study revisions found: {dict(study_revisions)}")

    report = {
        "valid": not errors,
        "expected_run_count": len(expected),
        "observed_run_count": len(runs),
        "expected_pair_count": len(expected) // 2,
        "identical_few_step_prior_pairs": identical_prior_pairs,
        "conditioning_cases_with_one_hash": sum(len(hashes) == 1 for hashes in conditioning_hashes.values()),
        "study_revisions": dict(study_revisions),
        "slurm_job_ids": dict(slurm_jobs),
        "video_summaries": [
            {
                "model_key": key[0],
                "video": key[1],
                "frames": key[2],
                "width": key[3],
                "height": key[4],
                "count": count,
            }
            for key, count in sorted(video_summaries.items())
        ],
        "attention_record_counts": [
            {"model_key": key[0], "records_per_run": key[1], "run_count": count}
            for key, count in sorted(attention_record_counts.items())
        ],
        "attention_module_counts": [
            {"model_key": key[0], "modules_per_run": key[1], "run_count": count}
            for key, count in sorted(module_counts.items())
        ],
        "errors": errors,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
