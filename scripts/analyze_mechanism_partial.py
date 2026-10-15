#!/usr/bin/env python3
"""Summarize completed baseline/PhaseLock pairs without touching partial runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CHECKPOINTS = (1, 2, 4, 8, 25, 50)
ATTENTION_METRICS = ("rms", "residual_rms", "input_output_cosine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trace(run_dir: Path) -> list[dict]:
    return json.loads((run_dir / "latent_delta_trace.json").read_text(encoding="utf-8"))


def mean(values: np.ndarray) -> float:
    return float(np.mean(values, dtype=np.float64))


def attention_summary(baseline_dir: Path, phaselock_dir: Path) -> dict:
    baseline = np.load(baseline_dir / "attention" / "attention_activation_stats.npz")
    phaselock = np.load(phaselock_dir / "attention" / "attention_activation_stats.npz")
    for key in ("step", "call", "module", "branch"):
        if not np.array_equal(baseline[key], phaselock[key]):
            raise ValueError(f"unaligned attention index {key}: {baseline_dir}")
    metadata = json.loads(
        (baseline_dir / "attention" / "attention_activation_metadata.json").read_text(encoding="utf-8")
    )
    steps = baseline["step"]
    early = steps < 25
    late = steps >= 25
    result: dict[str, object] = {}
    for metric in ATTENTION_METRICS:
        base = baseline[metric].astype(np.float64)
        phase = phaselock[metric].astype(np.float64)
        delta = phase - base
        result[metric] = {
            "mean_signed_delta": mean(delta),
            "mean_absolute_delta": mean(np.abs(delta)),
            "early_mean_absolute_delta": mean(np.abs(delta[early])),
            "late_mean_absolute_delta": mean(np.abs(delta[late])),
            "early_to_late_ratio": float(
                mean(np.abs(delta[early])) / max(mean(np.abs(delta[late])), 1e-12)
            ),
        }
    residual_delta = np.abs(
        phaselock["residual_rms"].astype(np.float64)
        - baseline["residual_rms"].astype(np.float64)
    )
    modules = baseline["module"]
    module_scores = []
    for module_id, module_name in enumerate(metadata["module_names"]):
        mask = early & (modules == module_id)
        module_scores.append(
            {"module": module_name, "early_mean_absolute_residual_delta": mean(residual_delta[mask])}
        )
    result["top_early_residual_modules"] = sorted(
        module_scores, key=lambda row: row["early_mean_absolute_residual_delta"], reverse=True
    )[:5]
    return result


def pair_summary(baseline_dir: Path, phaselock_dir: Path) -> dict:
    baseline_trace = load_trace(baseline_dir)
    phaselock_trace = load_trace(phaselock_dir)
    if len(baseline_trace) != 50 or len(phaselock_trace) != 50:
        raise ValueError(f"expected 50 trace records: {baseline_dir}")
    base_before = np.asarray(
        [row["cosine_to_two_step_prior_before_guidance"] for row in baseline_trace], dtype=np.float64
    )
    phase_before = np.asarray(
        [row["cosine_to_two_step_prior_before_guidance"] for row in phaselock_trace], dtype=np.float64
    )
    phase_after = np.asarray(
        [row["cosine_to_two_step_prior_after_guidance"] for row in phaselock_trace], dtype=np.float64
    )
    checkpoints = {}
    for step in CHECKPOINTS:
        index = step - 1
        checkpoints[str(step)] = {
            "baseline_before": float(base_before[index]),
            "phaselock_before": float(phase_before[index]),
            "phaselock_after": float(phase_after[index]),
            "phaselock_minus_baseline_before": float(phase_before[index] - base_before[index]),
            "instant_guidance_gain": float(phase_after[index] - phase_before[index]),
        }
    baseline_prior_hash = sha256(baseline_dir / "few_step.mp4")
    phaselock_prior_hash = sha256(phaselock_dir / "few_step.mp4")
    return {
        "few_step_prior_identical": baseline_prior_hash == phaselock_prior_hash,
        "few_step_prior_sha256": baseline_prior_hash,
        "checkpoints": checkpoints,
        "latent": {
            "mean_instant_guidance_gain_steps_1_25": mean((phase_after - phase_before)[:25]),
            "mean_phaselock_minus_baseline_steps_1_25": mean((phase_before - base_before)[:25]),
            "mean_phaselock_minus_baseline_steps_26_50": mean((phase_before - base_before)[25:]),
            "final_phaselock_minus_baseline": float(phase_before[-1] - base_before[-1]),
        },
        "attention": attention_summary(baseline_dir, phaselock_dir),
    }


def metric_stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": mean(array),
        "std": float(np.std(array, dtype=np.float64)),
        "median": float(np.median(array)),
        "positive_fraction": float(np.mean(array > 0)),
    }


def aggregate(pairs: list[dict], keys: str | tuple[str, ...]) -> dict:
    if isinstance(keys, str):
        keys = (keys,)
    groups: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        name = " | ".join(str(pair[key]) for key in keys)
        groups[name].append(pair)
    output = {}
    for name, rows in sorted(groups.items()):
        instant = [row["latent"]["mean_instant_guidance_gain_steps_1_25"] for row in rows]
        early = [row["latent"]["mean_phaselock_minus_baseline_steps_1_25"] for row in rows]
        late = [row["latent"]["mean_phaselock_minus_baseline_steps_26_50"] for row in rows]
        final = [row["latent"]["final_phaselock_minus_baseline"] for row in rows]
        attention_ratio = [row["attention"]["residual_rms"]["early_to_late_ratio"] for row in rows]
        output[name] = {
            "pair_count": len(rows),
            "identical_prior_count": sum(row["few_step_prior_identical"] for row in rows),
            "instant_guidance_gain_steps_1_25": metric_stats(instant),
            "phaselock_minus_baseline_steps_1_25": metric_stats(early),
            "phaselock_minus_baseline_steps_26_50": metric_stats(late),
            "final_phaselock_minus_baseline": metric_stats(final),
            "attention_residual_early_to_late_ratio": metric_stats(attention_ratio),
        }
    return output


def main() -> None:
    args = parse_args()
    completed: dict[tuple[str, str, int], dict[str, Path]] = defaultdict(dict)
    run_metadata = {}
    for run_file in sorted(args.output_root.glob("*/*/seed_*/*/run.json")):
        metadata = json.loads(run_file.read_text(encoding="utf-8"))
        key = (metadata["model_key"], metadata["case"]["case_id"], int(metadata["seed"]))
        completed[key][metadata["arm"]] = run_file.parent
        run_metadata[key] = metadata
    pairs = []
    for key, arms in sorted(completed.items()):
        if set(arms) != {"baseline", "phaselock"}:
            continue
        metadata = run_metadata[key]
        summary = pair_summary(arms["baseline"], arms["phaselock"])
        summary.update(
            {
                "model_key": key[0],
                "case_id": key[1],
                "seed": key[2],
                "benchmark": metadata["case"]["benchmark"],
                "complexity": metadata["case"]["complexity"],
            }
        )
        pairs.append(summary)
    report = {
        "completed_run_count": sum(len(arms) for arms in completed.values()),
        "completed_pair_count": len(pairs),
        "pairs": pairs,
        "aggregate_by_model": aggregate(pairs, "model_key"),
        "aggregate_by_benchmark": aggregate(pairs, "benchmark"),
        "aggregate_by_case": aggregate(pairs, "case_id"),
        "aggregate_by_complexity": aggregate(pairs, "complexity"),
        "aggregate_by_model_and_benchmark": aggregate(pairs, ("model_key", "benchmark")),
        "aggregate_by_model_and_complexity": aggregate(pairs, ("model_key", "complexity")),
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
