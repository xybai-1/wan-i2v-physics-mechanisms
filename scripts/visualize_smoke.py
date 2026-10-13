#!/usr/bin/env python3
"""Build a compact visual review bundle for the matched smoke experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_trace(run_dir: Path) -> list[dict]:
    return json.loads((run_dir / "latent_delta_trace.json").read_text(encoding="utf-8"))


def sample_frames(video: Path, count: int = 8) -> tuple[list[np.ndarray], list[int]]:
    frames = list(iio.imiter(video, plugin="FFMPEG"))
    if not frames:
        raise RuntimeError(f"no frames decoded from {video}")
    indices = np.linspace(0, len(frames) - 1, min(count, len(frames)), dtype=int).tolist()
    return [frames[index] for index in indices], indices


def save_contact_sheet(rows: list[tuple[str, Path]], output: Path) -> None:
    sampled = [(label, *sample_frames(video)) for label, video in rows]
    columns = max(len(frames) for _, frames, _ in sampled)
    figure, axes = plt.subplots(len(rows), columns, figsize=(2.4 * columns, 2.5 * len(rows)))
    axes = np.asarray(axes).reshape(len(rows), columns)
    for row_index, (label, frames, indices) in enumerate(sampled):
        for column in range(columns):
            axis = axes[row_index, column]
            axis.axis("off")
            if column < len(frames):
                axis.imshow(frames[column])
                axis.set_title(f"{label}\nf={indices[column]}", fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def trace_summary(trace: list[dict]) -> dict:
    before = np.asarray([row["cosine_to_two_step_prior_before_guidance"] for row in trace])
    after = np.asarray([row["cosine_to_two_step_prior_after_guidance"] for row in trace])
    split = max(1, len(trace) // 2)
    return {
        "num_steps": len(trace),
        "initial_before": float(before[0]),
        "final_before": float(before[-1]),
        "early_before_mean": float(before[:split].mean()),
        "late_before_mean": float(before[split:].mean()),
        "final_after": float(after[-1]),
        "minimum_before": float(before.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--phaselock-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    baseline = read_trace(args.baseline_dir)
    phaselock = read_trace(args.phaselock_dir)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        [row["step_index"] for row in baseline],
        [row["cosine_to_two_step_prior_before_guidance"] for row in baseline],
        label="baseline: before",
    )
    axis.plot(
        [row["step_index"] for row in phaselock],
        [row["cosine_to_two_step_prior_before_guidance"] for row in phaselock],
        label="PhaseLock: before",
    )
    axis.plot(
        [row["step_index"] for row in phaselock],
        [row["cosine_to_two_step_prior_after_guidance"] for row in phaselock],
        linestyle="--",
        label="PhaseLock: after guidance",
    )
    axis.axvline(19.5, color="black", linewidth=0.8, alpha=0.35, label="guidance end")
    axis.set(xlabel="Denoising step", ylabel="Cosine similarity to 2-step latent delta")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "latent_delta_similarity.png", dpi=180)
    plt.close(figure)

    save_contact_sheet(
        [
            ("2-step baseline", args.baseline_dir / "few_step.mp4"),
            ("2-step PhaseLock", args.phaselock_dir / "few_step.mp4"),
            ("final baseline", args.baseline_dir / "final.mp4"),
            ("final PhaseLock", args.phaselock_dir / "final.mp4"),
        ],
        args.output_dir / "temporal_contact_sheet.png",
    )

    summary = {
        "baseline": trace_summary(baseline),
        "phaselock": trace_summary(phaselock),
        "interpretation_boundary": {
            "formation_failure": "The two-step video lacks the required ball trajectory/contact/event structure.",
            "preservation_failure": "The two-step video contains a plausible event structure, but the final video loses or corrupts it.",
            "attention_caveat": "Attention and latent similarity are correlational diagnostics, not causal proof."
        },
        "classification": "requires visual review"
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "review.md").write_text(
        "# Smoke review\n\n"
        "Use `temporal_contact_sheet.png` together with both full videos.\n\n"
        "- [ ] The two-step prior contains a coherent rolling-ball trajectory.\n"
        "- [ ] Ball–pin contact occurs in the expected temporal order.\n"
        "- [ ] Pins remain individually visible rather than merging/disappearing.\n"
        "- [ ] The final baseline preserves the early event structure.\n"
        "- [ ] The final PhaseLock output preserves it better than baseline.\n"
        "- [ ] Classification recorded as formation, preservation, mixed, or pass.\n"
        "- [ ] No causal claim is made from attention or cosine similarity alone.\n",
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
