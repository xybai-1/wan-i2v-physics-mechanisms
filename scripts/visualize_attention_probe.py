#!/usr/bin/env python3
"""Create compact, paper-ready summaries of matched Wan attention probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--phaselock-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_arm(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    archive = np.load(path / "attention" / "attention_maps.npz")
    arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads((path / "attention" / "attention_metadata.json").read_text())
    return arrays, metadata


def normalized_entropy(values: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1).astype(np.float64)
    probabilities = flat / np.maximum(flat.sum(axis=1, keepdims=True), 1e-12)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)
    return entropy / np.log(flat.shape[1])


def centroid_track(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, height, width = values.shape
    yy, xx = np.mgrid[0:height, 0:width]
    mass = np.maximum(values.sum(axis=(1, 2)), 1e-12)
    return (
        (values * xx).sum(axis=(1, 2)) / mass,
        (values * yy).sum(axis=(1, 2)) / mass,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    arms = {
        "Baseline": load_arm(args.baseline_dir),
        "PhaseLock": load_arm(args.phaselock_dir),
    }
    reference_meta = arms["Baseline"][1]
    steps = reference_meta["plan"]["steps"]
    cross_layer = max(reference_meta["plan"]["cross_layers"])
    self_layer = max(reference_meta["plan"]["self_layers"])
    labels = list(reference_meta["tokenization"]["positions"])

    # Spatial cross-attention: temporal maximum at the deepest sampled layer.
    figure, axes = plt.subplots(
        len(arms), len(labels) * 2, figsize=(3.0 * len(labels) * 2, 5.4), squeeze=False
    )
    for row, (arm_name, (arrays, _)) in enumerate(arms.items()):
        for label_index, label in enumerate(labels):
            for endpoint_index, step in enumerate((steps[0], steps[-1])):
                key = f"cross_s{step:02d}_l{cross_layer:02d}_{label}"
                panel = arrays[key].max(axis=0)
                axis = axes[row, label_index * 2 + endpoint_index]
                image = axis.imshow(panel, cmap="magma", origin="upper")
                axis.set_title(f"{label}: denoise {step + 1}")
                axis.set_xticks([])
                axis.set_yticks([])
                if label_index == 0 and endpoint_index == 0:
                    axis.set_ylabel(arm_name)
                figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    figure.suptitle(f"Text-normalized cross-attention, layer {cross_layer}")
    figure.tight_layout()
    figure.savefig(args.output_dir / "cross_attention_spatial.png", dpi=180)
    plt.close(figure)

    # Entity centroid paths through latent time, one panel per denoising checkpoint.
    entity_labels = labels[:2]
    primary_label, secondary_label = entity_labels
    figure, axes = plt.subplots(
        len(arms), len(steps), figsize=(2.8 * len(steps), 5.4), squeeze=False
    )
    colors = ("tab:blue", "tab:orange")
    for row, (arm_name, (arrays, _)) in enumerate(arms.items()):
        for column, step in enumerate(steps):
            axis = axes[row, column]
            for label, color in zip(entity_labels, colors):
                values = arrays[f"cross_s{step:02d}_l{cross_layer:02d}_{label}"]
                x, y = centroid_track(values)
                axis.plot(x, y, "-o", color=color, markersize=2.2, linewidth=1.2, label=label)
                axis.scatter(x[0], y[0], color=color, marker="s", s=24)
                axis.scatter(x[-1], y[-1], color=color, marker="*", s=42)
            axis.invert_yaxis()
            axis.set_xlim(0, reference_meta["grid_thw"][2] - 1)
            axis.set_ylim(reference_meta["grid_thw"][1] - 1, 0)
            axis.set_title(f"denoise {step + 1}")
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(arm_name)
            if row == 0 and column == 0:
                axis.legend(fontsize=8, loc="best")
    figure.suptitle(f"Entity cross-attention centroid paths, layer {cross_layer}\n(square=start, star=end)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "cross_attention_centroid_paths.png", dpi=180)
    plt.close(figure)

    # Frame-to-frame sparse self-attention matrices at four denoising checkpoints.
    display_steps = (steps[0], steps[1], steps[-2], steps[-1])
    for relation, title in (
        (f"self_{primary_label}", "primary entity → primary entity"),
        (f"self_{primary_label}_to_{secondary_label}", "primary entity → secondary entity"),
    ):
        figure, axes = plt.subplots(
            len(arms),
            len(display_steps),
            figsize=(3.0 * len(display_steps), 6.8),
            squeeze=False,
            layout="constrained",
        )
        for row, (arm_name, (arrays, _)) in enumerate(arms.items()):
            for column, step in enumerate(display_steps):
                values = arrays[f"{relation}_s{step:02d}_l{self_layer:02d}"]
                axis = axes[row, column]
                image = axis.imshow(values, cmap="viridis", origin="lower", aspect="equal")
                axis.set_title(f"denoise {step + 1}")
                axis.set_xlabel("key frame")
                if column == 0:
                    axis.set_ylabel(f"{arm_name}\nquery frame")
                figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        figure.suptitle(f"Sparse self-attention: {title}, layer {self_layer}", fontsize=15)
        figure.savefig(args.output_dir / f"{relation}_connectivity.png", dpi=180)
        plt.close(figure)

    summary: dict[str, object] = {
        "schema_version": 1,
        "cross_layer": cross_layer,
        "self_layer": self_layer,
        "steps_zero_based": steps,
        "metrics_are_descriptive_not_causal": True,
        "arms": {},
    }
    for arm_name, (arrays, _) in arms.items():
        arm_metrics: dict[str, object] = {}
        for step in steps:
            step_metrics: dict[str, object] = {}
            for label in labels:
                values = arrays[f"cross_s{step:02d}_l{cross_layer:02d}_{label}"]
                x, y = centroid_track(values)
                step_metrics[label] = {
                    "mean_normalized_spatial_entropy": float(normalized_entropy(values, (1, 2)).mean()),
                    "centroid_net_displacement_tokens": float(np.hypot(x[-1] - x[0], y[-1] - y[0])),
                    "mean_cross_attention": float(values.mean()),
                }
            primary = arrays[f"self_{primary_label}_s{step:02d}_l{self_layer:02d}"]
            secondary = arrays[f"self_{primary_label}_to_{secondary_label}_s{step:02d}_l{self_layer:02d}"]
            step_metrics["self_attention"] = {
                "primary_same_frame_mean": float(np.diag(primary).mean()),
                "primary_next_frame_mean": float(np.diag(primary, k=1).mean()),
                "primary_to_secondary_same_frame_mean": float(np.diag(secondary).mean()),
                "primary_to_secondary_total_mean": float(secondary.mean()),
            }
            arm_metrics[str(step)] = step_metrics
        summary["arms"][arm_name.lower()] = arm_metrics
    (args.output_dir / "attention_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "files": sorted(p.name for p in args.output_dir.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
