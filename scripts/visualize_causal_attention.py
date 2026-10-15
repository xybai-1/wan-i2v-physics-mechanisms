#!/usr/bin/env python3
"""Render compact, dependency-light causal-attention diagnostic grids."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def safe(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def heatmap(value: np.ndarray, size: int = 156) -> Image.Image:
    array = np.asarray(value, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        normalized = np.zeros_like(array)
    else:
        low, high = np.percentile(finite, [2, 98])
        if high <= low:
            high = low + 1e-6
        normalized = np.clip((array - low) / (high - low), 0, 1)
    red = (255 * normalized).astype(np.uint8)
    blue = (255 * (1 - normalized)).astype(np.uint8)
    green = (255 * (1 - np.abs(2 * normalized - 1))).astype(np.uint8)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(rgb, mode="RGB").resize((size, size), Image.Resampling.NEAREST)


def grid_figure(
    tiles: dict[tuple[int, int], np.ndarray],
    rows: list[int],
    columns: list[int],
    title: str,
    path: Path,
) -> None:
    tile = 156
    label_height = 28
    left = 72
    top = 48
    canvas = Image.new(
        "RGB",
        (left + len(columns) * tile, top + len(rows) * (tile + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black")
    for column_index, column in enumerate(columns):
        draw.text((left + column_index * tile + 4, 28), f"layer {column}", fill="black")
    for row_index, row in enumerate(rows):
        y = top + row_index * (tile + label_height)
        draw.text((5, y + tile // 2), f"step {row + 1}", fill="black")
        for column_index, column in enumerate(columns):
            image = heatmap(tiles[(row, column)], tile)
            canvas.paste(image, (left + column_index * tile, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (args.run_dir / "attention" / "attention_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    output_dir = args.run_dir / "visualizations"
    output_dir.mkdir(exist_ok=False)
    summary: dict[str, dict] = {}
    with np.load(args.run_dir / "attention" / "attention_maps.npz") as arrays:
        for name in arrays.files:
            value = arrays[name].astype(np.float32)
            record = {
                "shape": list(value.shape),
                "mean": float(value.mean()),
                "std": float(value.std()),
                "min": float(value.min()),
                "max": float(value.max()),
                "p95": float(np.percentile(value, 95)),
            }
            if name.startswith("self_"):
                total = float(value.sum())
                record.update(
                    {
                        "diagonal_fraction": float(np.trace(value) / total) if total else 0.0,
                        "future_key_fraction": float(np.triu(value, 1).sum() / total) if total else 0.0,
                        "past_key_fraction": float(np.tril(value, -1).sum() / total) if total else 0.0,
                    }
                )
            else:
                temporal = value.mean(axis=(1, 2))
                record["peak_frame"] = int(np.argmax(temporal))
                record["temporal_profile"] = temporal.tolist()
            summary[name] = record

        steps = metadata["plan"]["steps"]
        entity_layers = metadata["plan"]["entity_map_layers"]
        labels = list(metadata["tokenization"]["positions"])
        for label in labels:
            tiles = {}
            for step in steps:
                for layer in entity_layers:
                    name = f"entity_qk_s{step:02d}_l{layer:02d}_{safe(label)}"
                    tiles[(step, layer)] = arrays[name].astype(np.float32).mean(axis=0)
            grid_figure(
                tiles,
                steps,
                entity_layers,
                f"{run['model_key']} | {run['case']['case_id']} | {run['arm']} | entity={label}",
                output_dir / f"entity_{safe(label)}.png",
            )
        self_layers = [item[0] for item in metadata["plan"]["self_source_layer_pairs"]]
        for source, target in zip(labels[:-1], labels[1:]):
            tiles = {}
            for step in steps:
                for layer in self_layers:
                    name = (
                        f"self_s{step:02d}_l{layer:02d}_{safe(source)}_to_{safe(target)}"
                    )
                    tiles[(step, layer)] = arrays[name].astype(np.float32)
            grid_figure(
                tiles,
                steps,
                self_layers,
                f"{run['model_key']} | {run['case']['case_id']} | {run['arm']} | {source} -> {target}",
                output_dir / f"directed_{safe(source)}_to_{safe(target)}.png",
            )
    (output_dir / "attention_summary.json").write_text(
        json.dumps(
            {
                "model_key": run["model_key"],
                "case_id": run["case"]["case_id"],
                "arm": run["arm"],
                "arrays": summary,
                "interpretation": metadata["interpretation"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
