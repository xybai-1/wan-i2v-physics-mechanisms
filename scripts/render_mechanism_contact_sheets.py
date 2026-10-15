#!/usr/bin/env python3
"""Render compact baseline/PhaseLock temporal contact sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("sheet_root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--frame-width", type=int, default=240)
    parser.add_argument(
        "--video-name",
        choices=("final.mp4", "few_step.mp4"),
        default="final.mp4",
    )
    return parser.parse_args()


def sample_frames(path: Path, frame_count: int, columns: int, width: int) -> list[Image.Image]:
    indices = np.linspace(0, frame_count - 1, columns, dtype=int).tolist()
    wanted = set(indices)
    selected: dict[int, Image.Image] = {}
    for index, array in enumerate(iio.imiter(path, plugin="FFMPEG")):
        if index in wanted:
            image = Image.fromarray(array).convert("RGB")
            height = round(image.height * width / image.width)
            selected[index] = image.resize((width, height), Image.Resampling.LANCZOS)
        if index >= indices[-1]:
            break
    missing = [index for index in indices if index not in selected]
    if missing:
        raise ValueError(f"could not decode selected frames {missing} from {path}")
    return [selected[index] for index in indices]


def render_pair(
    baseline: Path,
    phaselock: Path,
    destination: Path,
    columns: int,
    width: int,
    video_name: str,
) -> None:
    metadata = json.loads((baseline / "run.json").read_text(encoding="utf-8"))
    frame_count = int(metadata["model"]["num_frames"])
    rows = [
        ("baseline", sample_frames(baseline / video_name, frame_count, columns, width)),
        ("PhaseLock", sample_frames(phaselock / video_name, frame_count, columns, width)),
    ]
    label_width = 92
    title_height = 34
    row_height = rows[0][1][0].height
    canvas = Image.new(
        "RGB",
        (label_width + columns * width, title_height + len(rows) * row_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    title = (
        f"{metadata['model_key']} | {metadata['case']['case_id']} | "
        f"seed {metadata['seed']} | {video_name}"
    )
    draw.text((8, 10), title, fill="black", font=font)
    for row_index, (label, frames) in enumerate(rows):
        y = title_height + row_index * row_height
        draw.text((8, y + row_height // 2 - 6), label, fill="black", font=font)
        for column, image in enumerate(frames):
            canvas.paste(image, (label_width + column * width, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=88, optimize=True)


def main() -> None:
    args = parse_args()
    count = 0
    for baseline in sorted(args.output_root.glob(f"*/*/seed_{args.seed}/baseline")):
        phaselock = baseline.parent / "phaselock"
        if not (baseline / "run.json").is_file() or not (phaselock / "run.json").is_file():
            continue
        model = baseline.parents[2].name
        case = baseline.parents[1].name
        destination = args.sheet_root / model / f"{case}_seed_{args.seed}.jpg"
        render_pair(
            baseline,
            phaselock,
            destination,
            args.columns,
            args.frame_width,
            args.video_name,
        )
        count += 1
    print(json.dumps({"contact_sheets": count, "output": str(args.sheet_root)}, indent=2))


if __name__ == "__main__":
    main()
