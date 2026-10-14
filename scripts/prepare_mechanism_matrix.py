#!/usr/bin/env python3
"""Fetch, generate, and checksum frozen mechanism-matrix conditioning images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fetch-public", "generate-phygen", "lock"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image-model", type=Path)
    parser.add_argument("--lock-output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def fetch_public(manifest: dict, data_root: Path) -> None:
    for case in manifest["cases"]:
        url = case.get("first_image_url")
        if not url:
            continue
        destination = data_root / case["first_image"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".partial")
            request = urllib.request.Request(url, headers={"User-Agent": "mechanism-matrix-v1"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            os.replace(temporary, destination)
        width, height = verify_image(destination)
        print(json.dumps({"case_id": case["case_id"], "path": str(destination), "size": [width, height]}))


def generate_phygen(manifest: dict, data_root: Path, image_model: Path) -> None:
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        str(image_model), torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True
    )
    pipe.enable_model_cpu_offload()
    for case in manifest["cases"]:
        specification = case.get("first_image_generator")
        if not specification:
            continue
        destination = data_root / case["first_image"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            verify_image(destination)
            print(f"already exists: {destination}")
            continue
        image = pipe(
            prompt=specification["prompt"],
            height=480,
            width=832,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=torch.Generator(device="cuda").manual_seed(specification["seed"]),
        ).images[0]
        image.save(destination)
        verify_image(destination)
        print(f"generated: {destination}")


def write_lock(manifest: dict, data_root: Path, destination: Path) -> None:
    files = {}
    for case in manifest["cases"]:
        path = data_root / case["first_image"]
        if not path.exists():
            raise SystemExit(f"missing conditioning image: {path}")
        width, height = verify_image(path)
        files[case["case_id"]] = {
            "relative_path": case["first_image"],
            "sha256": sha256(path),
            "width": width,
            "height": height,
            "source_url": case.get("first_image_url"),
            "generator": case.get("first_image_generator"),
        }
    lock = {
        "schema_version": 1,
        "study_id": manifest["study_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.command == "fetch-public":
        fetch_public(manifest, args.data_root)
    elif args.command == "generate-phygen":
        if args.image_model is None:
            raise SystemExit("--image-model is required for generate-phygen")
        generate_phygen(manifest, args.data_root, args.image_model)
    else:
        if args.lock_output is None:
            raise SystemExit("--lock-output is required for lock")
        write_lock(manifest, args.data_root, args.lock_output)


if __name__ == "__main__":
    main()
