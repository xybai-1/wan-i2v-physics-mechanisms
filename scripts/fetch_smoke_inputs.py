#!/usr/bin/env python3
"""Fetch and checksum the small conditioning inputs in a frozen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        destination = args.data_root / case["first_image"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".partial")
            urllib.request.urlretrieve(case["first_image_url"], temporary)
            temporary.replace(destination)
        actual = sha256(destination)
        expected = case["first_image_sha256"]
        if actual != expected:
            raise SystemExit(
                f"checksum mismatch for {destination}: expected {expected}, got {actual}"
            )
        print(f"verified {case['video_id']}: {destination} ({actual})")


if __name__ == "__main__":
    main()
