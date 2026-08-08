#!/usr/bin/env python3
"""Export spaced JPEGs from recorded MP4 files for daughter-model labeling."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--every-seconds", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if args.every_seconds <= 0:
        parser.error("--every-seconds must be positive")
    if shutil.which(args.ffmpeg) is None:
        parser.error(f"ffmpeg not found: {args.ffmpeg}")

    videos = sorted(args.input.rglob("*.mp4"))
    args.output.mkdir(parents=True, exist_ok=True)
    for index, video in enumerate(videos, start=1):
        prefix = f"{index:05d}_{video.stem}"
        pattern = args.output / f"{prefix}_%05d.jpg"
        command = [
            args.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-an",
            "-vf",
            f"fps=1/{args.every_seconds:g},scale={max(32, args.width)}:-2",
            "-q:v",
            "4",
            str(pattern),
        ]
        subprocess.run(command, check=True)
    print(f"exported frames from {len(videos)} videos to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
