"""Render a short demo clip from the synthetic dataset so the video path can be
tested without downloading real footage.

    python tools/make_demo_video.py --label fake --out samples/demo_fake.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import make_sample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=["real", "fake"], default="fake")
    ap.add_argument("--out", default="samples/demo.mp4")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", type=int, default=64,
                    help="frame size; keep at the native 64 px -- upscaling the "
                         "synthetic frames injects resampling artefacts of its own")
    ap.add_argument("--seed", type=int, default=5_000_000)
    args = ap.parse_args()

    label = 1 if args.label == "fake" else 0
    n = int(args.seconds * args.fps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (args.size, args.size))
    for i in range(n):
        # a new draw per frame keeps the manipulation statistics but varies noise
        a = make_sample(args.seed + i, label)
        frame = cv2.resize((a * 255).astype(np.uint8), (args.size, args.size),
                           interpolation=cv2.INTER_NEAREST)
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"wrote {args.out}  ({n} frames, {args.seconds}s @ {args.fps}fps, "
          f"ground truth = {args.label})")


if __name__ == "__main__":
    main()
