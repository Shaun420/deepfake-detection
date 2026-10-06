"""Quantitative evaluation of the video pipeline + the resampling ablation.

Builds short clips of known ground truth from the held-out seed range, encodes
them at several resolutions, and reports clip-level and per-frame accuracy for
one or more checkpoints.  This is what produces the video results table in the
report.

    python src/video_eval.py --ckpts checkpoints/best.pt checkpoints/cnn_noaug.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from data import make_sample
from model import build_model
from video import analyse_video

TMP = Path("/tmp/video_eval")


def make_clip(path: Path, label: int, seed: int, n_frames: int = 50,
              size: int = 64, fps: int = 25):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    for i in range(n_frames):
        a = (make_sample(seed + i, label) * 255).astype(np.uint8)
        if size != a.shape[0]:
            a = cv2.resize(a, (size, size), interpolation=cv2.INTER_CUBIC)
        vw.write(cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    vw.release()


def load(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = build_model(ck.get("model", "cnn"), pretrained=False).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=["checkpoints/best.pt"])
    ap.add_argument("--clips", type=int, default=20, help="clips per class")
    ap.add_argument("--frames", type=int, default=12, help="frames sampled per clip")
    ap.add_argument("--sizes", nargs="+", type=int, default=[64, 256])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {Path(c).stem: load(c, device) for c in args.ckpts}

    # build the clip corpus once per resolution
    corpus = {}
    for size in args.sizes:
        items = []
        for k in range(args.clips * 2):
            label = k % 2
            p = TMP / f"{size}" / f"clip_{k:03d}_{label}.mp4"
            make_clip(p, label, seed=4_000_000 + k * 1000, size=size)
            items.append((p, label))
        corpus[size] = items

    results = {}
    for name, model in models.items():
        results[name] = {}
        for size in args.sizes:
            clip_ok, frame_acc, mean_probs = [], [], []
            for p, label in corpus[size]:
                r = analyse_video(str(p), model, device, num_frames=args.frames)
                if not r.frames:
                    continue
                clip_ok.append(int((r.verdict == "FAKE") == (label == 1)))
                frame_acc.append(r.accuracy("fake" if label else "real"))
                mean_probs.append(r.mean_prob_fake)
            results[name][size] = {
                "clip_accuracy": float(np.mean(clip_ok)),
                "mean_per_frame_accuracy": float(np.mean(frame_acc)),
                "n_clips": len(clip_ok),
            }
            print(f"{name:12s} {size:>4}px  clip acc {np.mean(clip_ok):.3f}  "
                  f"mean per-frame acc {np.mean(frame_acc):.3f}")

    Path("checkpoints").mkdir(exist_ok=True)
    json.dump(results, open("checkpoints/video_metrics.json", "w"), indent=2)
    print("\nwrote checkpoints/video_metrics.json")


if __name__ == "__main__":
    main()
