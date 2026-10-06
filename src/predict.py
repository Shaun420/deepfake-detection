"""Inference on an image, a folder of images, or a video.

    python src/predict.py --ckpt checkpoints/best.pt --image face.jpg
    python src/predict.py --ckpt checkpoints/best.pt --dir  folder/
    python src/predict.py --ckpt checkpoints/best.pt --video clip.mp4 --frames 16

The CNN is image-only. For a video, still frames are sampled uniformly across
the clip, the largest face in each frame is cropped with a Haar cascade, every
crop is classified independently, and the per-frame scores are aggregated into
a clip-level verdict (see src/video.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from data import build_transforms
from model import build_model
from video import analyse_video, format_report, is_video, save_montage

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_model(ck.get("model", "cnn"), pretrained=False).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model


@torch.no_grad()
def predict_images(model, paths, device, tf):
    out = []
    for p in paths:
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        prob = F.softmax(model(x), 1)[0, 1].item()
        out.append((p, prob, "FAKE" if prob >= 0.5 else "REAL"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--image")
    ap.add_argument("--dir")
    ap.add_argument("--video")
    ap.add_argument("--frames", type=int, default=16,
                    help="number of still frames to sample from the video")
    ap.add_argument("--no-fallback", action="store_true",
                    help="skip frames with no detected face instead of centre-cropping")
    ap.add_argument("--label", choices=["real", "fake"],
                    help="ground-truth label of the clip; enables mean per-frame accuracy")
    ap.add_argument("--montage", default="figures/video_frames.png")
    args = ap.parse_args()
    if not (args.image or args.dir or args.video):
        ap.error("pass --image, --dir or --video")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load(args.ckpt, device)
    tf = build_transforms(train=False)

    if args.video:
        if not is_video(args.video):
            print(f"warning: {args.video} does not look like a video container")
        r = analyse_video(args.video, model, device, num_frames=args.frames,
                          fallback_centre_crop=not args.no_fallback, transform=tf)
        print(format_report(r, args.label))
        if out := save_montage(r, args.montage):
            print(f"\nper-frame montage written to {out}")
        return

    paths = ([Path(args.image)] if args.image
             else [p for p in sorted(Path(args.dir).rglob("*")) if p.suffix.lower() in EXT])
    for p, prob, lab in predict_images(model, paths, device, tf):
        print(f"{lab:4s}  P(fake)={prob:.4f}  {p}")


if __name__ == "__main__":
    main()
