"""Single-image / folder inference.

    python src/predict.py --ckpt checkpoints/best.pt --image path/to/face.jpg
    python src/predict.py --ckpt checkpoints/best.pt --dir path/to/folder
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from data import build_transforms
from model import build_model

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_model(ck.get("model", "cnn"), pretrained=False).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model


@torch.no_grad()
def predict(model, paths, device, tf):
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
    args = ap.parse_args()
    if not (args.image or args.dir):
        ap.error("pass --image or --dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load(args.ckpt, device)
    tf = build_transforms(train=False)
    paths = ([Path(args.image)] if args.image
             else [p for p in sorted(Path(args.dir).rglob("*")) if p.suffix.lower() in EXT])
    for p, prob, lab in predict(model, paths, device, tf):
        print(f"{lab:4s}  P(fake)={prob:.4f}  {p}")


if __name__ == "__main__":
    main()
