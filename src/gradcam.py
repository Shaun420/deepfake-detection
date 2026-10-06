"""Grad-CAM explainability: where does the detector look when it says "fake"?

    python src/gradcam.py --ckpt checkpoints/best.pt --n 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data import MEAN, STD, SyntheticFaceDataset
from model import build_model


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)
        logits.gather(1, class_idx.view(-1, 1)).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)           # GAP over spatial dims
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        cam = cam - cam.amin((1, 2), keepdim=True)
        cam = cam / (cam.amax((1, 2), keepdim=True) + 1e-8)
        return cam.detach().cpu().numpy(), F.softmax(logits, 1).detach().cpu().numpy()


def denorm(t):
    m = torch.tensor(MEAN).view(3, 1, 1)
    s = torch.tensor(STD).view(3, 1, 1)
    return (t.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--n", type=int, default=6, help="samples per class")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model(ck.get("model", "cnn"), pretrained=False).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    cam_fn = GradCAM(model, model.cam_layer)

    ds = SyntheticFaceDataset(args.n * 2, "test")
    xs = torch.stack([ds[i][0] for i in range(args.n * 2)]).to(device)
    ys = [ds[i][1] for i in range(args.n * 2)]
    cams, probs = cam_fn(xs)

    cols = args.n * 2
    fig, axes = plt.subplots(2, cols, figsize=(1.8 * cols, 4.3))
    for i in range(cols):
        img = denorm(xs[i])
        axes[0, i].imshow(img); axes[0, i].axis("off")
        axes[0, i].set_title(f"true={'fake' if ys[i] else 'real'}", fontsize=8)
        axes[1, i].imshow(img); axes[1, i].imshow(cams[i], cmap="jet", alpha=0.45)
        axes[1, i].axis("off")
        axes[1, i].set_title(f"P(fake)={probs[i,1]:.2f}", fontsize=8)
    fig.suptitle("Grad-CAM on the last convolutional block (top: input, bottom: evidence)")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/gradcam.png", dpi=140)
    print("wrote figures/gradcam.png")


if __name__ == "__main__":
    main()
