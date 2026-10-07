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

from data import (MEAN, STD, FaceFolderDataset, SyntheticFaceDataset,
                  build_transforms)
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
    ap.add_argument("--data-root", default=None,
                    help="real corpus root; omit this for a synthetic checkpoint")
    ap.add_argument("--img-size", type=int, default=None,
                    help="override input resolution (defaults to checkpoint args)")
    ap.add_argument("--n", type=int, default=6, help="samples per class")
    args = ap.parse_args()
    if args.n < 1:
        ap.error("--n must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    ck_args = ck.get("args", {})
    checkpoint_source = ck.get("data_source") or ""
    saved_data_root = ck_args.get("data_root")

    if args.data_root is None:
        if checkpoint_source.startswith("SyntheticFaceDataset"):
            pass  # Preserve the checkpoint's synthetic data provenance.
        elif checkpoint_source.startswith("FaceFolderDataset"):
            if saved_data_root and Path(saved_data_root).is_dir():
                args.data_root = saved_data_root
            else:
                raise FileNotFoundError(
                    "This checkpoint was trained on a real image corpus, but its recorded "
                    f"data root is unavailable ({saved_data_root!r}). Pass --data-root to a "
                    "valid corpus; synthetic Grad-CAM would not explain this checkpoint.")
        elif saved_data_root:
            if Path(saved_data_root).is_dir():
                # Older checkpoints did not record their actual data source.
                args.data_root = saved_data_root
            else:
                print("Warning: this older checkpoint records a data root that is missing "
                      "and has no source metadata; using the synthetic surrogate. "
                      "Pass --data-root to analyze a real corpus.")

    model = build_model(ck.get("model", "cnn"), pretrained=False).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    cam_fn = GradCAM(model, model.cam_layer)
    size = args.img_size or ck.get("args", {}).get("img_size", 64)

    if args.data_root:
        root = Path(args.data_root)
        if not root.is_dir():
            raise FileNotFoundError(
                f"--data-root '{root}' does not exist. Prepare the real corpus first. "
                "If this checkpoint was trained on the synthetic surrogate, omit --data-root.")
        if (root / "train").is_dir():
            if not (root / "test").is_dir():
                raise FileNotFoundError(
                    f"real corpus '{root}' has no test/ split for Grad-CAM. "
                    "Expected data/test/{real,fake}/... .")
            test_root = root / "test"
        else:
            test_root = root
        try:
            ds = FaceFolderDataset(
                str(test_root), size=size,
                transform=build_transforms(False, size), train=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"No real/fake class folders were found in Grad-CAM split '{test_root}'. "
                "For a synthetic checkpoint, omit --data-root.") from exc
        by_label = {0: [], 1: []}
        for index, (_, label) in enumerate(ds.items):
            by_label[label].append(index)
        if any(len(by_label[label]) < args.n for label in (0, 1)):
            raise ValueError(f"need at least {args.n} images per class in {test_root}")
        indices = by_label[0][:args.n] + by_label[1][:args.n]
        source_name = f"real corpus ({test_root})"
    else:
        ds = SyntheticFaceDataset(
            args.n * 2, "test", size=size,
            transform=build_transforms(False, size))
        indices = list(range(args.n * 2))
        source_name = "synthetic surrogate"

    samples = [ds[index] for index in indices]
    xs = torch.stack([sample[0] for sample in samples]).to(device)
    ys = [sample[1] for sample in samples]
    cams, probs = cam_fn(xs)

    cols = args.n * 2
    fig, axes = plt.subplots(2, cols, figsize=(1.8 * cols, 4.3), squeeze=False)
    for i in range(cols):
        img = denorm(xs[i])
        axes[0, i].imshow(img); axes[0, i].axis("off")
        axes[0, i].set_title(f"true={'fake' if ys[i] else 'real'}", fontsize=8)
        axes[1, i].imshow(img); axes[1, i].imshow(cams[i], cmap="jet", alpha=0.45)
        axes[1, i].axis("off")
        axes[1, i].set_title(f"P(fake)={probs[i, 1]:.2f}", fontsize=8)
    fig.suptitle(f"Grad-CAM on the last convolutional block — {source_name}")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/gradcam.png", dpi=140)
    plt.close(fig)
    print(f"wrote figures/gradcam.png ({source_name})")


if __name__ == "__main__":
    main()
