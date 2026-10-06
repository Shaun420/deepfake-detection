"""Test-set evaluation: metrics, confusion matrix, ROC/PR curves, training curves.

    python src/evaluate.py --ckpt checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from data import get_dataloaders
from model import build_model

FIG = Path("figures")


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        p = F.softmax(model(x.to(device)), dim=1)[:, 1]
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def plot_history(path="checkpoints/history.json"):
    if not Path(path).exists():
        return
    h = json.load(open(path))["history"]
    ep = [e["epoch"] for e in h]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, [e["train_loss"] for e in h], "-o", ms=3, label="train")
    ax[0].plot(ep, [e["val_loss"] for e in h], "-o", ms=3, label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, [e["train_acc"] for e in h], "-o", ms=3, label="train")
    ax[1].plot(ep, [e["val_acc"] for e in h], "-o", ms=3, label="val")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].set_title("Accuracy"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "training_curves.png", dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-test", type=int, default=1200)
    args = ap.parse_args()

    FIG.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model(ck.get("model", "cnn"), pretrained=False).to(device)
    model.load_state_dict(ck["state_dict"])

    *_, test_dl, source = get_dataloaders(args.data_root, args.batch_size,
                                          n_test=args.n_test)
    probs, y = collect(model, test_dl, device)
    pred = (probs >= 0.5).astype(int)

    metrics = {
        "data_source": source,
        "n_test": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_fake": float(precision_score(y, pred, zero_division=0)),
        "recall_fake": float(recall_score(y, pred, zero_division=0)),
        "f1_fake": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)),
    }
    fpr, tpr, thr = roc_curve(y, probs)
    eer_i = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    metrics["eer"] = float((fpr[eer_i] + (1 - tpr[eer_i])) / 2)
    cm = confusion_matrix(y, pred)
    metrics["confusion_matrix"] = cm.tolist()

    print(json.dumps(metrics, indent=2))
    print(classification_report(y, pred, target_names=["real", "fake"], digits=4))
    json.dump(metrics, open("checkpoints/test_metrics.json", "w"), indent=2)

    # confusion matrix
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["real", "fake"]); ax.set_yticks([0, 1], ["real", "fake"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (acc={metrics['accuracy']:.3f})")
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrix.png", dpi=140); plt.close(fig)

    # ROC + PR
    prec, rec, _ = precision_recall_curve(y, probs)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    ax[0].set_title("ROC curve"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(rec, prec, label=f"AP area = {auc(rec, prec):.4f}")
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
    ax[1].set_title("Precision-Recall (fake = positive)"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "roc_pr_curves.png", dpi=140); plt.close(fig)

    # score histogram
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.hist(probs[y == 0], bins=40, alpha=.65, label="real")
    ax.hist(probs[y == 1], bins=40, alpha=.65, label="fake")
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("P(fake)"); ax.set_ylabel("count"); ax.set_title("Score distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "score_hist.png", dpi=140); plt.close(fig)

    plot_history()
    print(f"figures written to {FIG}/")


if __name__ == "__main__":
    main()
