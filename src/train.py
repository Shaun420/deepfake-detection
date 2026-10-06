"""Training loop for the deepfake detection mini-project.

Example
-------
    python src/train.py --epochs 12 --model cnn
    python src/train.py --data-root data --model resnet18 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data import get_dataloaders
from model import build_model, count_parameters


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None,
                    help="root of a real real/fake dataset; omit to use the synthetic surrogate")
    ap.add_argument("--model", default="cnn", choices=["cnn", "resnet18"])
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl, _, source = get_dataloaders(
        args.data_root, args.batch_size, args.n_train, args.n_val, args.n_test)

    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"device={device}  data={source}  model={args.model} "
          f"({count_parameters(model):,} params)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_acc, t0 = [], 0.0, time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        bar = tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
            bar.set_postfix(loss=f"{loss_sum/n:.4f}", acc=f"{correct/n:.3f}")
        sched.step()

        tr_loss, tr_acc = loss_sum / n, correct / n
        va_loss, va_acc = evaluate(model, val_dl, criterion, device)
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, lr=sched.get_last_lr()[0]))
        print(f"epoch {epoch:2d}/{args.epochs}  train loss {tr_loss:.4f} acc {tr_acc:.4f}"
              f"  |  val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc >= best_acc:
            best_acc = va_acc
            torch.save({"model": args.model, "state_dict": model.state_dict(),
                        "val_acc": va_acc, "epoch": epoch, "args": vars(args)},
                       out_dir / "best.pt")

    json.dump({"history": history, "best_val_acc": best_acc,
               "minutes": (time.time() - t0) / 60, "source": source,
               "params": count_parameters(model), "args": vars(args)},
              open(out_dir / "history.json", "w"), indent=2)
    print(f"\nbest val acc {best_acc:.4f} -> {out_dir/'best.pt'} "
          f"({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
