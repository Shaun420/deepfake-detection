"""Fast, device-aware training loop for the deepfake-detection mini-project.

Examples
--------
    python src/train.py --epochs 12 --model cnn                 # auto-select CPU/GPU
    python src/train.py --device cuda --batch-size 128 --model cnn
    python src/train.py --device cpu                  # auto-tune CPU workers/threads
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
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


def available_cpu_count() -> int:
    """Return CPUs available to this process, including container CPU quotas."""
    counts = []
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        count = process_cpu_count()
        if count:
            counts.append(count)
    try:
        counts.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass

    # Python 3.11 does not expose process_cpu_count(); account for common Linux
    # cgroup v2/v1 quotas so container limits do not trigger thread oversubscription.
    for quota_path in (Path("/sys/fs/cgroup/cpu.max"),
                       Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")):
        try:
            fields = quota_path.read_text().split()
            quota = int(fields[0])
            period = int(fields[1]) if len(fields) > 1 else int(
                Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0 and period > 0:
                counts.append(max(1, (quota + period - 1) // period))
                break
        except (OSError, ValueError, IndexError):
            continue

    if not counts:
        counts.append(os.cpu_count() or 1)
    return max(1, min(counts))


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested, but CUDA is not available")
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise SystemExit("--device mps was requested, but MPS is not available")
    return device


def configure_runtime(requested_workers: int, requested_threads: int,
                      device: torch.device) -> tuple[int, int, int]:
    """Choose conservative auto defaults and avoid CPU thread oversubscription."""
    cpu_count = available_cpu_count()
    if requested_workers == -1:
        worker_cap = 8 if device.type == "cuda" else 4
        num_workers = min(worker_cap, max(0, cpu_count - 1), max(1, cpu_count // 2))
    else:
        num_workers = requested_workers

    if requested_threads == 0:
        thread_budget = max(1, cpu_count - num_workers)
        # CPU-bound training benefits from a few BLAS/oneDNN threads. On an
        # accelerator, keep the host pool smaller so it can feed the device.
        cpu_threads = min(8 if device.type == "cpu" else 4, thread_budget)
    else:
        cpu_threads = requested_threads

    torch.set_num_threads(cpu_threads)
    try:
        # There is little inter-op parallelism in this training loop; one thread
        # reduces scheduler overhead and prevents nested thread-pool contention.
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Can already be fixed by an embedding application; keep its setting.
        pass
    return cpu_count, num_workers, cpu_threads


def configure_backend(device: torch.device) -> None:
    if device.type == "cuda":
        # All batches have the same spatial dimensions. Let cuDNN select a fast
        # convolution algorithm and enable TF32 on supported NVIDIA GPUs.
        torch.backends.cudnn.benchmark = True
        if getattr(torch.version, "cuda", None) is not None:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True


def autocast_context(device: torch.device, enabled: bool, dtype):
    if enabled:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def move_batch(x, y, device: torch.device, channels_last: bool):
    non_blocking = device.type == "cuda"
    if channels_last:
        x = x.to(device=device, non_blocking=non_blocking,
                 memory_format=torch.channels_last)
    else:
        x = x.to(device=device, non_blocking=non_blocking)
    y = y.to(device=device, non_blocking=non_blocking)
    return x, y


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp_enabled=False,
             amp_dtype=None, channels_last=False):
    """Evaluate without synchronizing the accelerator once per batch."""
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    correct_sum = torch.zeros((), device=device, dtype=torch.int64)
    n = 0

    for x, y in loader:
        x, y = move_batch(x, y, device, channels_last)
        with autocast_context(device, amp_enabled, amp_dtype):
            out = model(x)
            loss = criterion(out, y)
        batch_size = y.size(0)
        loss_sum.add_(loss.detach().float() * batch_size)
        correct_sum.add_((out.argmax(1) == y).sum())
        n += batch_size

    if n == 0:
        raise ValueError("validation dataset is empty")
    loss_value, accuracy = torch.stack((loss_sum / n,
                                        correct_sum.float() / n)).cpu().tolist()
    return loss_value, accuracy


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    epoch: int, epochs: int, amp_enabled: bool,
                    amp_dtype, channels_last: bool, grad_clip: float,
                    log_interval: int = 50):
    model.train()
    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    correct_sum = torch.zeros((), device=device, dtype=torch.int64)
    n = 0
    total_batches = len(loader)
    bar = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False,
               dynamic_ncols=True)

    for step, (x, y) in enumerate(bar, start=1):
        x, y = move_batch(x, y, device, channels_last)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled, amp_dtype):
            out = model(x)
            loss = criterion(out, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip,
                                               foreach=device.type != "mps")
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip,
                                               foreach=device.type != "mps")
            optimizer.step()

        batch_size = y.size(0)
        # Keep metrics on-device: calling .item() for every batch forces CUDA to
        # synchronize and can erase much of the benefit of GPU acceleration.
        loss_sum.add_(loss.detach().float() * batch_size)
        correct_sum.add_((out.detach().argmax(1) == y).sum())
        n += batch_size

        if step % log_interval == 0 or step == total_batches:
            running_loss, running_acc = torch.stack((loss_sum / n,
                                                      correct_sum.float() / n)).cpu().tolist()
            bar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.3f}")

    if n == 0:
        raise ValueError("training dataset is empty")
    train_loss, train_acc = torch.stack((loss_sum / n,
                                         correct_sum.float() / n)).cpu().tolist()
    return train_loss, train_acc


def make_optimizer(model, args, device: torch.device):
    kwargs = dict(lr=args.lr, weight_decay=args.weight_decay)
    if device.type == "cuda" and getattr(torch.version, "cuda", None) is not None:
        # Fused AdamW combines optimizer work into fewer CUDA kernels. Fall back
        # cleanly on PyTorch builds that do not provide the fused implementation.
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs), True
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None,
                    help="root of a real real/fake dataset; omit to use the synthetic surrogate")
    ap.add_argument("--model", default="cnn", choices=["cnn", "resnet18"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                    help="auto uses CUDA, then Apple MPS, then CPU")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=64,
                    help="network input resolution (use 128+ for real photographic corpora)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=1200)
    ap.add_argument("--num-workers", type=int, default=-1,
                    help="DataLoader processes; -1 chooses a CPU-aware default, 0 disables workers")
    ap.add_argument("--cpu-threads", type=int, default=0,
                    help="PyTorch CPU threads; 0 chooses a CPU-aware default")
    ap.add_argument("--amp", choices=["auto", "on", "off"], default="auto",
                    help="mixed precision (auto enables it on CUDA GPUs)")
    ap.add_argument("--cache", action="store_true",
                    help="decode real-corpus images into RAM before training")
    ap.add_argument("--class-weights", action="store_true",
                    help="weight cross-entropy by inverse training-class frequency")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-rescale-aug", action="store_true",
                    help="disable the resampling-robustness augmentation (ablation)")
    ap.add_argument("--grad-clip", type=float, default=5.0,
                    help="maximum gradient norm; set 0 to disable clipping")
    ap.add_argument("--name", default="best", help="checkpoint basename")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        ap.error("--epochs and --batch-size must be positive")
    if args.num_workers < -1 or args.cpu_threads < 0 or args.grad_clip < 0:
        ap.error("--num-workers must be -1 or >= 0; --cpu-threads and --grad-clip must be >= 0")

    run_started = time.perf_counter()
    device = select_device(args.device)
    set_seed(args.seed)
    configure_backend(device)
    cpu_count, num_workers, cpu_threads = configure_runtime(
        args.num_workers, args.cpu_threads, device)
    if args.num_workers >= 0 and num_workers > cpu_count:
        print(f"Warning: --num-workers={num_workers} exceeds the {cpu_count} CPUs "
              "available to this process; try omitting it to use the auto setting.")

    amp_enabled = device.type == "cuda" and args.amp != "off"
    if args.amp == "on" and device.type != "cuda":
        print("AMP requested, but enabled mixed precision is currently CUDA-only; continuing without AMP.")
    amp_dtype = None
    scaler = None
    if amp_enabled:
        amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                     else torch.float16)
        if amp_dtype == torch.float16:
            scaler = torch.cuda.amp.GradScaler()

    channels_last = device.type == "cuda"
    pin_memory = device.type == "cuda"
    batch_size = args.batch_size
    print(f"Preparing data on {device} (workers={num_workers})...", flush=True)
    train_dl, val_dl, _, source = get_dataloaders(
        args.data_root, batch_size, args.n_train, args.n_val, args.n_test,
        size=args.img_size, num_workers=num_workers,
        rescale_aug=not args.no_rescale_aug, cache=args.cache,
        include_test=False, pin_memory=pin_memory)
    data_ready = time.perf_counter()

    base_model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    if channels_last:
        base_model = base_model.to(memory_format=torch.channels_last)
    class_weights = None
    if args.class_weights:
        counts = getattr(train_dl.dataset, "counts", {})
        total = sum(counts.values())
        if not counts or total == 0 or any(counts.get(c, 0) == 0 for c in ("real", "fake")):
            raise ValueError("--class-weights requires non-empty real and fake classes")
        class_weights = torch.tensor(
            [total / (len(counts) * counts[c]) for c in ("real", "fake")],
            dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer, fused_optimizer = make_optimizer(base_model, args, device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    extras = [f"workers={num_workers}", f"cpu_threads={cpu_threads}"]
    if amp_enabled:
        extras.append(f"amp={str(amp_dtype).replace('torch.', '')}")
    if channels_last:
        extras.append("channels_last")
    if fused_optimizer:
        extras.append("fused_adamw")
    if args.cache and source.startswith("FaceFolderDataset"):
        extras.append("cache=on")
    elif args.cache:
        extras.append("cache=not-needed(synthetic)")
    if class_weights is not None:
        extras.append(f"class_weights={class_weights.detach().cpu().numpy().round(3).tolist()}")
    print(f"device={device}  data={source}  model={args.model} "
          f"({count_parameters(base_model):,} params)  " + " ".join(extras))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_args = vars(args).copy()
    run_args.update({
        "resolved_device": str(device),
        "available_cpu_count": cpu_count,
        "resolved_num_workers": num_workers,
        "resolved_cpu_threads": cpu_threads,
        "amp_enabled": amp_enabled,
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else None,
        "channels_last": channels_last,
        "fused_optimizer": fused_optimizer,
    })

    history = []
    best_acc = float("-inf")
    train_started = time.perf_counter()
    setup_seconds = train_started - run_started

    for epoch in range(1, args.epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            base_model, train_dl, criterion, optimizer, scaler, device,
            epoch, args.epochs, amp_enabled, amp_dtype, channels_last,
            args.grad_clip)
        val_loss, val_acc = evaluate(
            base_model, val_dl, criterion, device, amp_enabled, amp_dtype,
            channels_last)
        history.append(dict(epoch=epoch, train_loss=train_loss,
                            train_acc=train_acc, val_loss=val_loss,
                            val_acc=val_acc, lr=epoch_lr))
        print(f"epoch {epoch:2d}/{args.epochs}  train loss {train_loss:.4f} acc {train_acc:.4f}"
              f"  |  val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"model": args.model, "state_dict": base_model.state_dict(),
                        "val_acc": val_acc, "epoch": epoch,
                        "data_source": source, "args": run_args},
                       out_dir / f"{args.name}.pt")
        scheduler.step()

    train_seconds = time.perf_counter() - train_started
    total_seconds = time.perf_counter() - run_started
    summary = {
        "history": history,
        "best_val_acc": best_acc,
        "minutes": train_seconds / 60,
        "data_setup_seconds": data_ready - run_started,
        "setup_seconds": setup_seconds,
        "total_minutes": total_seconds / 60,
        "source": source,
        "params": count_parameters(base_model),
        "args": run_args,
    }
    with open(out_dir / f"history_{args.name}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nbest val acc {best_acc:.4f} -> {out_dir / (args.name + '.pt')} "
          f"(training {train_seconds / 60:.1f} min; setup "
          f"{setup_seconds:.1f} s)")


if __name__ == "__main__":
    main()
