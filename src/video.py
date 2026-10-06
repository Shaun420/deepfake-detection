"""Video support for the deepfake detector.

The classifier itself is image-only. For a video input we therefore:

  1. sample a small number of still frames spread evenly over the clip,
  2. run a Haar-cascade face detector on each sampled frame and crop the
     largest face (frames with no detectable face are skipped; an optional
     centre-crop fallback keeps face-less clips usable),
  3. classify every face crop independently with the CNN,
  4. aggregate the per-frame scores into one clip-level decision.

Aggregation is deliberately simple and reported in full, because a single
number hides a lot: we report the mean P(fake) over frames, the mean
confidence in the predicted class, the fraction of frames voting "fake", and
-- when the true label of the clip is supplied with --label -- the actual
per-frame accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import build_transforms

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXT


# --------------------------------------------------------------------------- #
# Face detection
# --------------------------------------------------------------------------- #
_CASCADE: Optional[cv2.CascadeClassifier] = None


def _cascade() -> cv2.CascadeClassifier:
    global _CASCADE
    if _CASCADE is None:
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        c = cv2.CascadeClassifier(xml)
        if c.empty():
            raise RuntimeError(f"could not load Haar cascade from {xml}")
        _CASCADE = c
    return _CASCADE


def detect_face(frame_bgr: np.ndarray, margin: float = 0.25
                ) -> Optional[Tuple[int, int, int, int]]:
    """Return (x, y, w, h) of the largest face in the frame, with margin, or None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                        minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])       # largest face
    H, W = frame_bgr.shape[:2]
    mx, my = int(w * margin), int(h * margin)                # context margin
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    return x0, y0, x1 - x0, y1 - y0


def _centre_crop_box(frame: np.ndarray) -> Tuple[int, int, int, int]:
    H, W = frame.shape[:2]
    s = int(min(H, W) * 0.8)
    return (W - s) // 2, (H - s) // 2, s, s


# --------------------------------------------------------------------------- #
# Frame sampling + scoring
# --------------------------------------------------------------------------- #
@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    prob_fake: float
    box: Tuple[int, int, int, int]
    face_detected: bool
    crop_rgb: np.ndarray = field(repr=False, default=None)

    @property
    def label(self) -> str:
        return "FAKE" if self.prob_fake >= 0.5 else "REAL"


@dataclass
class VideoResult:
    path: str
    n_frames_total: int
    fps: float
    duration: float
    frames: List[FrameResult]
    n_sampled: int
    n_no_face: int

    @property
    def probs(self) -> np.ndarray:
        return np.array([f.prob_fake for f in self.frames], dtype=float)

    @property
    def mean_prob_fake(self) -> float:
        return float(self.probs.mean()) if len(self.frames) else float("nan")

    @property
    def verdict(self) -> str:
        return "FAKE" if self.mean_prob_fake >= 0.5 else "REAL"

    @property
    def fake_frame_ratio(self) -> float:
        p = self.probs
        return float((p >= 0.5).mean()) if len(p) else float("nan")

    @property
    def mean_confidence(self) -> float:
        """Mean probability assigned to each frame's own predicted class."""
        p = self.probs
        return float(np.maximum(p, 1 - p).mean()) if len(p) else float("nan")

    @property
    def agreement(self) -> float:
        """Fraction of frames agreeing with the clip-level verdict."""
        p = self.probs
        if not len(p):
            return float("nan")
        return float(((p >= 0.5) == (self.verdict == "FAKE")).mean())

    def accuracy(self, true_label: str) -> float:
        """Per-frame accuracy against a known ground-truth label ('real'/'fake')."""
        p = self.probs
        if not len(p):
            return float("nan")
        truth = 1 if true_label.lower() == "fake" else 0
        return float(((p >= 0.5).astype(int) == truth).mean())


@torch.no_grad()
def analyse_video(path: str, model, device, num_frames: int = 16,
                  fallback_centre_crop: bool = True,
                  transform=None) -> VideoResult:
    transform = transform or build_transforms(train=False)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0

    if total > 0:
        idxs = np.unique(np.linspace(0, total - 1, num_frames).astype(int))
        seek = True
    else:                                   # stream without a frame count
        idxs, seek = np.arange(num_frames), False

    frames: List[FrameResult] = []
    n_sampled = n_no_face = 0
    for i in idxs:
        if seek:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        n_sampled += 1

        box = detect_face(frame)
        detected = box is not None
        if not detected:
            n_no_face += 1
            if not fallback_centre_crop:
                continue
            box = _centre_crop_box(frame)

        x, y, w, h = box
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        xt = transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
        prob = F.softmax(model(xt), 1)[0, 1].item()
        frames.append(FrameResult(int(i), float(i) / fps, prob, box, detected, crop_rgb))

    cap.release()
    return VideoResult(str(path), total, fps, (total / fps) if total else 0.0,
                       frames, n_sampled, n_no_face)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def format_report(r: VideoResult, true_label: Optional[str] = None) -> str:
    L = [f"\nVideo      : {r.path}",
         f"Duration   : {r.duration:.2f} s  ({r.n_frames_total} frames @ {r.fps:.1f} fps)",
         f"Sampled    : {r.n_sampled} frames | faces found {r.n_sampled - r.n_no_face}"
         f" | no face {r.n_no_face} | scored {len(r.frames)}",
         "",
         f"{'frame':>7} {'time(s)':>8} {'face':>5} {'P(fake)':>9}  verdict"]
    for f in r.frames:
        L.append(f"{f.frame_index:>7} {f.timestamp:>8.2f} {'yes' if f.face_detected else 'no':>5}"
                 f" {f.prob_fake:>9.4f}  {f.label}")
    if not r.frames:
        return "\n".join(L + ["", "No usable frames -- nothing to score."])

    p = r.probs
    L += ["",
          f"Mean P(fake)        : {r.mean_prob_fake:.4f}  (std {p.std():.4f}, "
          f"min {p.min():.4f}, max {p.max():.4f})",
          f"Frames voting FAKE  : {r.fake_frame_ratio*100:.1f} %",
          f"Mean confidence     : {r.mean_confidence*100:.2f} %   "
          "(mean probability of each frame's own predicted class)",
          f"Frame agreement     : {r.agreement*100:.2f} %   (frames matching the verdict)",
          f"CLIP VERDICT        : {r.verdict}  (threshold 0.5 on mean P(fake))"]
    if true_label:
        L.append(f"Mean accuracy       : {r.accuracy(true_label)*100:.2f} %   "
                 f"(per-frame, vs. ground truth '{true_label.lower()}')")
    return "\n".join(L)


def save_montage(r: VideoResult, out_path: str = "figures/video_frames.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not r.frames:
        return None
    n = len(r.frames)
    cols = min(8, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.35 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, f in enumerate(r.frames):
        ax = axes[k // cols][k % cols]
        ax.imshow(f.crop_rgb)
        ax.set_title(f"t={f.timestamp:.1f}s  {f.label}\nP(fake)={f.prob_fake:.2f}"
                     + ("" if f.face_detected else "  [no face]"),
                     fontsize=7.5,
                     color="tab:red" if f.prob_fake >= 0.5 else "tab:green")
        ax.axis("off")
    fig.suptitle(f"{Path(r.path).name} -- verdict {r.verdict} "
                 f"(mean P(fake) = {r.mean_prob_fake:.3f})", fontsize=11)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
