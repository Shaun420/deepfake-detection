"""
Dataset utilities for the deepfake-detection mini-project.

Two data sources are supported:

1. `FaceFolderDataset` -- a real dataset laid out as

       data/
         train/real/*.jpg
         train/fake/*.jpg
         val/real/...   val/fake/...
         test/real/...  test/fake/...

   This is the layout produced by the usual FaceForensics++ / Celeb-DF /
   "140k Real and Fake Faces" (Kaggle) extraction scripts.

2. `SyntheticFaceDataset` -- a self-contained, procedurally generated
   surrogate dataset used so that the whole pipeline (training, evaluation,
   Grad-CAM, inference) is reproducible on any machine without downloading
   several gigabytes of video data.

   The *pristine* class is a simple parametric face rendering (skin tone
   gradient + facial features + sensor noise).  The *manipulated* class is
   produced by applying the artefacts that real face-swap pipelines are known
   to leave behind:

     * a resampled (down/up-sampled) inner-face region  -> resolution
       mismatch between the swapped face and the rest of the image,
     * a soft blending boundary around the face mask    -> blending artefact,
     * a colour / contrast mismatch of the swapped area -> colour transfer error,
     * a weak checkerboard pattern                      -> transposed-conv
                                                            (GAN upsampling) artefact.

   These are exactly the cues the literature reports CNN detectors latch onto
   (Li & Lyu 2018, "Exposing DeepFake Videos By Detecting Face Warping
   Artifacts"; Rossler et al. 2019, FaceForensics++).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_SIZE = 64
CLASSES = ("real", "fake")  # label 0 = real/pristine, 1 = fake/manipulated

# Real-world corpora name their class folders inconsistently.  These synonyms let
# FaceFolderDataset read e.g. the Kaggle "Deepfake image detection" set unchanged.
CLASS_SYNONYMS = {
    0: ("real", "reals", "original", "originals", "authentic", "pristine",
        "training_real", "test_real", "real_images", "0"),
    1: ("fake", "fakes", "deepfake", "deepfakes", "manipulated", "forged",
        "synthetic", "training_fake", "test_fake", "fake_images", "1"),
}


# --------------------------------------------------------------------------- #
# 1. Procedural face renderer
# --------------------------------------------------------------------------- #
def _render_face(rng: np.random.Generator, size: int = IMG_SIZE) -> np.ndarray:
    """Return an HxWx3 float image in [0, 1] that loosely looks like a face."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2 + rng.normal(0, 1.5), size / 2 + rng.normal(0, 1.5)

    # background: smooth low-frequency gradient
    bg = rng.uniform(0.15, 0.55, size=3)
    img = np.ones((size, size, 3), np.float32) * bg
    img += 0.06 * (xx / size)[..., None] * rng.normal(0, 1, 3)

    # head: ellipse with a skin tone and vertical lighting gradient
    rx, ry = size * rng.uniform(0.26, 0.33), size * rng.uniform(0.34, 0.42)
    ell = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    face_mask = np.clip(1.5 * (1.0 - ell), 0, 1)

    skin = np.array([rng.uniform(0.55, 0.92), rng.uniform(0.40, 0.72),
                     rng.uniform(0.32, 0.62)], np.float32)
    shading = 1.0 - 0.35 * ((yy - cy) / size + 0.5)
    face = skin[None, None, :] * shading[..., None]
    img = img * (1 - face_mask[..., None]) + face * face_mask[..., None]

    # hair cap
    hair = np.clip(1.5 * (1.0 - (((xx - cx) / (rx * 1.12)) ** 2 +
                                 ((yy - (cy - ry * 0.55)) / (ry * 0.55)) ** 2)), 0, 1)
    hair_col = np.array([rng.uniform(0.05, 0.35)] * 3, np.float32) * rng.uniform(0.6, 1.4, 3)
    img = img * (1 - hair[..., None]) + hair_col[None, None, :] * hair[..., None]

    def blob(bx, by, brx, bry, colour, strength=1.0):
        m = np.clip(2.0 * (1.0 - (((xx - bx) / brx) ** 2 + ((yy - by) / bry) ** 2)), 0, 1)
        m = m * strength
        return m[..., None] * np.asarray(colour, np.float32)[None, None, :]

    eye_dx = rx * rng.uniform(0.42, 0.55)
    eye_y = cy - ry * rng.uniform(0.12, 0.24)
    eye_r = size * rng.uniform(0.045, 0.065)
    for sgn in (-1, 1):
        m = blob(cx + sgn * eye_dx, eye_y, eye_r * 1.6, eye_r, (1, 1, 1), 0.9)
        img = img * (1 - m.max(-1, keepdims=True)) + m
        p = blob(cx + sgn * eye_dx, eye_y, eye_r * 0.5, eye_r * 0.55,
                 (rng.uniform(0, .3), rng.uniform(0, .4), rng.uniform(.1, .5)), 1.0)
        img = img * (1 - p.max(-1, keepdims=True)) + p

    # nose + mouth
    n = blob(cx, cy + ry * 0.08, size * 0.035, size * 0.075, skin * 0.78, 0.8)
    img = img * (1 - n.max(-1, keepdims=True) * 0.8) + n * 0.8
    m = blob(cx, cy + ry * rng.uniform(0.38, 0.5), size * rng.uniform(0.09, 0.13),
             size * rng.uniform(0.025, 0.045),
             (rng.uniform(.5, .8), rng.uniform(.2, .4), rng.uniform(.2, .4)), 0.95)
    img = img * (1 - m.max(-1, keepdims=True)) + m

    # sensor noise + global exposure jitter
    img = img * rng.uniform(0.9, 1.1) + rng.normal(0, 0.015, img.shape)
    return np.clip(img, 0, 1).astype(np.float32), (cx, cy, rx, ry)


# --------------------------------------------------------------------------- #
# 2. Manipulation ("deepfake") pipeline
# --------------------------------------------------------------------------- #
def _resample(a: np.ndarray, factor: float) -> np.ndarray:
    """Down-sample then up-sample -> the resolution loss of a swapped face."""
    h, w = a.shape[:2]
    small = Image.fromarray((a * 255).astype(np.uint8)).resize(
        (max(2, int(w * factor)), max(2, int(h * factor))), Image.BILINEAR)
    return np.asarray(small.resize((w, h), Image.BILINEAR), np.float32) / 255.0


def _manipulate(img: np.ndarray, geom, rng: np.random.Generator,
                strength: float = 1.0) -> np.ndarray:
    """Apply face-swap style artefacts inside the face region.

    `strength` scales every artefact; values well below 1 make the detection
    problem genuinely hard (which is what we want for a meaningful report).
    """
    size = img.shape[0]
    cx, cy, rx, ry = geom
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    # soft inner-face mask (the "donor" region of a face swap)
    d = ((xx - cx) / (rx * rng.uniform(0.80, 1.00))) ** 2 + \
        ((yy - cy) / (ry * rng.uniform(0.80, 1.00))) ** 2
    mask = np.clip((1.0 - d) * rng.uniform(2.0, 5.0), 0, 1)       # soft boundary
    mask = mask[..., None]

    swapped = _resample(img, rng.uniform(0.45, 0.80))             # resolution mismatch
    swapped = swapped * (1 + (rng.uniform(0.97, 1.03, 3) - 1) * strength)  # colour shift
    swapped = np.clip((swapped - 0.5) * (1 + (rng.uniform(0.96, 1.05) - 1) * strength)
                      + 0.5, 0, 1)

    # GAN upsampling / checkerboard residual
    cb = ((np.indices((size, size)).sum(0) % 2) * 2 - 1).astype(np.float32)
    swapped = swapped + rng.uniform(0.001, 0.004) * strength * cb[..., None]

    alpha = strength * rng.uniform(0.55, 1.0)      # partial blend -> weak manipulation
    out = img * (1 - mask * alpha) + swapped * mask * alpha
    return np.clip(out, 0, 1).astype(np.float32)


def _post_process(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Capture/transmission pipeline applied to BOTH classes.

    Without this the fake class is trivially separable by a blur detector; JPEG
    recompression and resizing also destroy part of the forensic evidence,
    which is exactly what happens to real-world social-media video.
    """
    import io
    if rng.random() < 0.5:                                  # mild global blur
        img = img * 0.0 + _resample(img, rng.uniform(0.7, 1.0))
    img = np.clip(img + rng.normal(0, rng.uniform(0.004, 0.02), img.shape), 0, 1)
    buf = io.BytesIO()                                       # JPEG recompression
    Image.fromarray((img * 255).astype(np.uint8)).save(
        buf, format="JPEG", quality=int(rng.integers(55, 96)))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), np.float32) / 255.0


def make_sample(seed: int, label: int, size: int = IMG_SIZE,
                strength: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img, geom = _render_face(rng, size)
    if label == 1:
        img = _manipulate(img, geom, rng, strength)
    return _post_process(img, rng)


# --------------------------------------------------------------------------- #
# 3. Datasets
# --------------------------------------------------------------------------- #
MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


class RandomRescale:
    """Randomly down- then up-sample the image (a resampling-robustness augmentation).

    A video frame is almost never delivered at the training resolution: it is
    decoded at e.g. 1080p, the face crop is a few hundred pixels, and it is then
    resized down to the network input.  That resampling leaves traces that look
    like the forgery cue, so without this augmentation the detector fires on
    *any* rescaled real frame.  Exposing it to random rescaling at training time
    removes that shortcut.
    """

    def __init__(self, p: float = 0.5, lo: float = 0.4, hi: float = 1.0):
        self.p, self.lo, self.hi = p, lo, hi

    def __call__(self, img: Image.Image) -> Image.Image:
        import random
        if random.random() > self.p:
            return img
        w, h = img.size
        f = random.uniform(self.lo, self.hi)
        method = random.choice([Image.BILINEAR, Image.BICUBIC, Image.NEAREST])
        small = img.resize((max(4, int(w * f)), max(4, int(h * f))), method)
        return small.resize((w, h), random.choice([Image.BILINEAR, Image.BICUBIC]))


def build_transforms(train: bool, size: int = IMG_SIZE, rescale_aug: bool = True):
    aug = [transforms.Resize((size, size))]
    if train:
        aug += [transforms.RandomHorizontalFlip()]
        if rescale_aug:
            aug += [RandomRescale(p=0.5, lo=0.4, hi=1.0)]
        aug += [transforms.ColorJitter(0.1, 0.1, 0.1)]
    aug += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(aug)


class SyntheticFaceDataset(Dataset):
    """Deterministic, on-the-fly synthetic real/fake face dataset."""

    def __init__(self, n: int, split: str = "train", size: int = IMG_SIZE,
                 transform: Optional[Callable] = None, strength: float = 0.6,
                 rescale_aug: bool = True):
        assert split in ("train", "val", "test")
        self.n = n
        self.size = size
        self.offset = {"train": 0, "val": 1_000_000, "test": 2_000_000}[split]
        self.transform = transform if transform is not None else build_transforms(
            split == "train", size, rescale_aug)
        # rendering is the bottleneck, so materialise the split once (uint8, cheap:
        # 4000 x 64 x 64 x 3 = 49 MB) and reuse it across epochs
        self.labels = [i % 2 for i in range(n)]          # perfectly balanced
        self.counts = {c: self.labels.count(i) for i, c in enumerate(CLASSES)}
        self.cache = True  # synthetic samples are always materialised in RAM
        self.images = np.stack([
            (make_sample(self.offset + i, self.labels[i], size, strength) * 255
             ).astype(np.uint8) for i in range(n)]) if n else np.zeros((0, size, size, 3), np.uint8)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        return self.transform(Image.fromarray(self.images[i])), self.labels[i]


class FaceFolderDataset(Dataset):
    """real/fake image-folder dataset (used when a real corpus is available).

    Class folders are matched case-insensitively against CLASS_SYNONYMS and may
    sit at any depth below `root`, so most public deepfake corpora load as-is.
    """

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str, size: int = IMG_SIZE,
                 transform: Optional[Callable] = None, train: bool = False,
                 rescale_aug: bool = True, cache: bool = False):
        root = Path(root)
        lookup = {name: label for label, names in CLASS_SYNONYMS.items() for name in names}
        self.cache = cache

        self.items: list[tuple[Path, int]] = []
        seen_dirs: set[Path] = set()
        # any directory whose name matches a class synonym contributes its images
        for d in sorted([root] + [p for p in root.rglob("*") if p.is_dir()]):
            label = lookup.get(d.name.strip().lower())
            if label is None or d in seen_dirs:
                continue
            seen_dirs.add(d)
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in self.EXTS:
                    self.items.append((p, label))

        if not self.items:
            raise FileNotFoundError(
                f"no class folders found under {root}. Expected sub-directories named "
                f"one of {sorted(lookup)} containing images.")

        counts = {c: sum(1 for _, l in self.items if l == i)
                  for i, c in enumerate(CLASSES)}
        self.counts = counts
        self.transform = transform if transform is not None else build_transforms(
            train, size, rescale_aug)

        # Keeping decoded RGB arrays in RAM avoids reopening and decoding every
        # image on every epoch.  The cache deliberately stores the *raw* image;
        # random training transforms must still run on every access.
        self._cache: Optional[list[np.ndarray]] = None
        if self.cache:
            self._cache = []
            for p, _ in self.items:
                with Image.open(p) as image:
                    self._cache.append(np.asarray(image.convert("RGB"), dtype=np.uint8).copy())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, label = self.items[i]
        if self._cache is None:
            with Image.open(p) as source:
                image = source.convert("RGB")
        else:
            image = Image.fromarray(self._cache[i], mode="RGB")
        return self.transform(image), label


def get_dataloaders(data_root: Optional[str], batch_size: int = 64,
                    n_train: int = 4000, n_val: int = 800, n_test: int = 1200,
                    size: int = IMG_SIZE, num_workers: int = 0, strength: float = 0.6,
                    rescale_aug: bool = True, cache: bool = False,
                    include_test: bool = True, pin_memory: Optional[bool] = None):
    """Return train/validation/test loaders for a real corpus or the synthetic set.

    The test loader is ``None`` when ``include_test=False``.

    ``cache`` only changes the real-image loader: the synthetic dataset is already
    generated and stored in RAM.  Training can set ``include_test=False`` to avoid
    decoding/materialising a test split that it never uses.  Worker-related options
    are kept here so training, evaluation and explainability share the same loaders.
    """
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    split_names = ("train", "val", "test") if include_test else ("train", "val")
    if data_root:
        root = Path(data_root)
        if not root.is_dir():
            raise FileNotFoundError(
                f"data root '{root}' does not exist. To use the synthetic surrogate, "
                "omit --data-root; for a real corpus, provide a root containing "
                "train/ and val/ class folders (and test/ for evaluation).")
        if not (root / "train").is_dir():
            raise FileNotFoundError(
                f"data root '{root}' has no train/ split. Expected train/{CLASSES[0]} and "
                f"train/{CLASSES[1]} (plus val/ and, when requested, test/). "
                "To use the synthetic surrogate, omit --data-root.")
        sets = [FaceFolderDataset(str(root / split), size,
                                  train=(split == "train"),
                                  rescale_aug=rescale_aug, cache=cache)
                for split in split_names]
        counts = " ".join(f"{split}={dataset.counts}"
                          for split, dataset in zip(split_names, sets))
        source = f"FaceFolderDataset({root}) {counts}"
    else:
        sizes = {"train": n_train, "val": n_val, "test": n_test}
        sets = [SyntheticFaceDataset(sizes[split], split, size, strength=strength,
                                     rescale_aug=rescale_aug)
                for split in split_names]
        source = "SyntheticFaceDataset (procedural surrogate)"

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if num_workers > 0:
        # Persistent workers avoid worker startup and cache rebuilding at every epoch.
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loaders = [DataLoader(dataset, shuffle=(i == 0), **loader_kwargs)
               for i, dataset in enumerate(sets)]
    if not include_test:
        loaders.append(None)
    return loaders[0], loaders[1], loaders[2], source


if __name__ == "__main__":  # quick visual sanity check -> figures/dataset_samples.png
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 6, figsize=(12, 4.2))
    for j in range(6):
        for label in (0, 1):
            axes[label, j].imshow(make_sample(100 + j, label))
            axes[label, j].axis("off")
        axes[0, j].set_title("real", fontsize=9)
        axes[1, j].set_title("fake", fontsize=9)
    fig.suptitle("Synthetic surrogate dataset: pristine (top) vs. manipulated (bottom)")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/dataset_samples.png", dpi=140)
    print("wrote figures/dataset_samples.png")
