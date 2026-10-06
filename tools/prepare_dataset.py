"""Turn a downloaded real deepfake corpus into the layout this project expects.

Target layout:

    data/
      train/{real,fake}/*.jpg
      val/{real,fake}/*.jpg
      test/{real,fake}/*.jpg

Built for the Kaggle set used in this project --
`saurabhbagchi/deepfake-image-detection` -- whose archive expands to

    Sample_fake_images/
    train-20250112T065955Z-001/train/{real,fake}/...
    test-20250112T065939Z-001/test/{real,fake}/...

but it is deliberately generic: it searches the source tree for *any*
directory whose name matches a real/fake synonym (see src/data.py), so most
public corpora work unchanged.

Typical use
-----------
    # 1. download + unzip (needs a Kaggle account; see README)
    kaggle datasets download -d saurabhbagchi/deepfake-image-detection
    unzip -q deepfake-image-detection.zip -d raw/

    # 2. build the splits
    python tools/prepare_dataset.py --src raw --out data --val-frac 0.15

If the corpus already has its own test split, it is preserved and the
validation set is carved out of the training split only (no leakage).
Pass --copy to copy instead of symlink, and --face-crop to store detected
face crops instead of the full images.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import CLASS_SYNONYMS, CLASSES  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOOKUP = {n: l for l, names in CLASS_SYNONYMS.items() for n in names}
SPLIT_HINTS = {"train": "train", "training": "train",
               "test": "test", "testing": "test", "val": "val",
               "valid": "val", "validation": "val"}


def scan(src: Path):
    """Return {(split_hint, label): [paths]} for every class folder under src."""
    found = defaultdict(list)
    for d in sorted([src] + [p for p in src.rglob("*") if p.is_dir()]):
        label = LOOKUP.get(d.name.strip().lower())
        if label is None:
            continue
        imgs = [p for p in sorted(d.rglob("*")) if p.suffix.lower() in EXTS]
        if not imgs:
            continue
        # infer the split from any ancestor directory name
        split = None
        for part in d.parts[::-1]:
            key = part.strip().lower()
            for hint, canon in SPLIT_HINTS.items():
                if key == hint or key.startswith(hint + "-") or key.startswith(hint + "_"):
                    split = canon
                    break
            if split:
                break
        found[(split or "train", label)].extend(imgs)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing the extracted corpus")
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of the TRAIN split held out for validation")
    ap.add_argument("--test-frac", type=float, default=0.15,
                    help="only used if the corpus has no test split of its own")
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    ap.add_argument("--face-crop", action="store_true",
                    help="detect and save the largest face crop instead of the full image")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.is_dir():
        sys.exit(f"source folder not found: {src}")

    found = scan(src)
    if not found:
        sys.exit(f"no real/fake class folders found under {src}.\n"
                 f"Expected directory names among: {sorted(LOOKUP)}")

    print("discovered:")
    for (split, label), imgs in sorted(found.items()):
        print(f"  {split:5s} / {CLASSES[label]:4s}: {len(imgs)} images")

    rng = random.Random(args.seed)
    has_test = any(s == "test" for s, _ in found)

    # assemble final split -> label -> paths
    final = defaultdict(lambda: defaultdict(list))
    for (split, label), imgs in found.items():
        imgs = sorted(set(imgs))
        rng.shuffle(imgs)
        if split == "train":
            if has_test:
                k = int(len(imgs) * args.val_frac)
                final["val"][label] += imgs[:k]
                final["train"][label] += imgs[k:]
            else:
                nt = int(len(imgs) * args.test_frac)
                nv = int(len(imgs) * args.val_frac)
                final["test"][label] += imgs[:nt]
                final["val"][label] += imgs[nt:nt + nv]
                final["train"][label] += imgs[nt + nv:]
        else:
            final[split][label] += imgs

    cropper = None
    if args.face_crop:
        import cv2
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from video import detect_face
        cropper = (cv2, detect_face)

    manifest, n_written, n_nocrop = {}, 0, 0
    for split in ("train", "val", "test"):
        manifest[split] = {}
        for label, cls in enumerate(CLASSES):
            dst_dir = out / split / cls
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)
            paths = final[split][label]
            for i, p in enumerate(paths):
                dst = dst_dir / f"{i:06d}_{p.stem[:40]}{p.suffix.lower()}"
                if cropper:
                    cv2, detect_face = cropper
                    im = cv2.imread(str(p))
                    if im is None:
                        continue
                    box = detect_face(im)
                    if box is None:
                        n_nocrop += 1
                    else:
                        x, y, w, h = box
                        im = im[y:y + h, x:x + w]
                    cv2.imwrite(str(dst.with_suffix(".jpg")), im)
                elif args.copy:
                    shutil.copy2(p, dst)
                else:
                    try:
                        dst.symlink_to(p.resolve())
                    except OSError:
                        shutil.copy2(p, dst)
                n_written += 1
            manifest[split][cls] = len(paths)

    out.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)

    print(f"\nwrote {n_written} images to {out}/")
    if args.face_crop and n_nocrop:
        print(f"  ({n_nocrop} images had no detectable face and were stored uncropped)")
    for split in ("train", "val", "test"):
        tot = sum(manifest[split].values())
        print(f"  {split:5s}: {manifest[split]} (total {tot})")
    print(f"\nmanifest -> {out/'manifest.json'}")
    print(f"\nnow run:\n  python src/train.py --data-root {out} --img-size 128 "
          f"--epochs 20 --model resnet18")


if __name__ == "__main__":
    main()
