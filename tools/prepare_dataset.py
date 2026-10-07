"""Turn a downloaded real deepfake corpus into the layout this project expects.

Target layout::

    data/
      train/{real,fake}/*.jpg
      val/{real,fake}/*.jpg
      test/{real,fake}/*.jpg

The importer intentionally has no dependency on PyTorch.  This makes it useful in
an environment where the dataset preparation step is run before the training
environment is installed, and avoids importing the model's data module just to
look up two class names.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

CLASSES = ("real", "fake")
CLASS_SYNONYMS = {
    0: ("real", "reals", "original", "originals", "authentic", "pristine",
        "training_real", "test_real", "real_images", "0"),
    1: ("fake", "fakes", "deepfake", "deepfakes", "manipulated", "forged",
        "synthetic", "training_fake", "test_fake", "fake_images", "1"),
}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOOKUP = {name: label for label, names in CLASS_SYNONYMS.items() for name in names}
SPLIT_HINTS = {
    "train": "train", "training": "train",
    "test": "test", "testing": "test",
    "val": "val", "valid": "val", "validation": "val",
}


def scan(src: Path):
    """Return ``{(split_hint, label): [paths]}`` for class folders below *src*."""
    found = defaultdict(list)
    directories = [src] + [p for p in src.rglob("*") if p.is_dir()]
    for directory in sorted(directories):
        label = LOOKUP.get(directory.name.strip().lower())
        if label is None:
            continue
        images = [p for p in sorted(directory.rglob("*"))
                  if p.is_file() and p.suffix.lower() in EXTS]
        if not images:
            continue

        split = None
        for part in reversed(directory.parts):
            key = part.strip().lower()
            for hint, canonical in SPLIT_HINTS.items():
                if key == hint or key.startswith(hint + "-") or key.startswith(hint + "_"):
                    split = canonical
                    break
            if split:
                break
        found[(split or "train", label)].extend(images)
    return found


def _normalise_image(source: Path, destination: Path, size: int) -> None:
    """Write an RGB, square JPEG with no source-dimension metadata shortcut."""
    from PIL import Image, ImageOps

    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize((size, size), resampling)
        image.save(destination, format="JPEG", quality=95, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="folder containing the extracted corpus")
    parser.add_argument("--out", default="data")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="fraction of the TRAIN split held out for validation")
    parser.add_argument("--test-frac", type=float, default=0.15,
                        help="used only when the corpus has no test split")
    parser.add_argument("--copy", action="store_true",
                        help="copy files instead of symlinking")
    parser.add_argument("--face-crop", action="store_true",
                        help="detect and save the largest face crop instead of the full image")
    parser.add_argument("--normalise", "--normalize", action="store_true",
                        help="rewrite every output image as a square RGB JPEG")
    parser.add_argument("--normalise-size", "--normalize-size", type=int, default=128,
                        help="side length used with --normalise (default: 128)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 <= args.val_frac < 1 or not 0 <= args.test_frac < 1:
        parser.error("--val-frac and --test-frac must be in [0, 1)")
    if args.normalise_size < 1:
        parser.error("--normalise-size must be positive")

    src, out = Path(args.src), Path(args.out)
    if not src.is_dir():
        sys.exit(f"source folder not found: {src}")

    found = scan(src)
    if not found:
        sys.exit(f"no real/fake class folders found under {src}.\n"
                 f"Expected directory names among: {sorted(LOOKUP)}")

    print("discovered:")
    for (split, label), images in sorted(found.items()):
        print(f"  {split:5s} / {CLASSES[label]:4s}: {len(images)} images")

    rng = random.Random(args.seed)
    has_test = any(split == "test" for split, _ in found)
    final = defaultdict(lambda: defaultdict(list))
    for (split, label), images in found.items():
        images = sorted(set(images))
        rng.shuffle(images)
        if split == "train":
            if has_test:
                n_val = int(len(images) * args.val_frac)
                final["val"][label].extend(images[:n_val])
                final["train"][label].extend(images[n_val:])
            else:
                n_test = int(len(images) * args.test_frac)
                n_val = int(len(images) * args.val_frac)
                final["test"][label].extend(images[:n_test])
                final["val"][label].extend(images[n_test:n_test + n_val])
                final["train"][label].extend(images[n_test + n_val:])
        else:
            final[split][label].extend(images)

    cropper = None
    if args.face_crop:
        import cv2
        # Keep this optional import local: the normal importer remains lightweight.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from video import detect_face
        cropper = (cv2, detect_face)

    manifest = {}
    n_written, n_nocrop, n_failed = 0, 0, 0
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        manifest[split] = {}
        for label, class_name in enumerate(CLASSES):
            destination_dir = out / split / class_name
            if destination_dir.exists():
                shutil.rmtree(destination_dir)
            destination_dir.mkdir(parents=True, exist_ok=True)

            written_for_class = 0
            for index, source in enumerate(final[split][label]):
                suffix = ".jpg" if args.normalise or args.face_crop else source.suffix.lower()
                destination = destination_dir / f"{index:06d}_{source.stem[:40]}{suffix}"
                try:
                    if args.face_crop:
                        cv2, detect_face = cropper
                        image = cv2.imread(str(source))
                        if image is None:
                            raise ValueError("OpenCV could not decode the image")
                        box = detect_face(image)
                        if box is None:
                            n_nocrop += 1
                        else:
                            x, y, width, height = box
                            image = image[y:y + height, x:x + width]
                        if args.normalise:
                            from PIL import Image
                            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            pil = Image.fromarray(rgb)
                            resampling = getattr(Image, "Resampling", Image).LANCZOS
                            pil.resize((args.normalise_size, args.normalise_size), resampling).save(
                                destination, format="JPEG", quality=95, optimize=True)
                        else:
                            cv2.imwrite(str(destination), image)
                    elif args.normalise:
                        _normalise_image(source, destination, args.normalise_size)
                    elif args.copy:
                        shutil.copy2(source, destination)
                    else:
                        try:
                            destination.symlink_to(source.resolve())
                        except OSError:
                            shutil.copy2(source, destination)
                except (OSError, ValueError) as exc:
                    n_failed += 1
                    print(f"warning: skipped {source}: {exc}", file=sys.stderr)
                    continue
                n_written += 1
                written_for_class += 1
            manifest[split][class_name] = written_for_class

    with (out / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nwrote {n_written} images to {out}/")
    if args.face_crop and n_nocrop:
        print(f"  ({n_nocrop} images had no detectable face and were stored uncropped)")
    if n_failed:
        print(f"  ({n_failed} images could not be decoded and were skipped)")
    for split in ("train", "val", "test"):
        total = sum(manifest[split].values())
        print(f"  {split:5s}: {manifest[split]} (total {total})")
    print(f"\nmanifest -> {out / 'manifest.json'}")
    print(f"\nnow run:\n  python src/train.py --data-root {out} --img-size 128 "
          f"--epochs 20 --model resnet18")


if __name__ == "__main__":
    main()
