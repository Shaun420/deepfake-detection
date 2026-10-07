"""Audit a real/fake image corpus for metadata-only leakage.

A detector can look impressive while learning a collection artefact such as
"all fake images are square" rather than a forgery trace.  This script trains a
small decision tree on image metadata only (dimensions, aspect ratio, byte size,
and format) and reports how accurately that metadata predicts the label.

It is deliberately independent of PyTorch so it can run immediately after
``tools/prepare_dataset.py``::

    python tools/dataset_audit.py --root data

The audit is a warning signal, not proof that the pixels are clean.  A high
metadata-only score means the corpus should be normalised or the split
construction investigated before a model result is written up.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

CLASSES = ("real", "fake")
CLASS_SYNONYMS = {
    0: ("real", "reals", "original", "originals", "authentic", "pristine",
        "training_real", "test_real", "real_images", "0"),
    1: ("fake", "fakes", "deepfake", "deepfakes", "manipulated", "forged",
        "synthetic", "training_fake", "test_fake", "fake_images", "1"),
}
LOOKUP = {name: label for label, names in CLASS_SYNONYMS.items() for name in names}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train", "training", "val", "valid", "validation", "test", "testing"}
FEATURE_NAMES = ("width", "height", "aspect_ratio", "pixels", "file_bytes", "format")


def _split_for(path: Path, root: Path) -> str:
    """Infer a canonical split name from path components, if one is present."""
    for part in reversed(path.relative_to(root).parts[:-1]):
        lower = part.lower()
        if lower in {"train", "training"} or lower.startswith(("train-", "train_")):
            return "train"
        if lower in {"val", "valid", "validation"}:
            return "val"
        if lower in {"test", "testing"} or lower.startswith(("test-", "test_")):
            return "test"
    return "all"


def collect_metadata(root: Path) -> list[dict]:
    """Read image headers and return one metadata row per labelled image."""
    rows: list[dict] = []
    class_dirs: set[Path] = set()
    for directory in sorted([root] + [p for p in root.rglob("*") if p.is_dir()]):
        label = LOOKUP.get(directory.name.strip().lower())
        if label is None or directory in class_dirs:
            continue
        class_dirs.add(directory)
        for image_path in sorted(directory.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in EXTS:
                continue
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    image_format = (image.format or image_path.suffix.lstrip(".")).lower()
                rows.append({
                    "path": str(image_path),
                    "split": _split_for(image_path, root),
                    "label": label,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height if height else 0.0,
                    "pixels": width * height,
                    "file_bytes": image_path.stat().st_size,
                    "format": image_format,
                })
            except (OSError, ValueError) as exc:
                print(f"warning: could not read {image_path}: {exc}", file=sys.stderr)
    return rows


def _matrix(rows: list[dict], formats: Optional[dict[str, int]] = None) -> tuple[np.ndarray, np.ndarray]:
    if formats is None:
        formats = {value: i for i, value in enumerate(sorted({r["format"] for r in rows}))}
    x = np.asarray([
        [r["width"], r["height"], r["aspect_ratio"], r["pixels"],
         r["file_bytes"], formats.get(r["format"], -1)] for r in rows
    ], dtype=np.float64)
    y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    return x, y


def square_fake_accuracy(rows: list[dict]) -> float:
    """Accuracy of the simple rule ``square => fake, non-square => real``."""
    if not rows:
        return float("nan")
    predicted = np.asarray([int(r["width"] == r["height"]) for r in rows])
    labels = np.asarray([r["label"] for r in rows])
    return float(accuracy_score(labels, predicted))


def _metadata_score(rows: list[dict]) -> tuple[float, str]:
    """Return a holdout metadata-tree accuracy and how it was obtained."""
    if len(rows) < 4 or len({r["label"] for r in rows}) < 2:
        return float("nan"), "insufficient classes/images"
    formats = {value: i for i, value in enumerate(sorted({r["format"] for r in rows}))}
    x, y = _matrix(rows, formats)
    split_names = {r["split"] for r in rows}
    if "train" in split_names and "test" in split_names:
        train_rows = [r for r in rows if r["split"] == "train"]
        test_rows = [r for r in rows if r["split"] == "test"]
        if len({r["label"] for r in train_rows}) == 2 and len(test_rows):
            train_x, train_y = _matrix(train_rows, formats)
            test_x, test_y = _matrix(test_rows, formats)
            model = DecisionTreeClassifier(max_depth=3, random_state=42)
            model.fit(train_x, train_y)
            return float(accuracy_score(test_y, model.predict(test_x))), "train->test"

    try:
        train_x, test_x, train_y, test_y = train_test_split(
            x, y, test_size=0.25, random_state=42, stratify=y)
    except ValueError:
        return float("nan"), "insufficient samples per class"
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(train_x, train_y)
    return float(accuracy_score(test_y, model.predict(test_x))), "stratified holdout"


def audit(root: Path) -> dict:
    rows = collect_metadata(root)
    if not rows:
        raise SystemExit(f"no labelled images found under {root}")

    score, protocol = _metadata_score(rows)
    split_counts = defaultdict(Counter)
    for row in rows:
        split_counts[row["split"]][CLASSES[row["label"]]] += 1
    geometry = {
        "widths": sorted({r["width"] for r in rows}),
        "heights": sorted({r["height"] for r in rows}),
        "square_fraction": float(np.mean([r["width"] == r["height"] for r in rows])),
    }
    result = {
        "root": str(root),
        "n_images": len(rows),
        "class_counts": dict(Counter(CLASSES[row["label"]] for row in rows)),
        "split_counts": {split: dict(counts) for split, counts in sorted(split_counts.items())},
        "geometry": geometry,
        "square_implies_fake_accuracy": square_fake_accuracy(rows),
        "metadata_only_accuracy": score,
        "metadata_protocol": protocol,
        "features": list(FEATURE_NAMES),
        "warning": (
            "metadata-only accuracy is high; investigate split/metadata leakage and "
            "consider --normalise before interpreting a CNN score"
            if np.isfinite(score) and score >= 0.75 else
            "no strong metadata-only warning from this audit; pixel-level leakage "
            "and identity overlap still require separate checks"
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data", help="prepared dataset root (default: data)")
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args()

    result = audit(Path(args.root))
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"audit JSON -> {output}")


if __name__ == "__main__":
    main()
