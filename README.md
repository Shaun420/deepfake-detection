# Deepfake Detection — Deep Learning Mini-Project

A compact, fully reproducible image-forensics pipeline that classifies face images as
**pristine (real)** or **manipulated (fake)** using a convolutional neural network
trained from scratch in PyTorch, with Grad-CAM explainability, classical baselines
and a frequency-domain analysis.

Submitted as the mini-project for the Deep Learning course.

---

## Headline result

**Synthetic surrogate images** (2 000 held-out test images, balanced):

| Model | Test accuracy | ROC-AUC | EER | F1 (fake) |
|---|---|---|---|---|
| Logistic regression on raw pixels | 0.4760 | 0.4853 | – | – |
| Logistic regression on FFT spectrum (Durall et al.) | 0.7500 | 0.8140 | – | – |
| DeepfakeCNN, no rescale aug (`cnn_noaug.pt`) | **0.8895** | **0.9627** | **0.1075** | **0.8902** |
| **DeepfakeCNN, rescale aug — deployed (`best.pt`)** | 0.8260 | 0.9126 | 0.1805 | 0.8296 |

**Synthetic demo video** (40 procedurally generated clips of known label, 12 frames sampled each):

| Model | Clip acc @64 px | Clip acc @256 px | Mean per-frame acc @256 px |
|---|---|---|---|
| `cnn_noaug.pt` | 0.500 | 0.700 | 0.646 |
| **`best.pt` (deployed)** | **0.800** | **0.950** | **0.713** |

The augmentation costs 6 points of image accuracy but takes clip accuracy from 0.70 to
0.95 — the in-domain benchmark ranks the two models in the wrong order. Full discussion in
[`report/REPORT.md`](report/REPORT.md).

---

## Repository layout

```
src/
  data.py        dataset: real-corpus loader + procedural surrogate generator
  model.py       DeepfakeCNN (from scratch) and an optional ResNet-18 transfer baseline
  train.py       training loop (AdamW, cosine LR, label smoothing, grad clipping)
  evaluate.py    test metrics, confusion matrix, ROC/PR curves, score histogram
  gradcam.py     Grad-CAM explainability on the last conv block
  analysis.py    azimuthally averaged FFT power spectrum of both classes
  baselines.py   classical non-deep baselines for context
  predict.py     inference on a single image, a folder, or a video
  video.py       frame sampling, face detection, per-frame scoring, aggregation
  video_eval.py  quantitative clip-level evaluation / resampling ablation
tools/
  prepare_dataset.py  import and optionally normalise a real corpus into data/
  dataset_audit.py    check image metadata for label leakage before training
  make_demo_video.py  render demo clips of known ground truth
figures/         all generated plots
checkpoints/     best.pt, history.json, test_metrics.json, baseline_metrics.json
report/REPORT.md the written mini-project report
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing everything

```bash
python src/data.py                                           # figures/dataset_samples.png
python src/analysis.py                                       # figures/frequency_analysis.png
python src/baselines.py                                      # classical baselines
python src/train.py --epochs 15 --n-train 6000 --n-val 1200  # ~35 min on CPU
python src/train.py --epochs 15 --n-train 6000 --no-rescale-aug --name cnn_noaug  # ablation
python src/video_eval.py --ckpts checkpoints/best.pt checkpoints/cnn_noaug.pt     # video table
python src/evaluate.py --n-test 2000                         # metrics + all plots
python src/gradcam.py --n 6                                  # figures/gradcam.png
python src/predict.py --ckpt checkpoints/best.pt --image some_face.jpg
```

Everything is seeded (`--seed 42`) and runs on CPU; a GPU is used automatically if present.

## Using the real Kaggle dataset: audit first

Target corpus: **[saurabhbagchi/deepfake-image-detection](https://www.kaggle.com/datasets/saurabhbagchi/deepfake-image-detection)**
(504 MB, 983 files, CC0). Its archive expands to

```
Sample_fake_images/
train-20250112T065955Z-001/train/{real,fake}/...
test-20250112T065939Z-001/test/{real,fake}/...
```

Do not train or write up a headline score until the image metadata has been checked.
A model can learn a dimension, aspect-ratio, compression, or file-size shortcut rather
than a forensic trace. The importer is intentionally torch-free, so this check can run
before the training environment is installed:

```bash
pip install kaggle                       # one-time; needs ~/.kaggle/kaggle.json
kaggle datasets download -d saurabhbagchi/deepfake-image-detection
unzip -q deepfake-image-detection.zip -d raw/

python tools/prepare_dataset.py --src raw --out data --val-frac 0.15 --normalise
python tools/dataset_audit.py --root data
```

`--normalise` rewrites every output as an RGB 128 × 128 JPEG, removing the most obvious
image-geometry shortcut. Keep the unnormalised corpus if you want to compare the audit,
but do not silently mix the two in one experiment. The importer preserves an official
test split and carves validation out of training only. Other corpora work the same way:
class folders are matched case-insensitively against real/fake synonyms at any depth.

The real-corpus path uses imbalance-aware metrics and has an explicit majority baseline.
For a faster CPU/GPU run, cache decoded images, use workers, and weight the loss by the
training class frequencies:

```bash
python src/train.py --data-root data --model resnet18 --img-size 128 \
  --epochs 8 --num-workers 4 --cache --class-weights
python src/evaluate.py --ckpt checkpoints/best.pt --data-root data --img-size 128
python src/gradcam.py  --ckpt checkpoints/best.pt --data-root data --img-size 128
```

`evaluate.py` reports accuracy, balanced accuracy, the majority-class baseline, and a
threshold selected on validation scores. If accuracy is below that baseline it prints a
warning. `gradcam.py --data-root` is important: without it, Grad-CAM uses the procedural
surrogate rather than the corpus that produced the checkpoint.

> **Reproducibility note.** No Kaggle corpus is checked into this repository. The
> synthetic results in the report and the figures with synthetic suffixes are not real-
> corpus results. Run the audit and the commands above on a networked Kaggle/Colab
> environment before claiming performance on photographic data.

## Video input

The classifier is image-only, so a video is reduced to still frames:

```bash
python src/predict.py --ckpt checkpoints/best.pt --video clip.mp4 --frames 16
python src/predict.py --ckpt checkpoints/best.pt --video clip.mp4 --label fake  # adds accuracy
```

Uniformly sampled frames → Haar-cascade face crop per frame → per-frame CNN score →
aggregation into a clip verdict, mean P(fake), mean confidence, frame agreement and
(given `--label`) mean per-frame accuracy. A per-frame montage is written to
`figures/video_frames.png`. Demo clips: `python tools/make_demo_video.py --label fake`.

## Why a procedural dataset?

Real deepfake corpora are hundreds of gigabytes and licence-restricted, so the default
dataset is generated on the fly: pristine images are parametric face renderings, and
fakes get the artefacts that real face-swap pipelines are documented to leave behind —
a resampled inner-face region, a soft blending boundary, a colour-transfer mismatch and
a weak transposed-convolution checkerboard residual. **Both** classes then pass through
the same blur + noise + JPEG recompression pipeline so the task cannot be solved by a
trivial sharpness cue. See the report for the caveats this implies.
