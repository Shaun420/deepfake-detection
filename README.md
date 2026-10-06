# Deepfake Detection — Deep Learning Mini-Project

A compact, fully reproducible image-forensics pipeline that classifies face images as
**pristine (real)** or **manipulated (fake)** using a convolutional neural network
trained from scratch in PyTorch, with Grad-CAM explainability, classical baselines
and a frequency-domain analysis.

Submitted as the mini-project for the Deep Learning course.

---

## Headline result

| Model | Test accuracy | ROC-AUC | EER | F1 (fake) |
|---|---|---|---|---|
| Logistic regression on raw pixels | 0.4760 | 0.4853 | – | – |
| Logistic regression on FFT spectrum (Durall et al.) | 0.7500 | 0.8140 | – | – |
| **DeepfakeCNN (ours, 583 K params)** | **0.8895** | **0.9627** | **0.1075** | **0.8902** |

2 000 held-out test images, perfectly balanced. Full discussion in
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
  predict.py     inference on a single image or a folder
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
python src/train.py --epochs 15 --n-train 6000 --n-val 1200  # ~32 min on CPU
python src/evaluate.py --n-test 2000                         # metrics + all plots
python src/gradcam.py --n 6                                  # figures/gradcam.png
python src/predict.py --ckpt checkpoints/best.pt --image some_face.jpg
```

Everything is seeded (`--seed 42`) and runs on CPU; a GPU is used automatically if present.

## Using a real dataset

The code was written so the surrogate data can be swapped for a real corpus
(FaceForensics++, Celeb-DF, or the Kaggle *140k Real and Fake Faces* set) without
touching the model. Extract cropped faces into:

```
data/{train,val,test}/{real,fake}/*.jpg
```

then add `--data-root data` to `train.py` / `evaluate.py`. A pretrained ResNet-18
baseline is also available with `--model resnet18`.

## Why a procedural dataset?

Real deepfake corpora are hundreds of gigabytes and licence-restricted, so the default
dataset is generated on the fly: pristine images are parametric face renderings, and
fakes get the artefacts that real face-swap pipelines are documented to leave behind —
a resampled inner-face region, a soft blending boundary, a colour-transfer mismatch and
a weak transposed-convolution checkerboard residual. **Both** classes then pass through
the same blur + noise + JPEG recompression pipeline so the task cannot be solved by a
trivial sharpness cue. See the report for the caveats this implies.
