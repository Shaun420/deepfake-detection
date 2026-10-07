## Summary

This PR makes the real-corpus path safer and more useful before a headline result is
written up:

- adds a torch-free metadata-leakage audit at `tools/dataset_audit.py`;
- adds square RGB normalisation to `tools/prepare_dataset.py`;
- fixes Grad-CAM so `--data-root` explains the corpus used by the checkpoint;
- adds balanced accuracy, a majority-class baseline, and validation threshold calibration
to evaluation;
- adds RAM caching, DataLoader worker plumbing, and optional inverse-frequency class
  weights to training; and
- documents the audit-first protocol and separates synthetic from Kaggle evidence in the
  report and README.

## Validation

- `python -m py_compile tools/dataset_audit.py tools/prepare_dataset.py src/data.py src/evaluate.py src/gradcam.py src/train.py`
- `python tools/prepare_dataset.py --help` (confirmed the importer has no PyTorch import)
- `python -m py_compile ...` and `git diff --check`

The sandbox Python environment is missing Pillow, NumPy, and the training dependencies,
so the mock-corpus runtime checks could not be rerun here. The Kaggle corpus is not
checked into this repository, so a real-data audit/training run still needs to be performed
in a networked environment. The report explicitly preserves that caveat; the existing
0.889 confusion matrix and demo video are synthetic results.
