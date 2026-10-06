"""Non-deep baselines, so the CNN result can be put in context.

Two classical detectors are fitted on the same training split:

  * `pixel-LR`   : logistic regression on raw down-sampled pixels,
  * `fft-LR`     : logistic regression on the azimuthally averaged FFT power
                   spectrum (the hand-crafted forensic feature of
                   Durall et al., CVPR 2020).

    python src/baselines.py
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import radial_spectrum
from data import make_sample

IMG = 64


def build(n, offset):
    X_pix, X_fft, y = [], [], []
    for i in range(n):
        label = i % 2
        a = make_sample(offset + i, label, IMG)
        X_pix.append(a[::2, ::2].ravel())
        X_fft.append(radial_spectrum(a))
        y.append(label)
    return np.array(X_pix), np.array(X_fft), np.array(y)


def main(n_train=2000, n_test=1000):
    Xp, Xf, y = build(n_train, 0)
    Xp_t, Xf_t, y_t = build(n_test, 2_000_000)

    results = {}
    for name, (A, B) in {"pixel-LR": (Xp, Xp_t), "fft-LR": (Xf, Xf_t)}.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(A, y)
        p = clf.predict_proba(B)[:, 1]
        results[name] = {"accuracy": float(accuracy_score(y_t, p >= .5)),
                         "roc_auc": float(roc_auc_score(y_t, p))}
        print(f"{name:10s} acc={results[name]['accuracy']:.4f} "
              f"auc={results[name]['roc_auc']:.4f}")
    json.dump(results, open("checkpoints/baseline_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
