"""Frequency-domain analysis of the real vs. manipulated classes.

Up-sampling operations inside a face-swap / GAN pipeline leave a signature in
the Fourier spectrum (Durall et al. 2020; Frank et al. 2020).  This script
plots the azimuthally averaged power spectrum of both classes, which motivates
why a CNN can separate them at all.

    python src/analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import make_sample


def radial_spectrum(img: np.ndarray) -> np.ndarray:
    g = img.mean(-1)
    g = g - g.mean()
    f = np.fft.fftshift(np.abs(np.fft.fft2(g)))
    h, w = f.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    nb = r.max() + 1
    prof = np.bincount(r.ravel(), f.ravel(), nb) / np.bincount(r.ravel(), None, nb)
    return 20 * np.log10(prof + 1e-8)


def main(n: int = 400):
    spec = {0: [], 1: []}
    for i in range(n):
        label = i % 2
        spec[label].append(radial_spectrum(make_sample(3_000_000 + i, label)))
    Path("figures").mkdir(exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for label, name, c in [(0, "real", "tab:blue"), (1, "fake", "tab:red")]:
        a = np.stack(spec[label])
        m, s = a.mean(0), a.std(0)
        x = np.arange(len(m))
        ax[0].plot(x, m, color=c, label=name)
        ax[0].fill_between(x, m - s, m + s, color=c, alpha=.18)
    ax[0].set_xlabel("spatial frequency (radial bin)")
    ax[0].set_ylabel("power (dB)")
    ax[0].set_title("Azimuthally averaged power spectrum")
    ax[0].legend(); ax[0].grid(alpha=.3)

    d = np.stack(spec[1]).mean(0) - np.stack(spec[0]).mean(0)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].plot(d, color="tab:purple")
    ax[1].set_xlabel("spatial frequency (radial bin)")
    ax[1].set_ylabel("fake - real (dB)")
    ax[1].set_title("Spectral difference (forensic signature)")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig("figures/frequency_analysis.png", dpi=140)
    print("wrote figures/frequency_analysis.png  "
          f"max |delta| = {np.abs(d).max():.2f} dB at bin {int(np.abs(d).argmax())}")


if __name__ == "__main__":
    main()
