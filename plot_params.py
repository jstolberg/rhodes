"""Overview plots of a step-2 fit (step2_fit.npz) into plots/: every parameter across the
keyboard, the same against f_0, and the per-cell residuals.  Rerun after any fit.

    python plot_params.py [step2_fit.npz]
"""
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from step2_lib import (DISP_MAX, DYNS, J_FIT, PD_RANGE, PO_MAX, R_FIX, VELS, features, load_keys,
                       predict, raw_a0, raw_pd, raw_po)

FIT = sys.argv[1] if len(sys.argv) > 1 else "step2_fit.npz"
OUT = "plots"


def residuals(z, keys):
    """rms log residual per cell (Mk, N_DYN) and per key (Mk,) of the fit z on the real features."""
    Ht, eta, mask = features("real", keys)
    theta = dict(po_raw=raw_po(z["p_o"]), pd_raw=raw_pd(z["p_d"]),
                 log_sig=jnp.log(jnp.asarray(z["sigma"])), A0_raw=raw_a0(z["A0"]))
    pred = np.asarray(jnp.logaddexp(float(np.log(z["kappa"])) + predict(theta, keys), jnp.log(eta)))
    d = np.where(mask, np.log(Ht + eta) - pred, 0.0)
    cell = np.sqrt((d ** 2).sum((2, 3)) / np.maximum(mask.sum((2, 3)), 1))
    key  = np.sqrt((d ** 2).sum((1, 2, 3)) / np.maximum(mask.sum((1, 2, 3)), 1))
    return cell, key


def keys_axis(ax, notes, ylabel, log=False):
    x = np.arange(len(notes))
    ax.set_xticks(x[::3]); ax.set_xticklabels(notes[::3], rotation=90, fontsize=8)
    ax.set_xlim(-1, len(notes)); ax.grid(alpha=0.3, which="both"); ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")


def plot_across_keys(z, keys, rms_cell, rms_key, path):
    n, x = z["notes"], np.arange(len(z["notes"]))
    A0 = z["A0"] * 1e3
    fig, ax = plt.subplots(6, 1, figsize=(16, 22), sharex=True)

    ax[0].plot(x, z["p_o"] * 1e3, "o-", ms=4)
    ax[0].axhline(PO_MAX * 1e3, color="r", ls=":", lw=0.8)
    ax[0].axhline(R_FIX * 1e3, color="k", ls=":", lw=0.8)
    ax[0].text(0.5, R_FIX * 1e3, " r (pickup radius)", va="bottom", fontsize=8)
    keys_axis(ax[0], n, "p_o (mm)  (dotted red: bound)")

    ax[1].plot(x, z["p_d"] * 1e3, "o-", ms=4)
    for b in PD_RANGE:
        ax[1].axhline(b * 1e3, color="r", ls=":", lw=0.8)
    keys_axis(ax[1], n, "p_d (mm)  (dotted: bounds)")

    ax[2].plot(x, z["sigma"], "o-", ms=4, label="fitted sigma_0")
    ax[2].plot(x, keys["sig1"], "x", color="0.5", label="step-1 decay (init)")
    ax[2].legend()
    keys_axis(ax[2], n, "sigma_0 (1/s)", log=True)

    for j, dyn in enumerate(DYNS):
        ax[3].plot(x, A0[:, j], "o-", ms=3, color=f"C{j}", label=dyn)
    ax[3].axhline(DISP_MAX * 1e3, color="r", ls=":", lw=0.8)
    ax[3].legend(ncol=4)
    keys_axis(ax[3], n, "A_0 (mm)  (dotted: DISP_MAX)", log=True)

    for j, dyn in enumerate(DYNS[:J_FIT]):
        ax[4].plot(x, A0[:, j] / A0[:, J_FIT], "o-", ms=3, color=f"C{j}", label=f"{dyn}/f")
        ax[4].axhline(VELS[j], color=f"C{j}", ls=":", lw=0.8)
    ax[4].set_ylim(0, 1.5); ax[4].legend(ncol=3)
    keys_axis(ax[4], n, "A_0 ratio to f  (dotted: assumed velocities)")

    for j, dyn in enumerate(DYNS):
        ax[5].plot(x, rms_cell[:, j], ".", color=f"C{j}", label=dyn)
    ax[5].plot(x, rms_key, "k-", lw=1, label="key")
    ax[5].legend(ncol=5)
    keys_axis(ax[5], n, "rms log residual  (1 = 8.7 dB)")
    ax[5].set_xlabel("key (library names, one octave low)")

    plt.suptitle(f"Step 2 fit ({FIT}), kappa = {float(z['kappa']):.2e}")
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close(fig)


def plot_vs_f0(z, path):
    """Geometry and amplitudes against f_0: physical trends should be smooth here."""
    f0, A0 = z["f0"], z["A0"] * 1e3
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    ax[0].semilogx(f0, z["p_o"] * 1e3, "o", ms=4); ax[0].set_ylabel("p_o (mm)")
    ax[1].semilogx(f0, z["p_d"] * 1e3, "o", ms=4); ax[1].set_ylabel("p_d (mm)")
    ax[2].loglog(f0, z["sigma"], "o", ms=4); ax[2].set_ylabel("sigma_0 (1/s)")
    for j, dyn in enumerate(DYNS):
        ax[3].loglog(f0, A0[:, j], "o", ms=3, color=f"C{j}", label=dyn)
    ax[3].set_ylabel("A_0 (mm)"); ax[3].legend()
    for a in ax:
        a.set_xlabel("f_0 (Hz)"); a.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close(fig)


def print_outliers(z, keys, rms_key):
    n, A0 = z["notes"], z["A0"]
    print("p_o > r:", n[z["p_o"] > R_FIX].tolist())
    print("p_d on a bound:", n[(z["p_d"] < PD_RANGE[0] + 0.05e-3) | (z["p_d"] > PD_RANGE[1] - 0.05e-3)].tolist())
    print("A_0(f) within 5% of DISP_MAX:", n[A0[:, J_FIT] > 0.95 * DISP_MAX].tolist())
    print("rms > 0.9:", [(k, round(float(v), 2)) for k, v in zip(n, rms_key) if v > 0.9])
    q = z["sigma"] / keys["sig1"]
    print(f"sigma fitted / step-1: median {np.median(q):.2f}, range {q.min():.2f}-{q.max():.2f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    z = np.load(FIT, allow_pickle=True)
    keys = load_keys()
    assert (keys["notes"] == z["notes"]).all(), "fit is for other keys than KEY_STRIDE selects"
    rms_cell, rms_key = residuals(z, keys)
    plot_across_keys(z, keys, rms_cell, rms_key, f"{OUT}/step2_params.png")
    plot_vs_f0(z, f"{OUT}/step2_params_vs_f0.png")
    print_outliers(z, keys, rms_key)
    print(f"-> {OUT}/step2_params.png, {OUT}/step2_params_vs_f0.png")
