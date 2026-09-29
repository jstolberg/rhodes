# %% [markdown]
# # Step 2: Pickup and free oscillation
#
#     python step2_pickup.py [real|syn] [stride]
#
# Runs the multi-start (initial values, cached in `cache/`) and then the fit.  With
# `real` and stride 1 the result is written to `results/step2_pickup.npz` for step 3.
#
#
# The model in this step is a single free mode, $x(t) = p_o + A_0 e^{-\sigma_0 t}
# \sin 2\pi f_0 t$, seen through the pickup.  The pickup is a static
# nonlinearity, so the output has energy only at $n f_0$.  Rather than an STFT
# over all bins, both target and synthetic signals are projected onto the
# fundamental and its first four harmonics in a sequence of Hann-windowed frames,
# $$H[n, k] = \int w(t - t_k)\, \epsilon(t)\, e^{-2\pi i\, n f_0 t}\, \mathrm{d}t,
#   \qquad n = 1 \ldots 5,$$
# and the loss is the mean square of the log-magnitude difference, with a
# per-frame noise floor $\eta$ and a mask,
# $$L = \frac{1}{\sum m}\sum_{n,k} m[n,k]\big(\log(|H_t| + \eta) - \log(\kappa
#   |H_s| + \eta)\big)^2 .$$
# $\kappa$ is one global gain (the recording level and the pickup constant are
# both arbitrary); it is not a free parameter but the closed-form level match
# $\log\kappa = \overline{\log|H_t| - \log|H_s|}$ over the kept cells.  The
# coil's RLC filter is ignored for now.
#
# The pickup surface is fixed from photos ($r \approx 4$ mm,
# $a \approx 1.3$ mm, $m \approx 1$), so $\Psi(x, z)$ is tabulated **once**;
# the per-key $(p_o, p_d)$ enter only through the lookup.  $A_0$ is the
# amplitude at $t_0$ = onset + `T_START`, after the hammer has left; step 3
# needs the amplitude at handover, which differs by $e^{\sigma_0 (t_0 - \tau)}$.
# The decay of step 1 sets the frame span per key and the starting $\sigma_0$.
# Definitions and hyperparameters: below.
#
# ## Three stages
#
# $p_o$, $p_d$, $\sigma_0$ are per key, $A_0$ per key and dynamic.  The
# starting geometry comes from the multi-start below (cached).
#
# * **A** -- $p_o$, $p_d$, $\sigma_0$, $A_0(f)$ from the f samples alone.
# * **B** -- $A_0$ of p, mp, mf per cell with everything else fixed (one
#   parameter per cell, started from the best of a log-spaced scan).
# * **C** -- all parameters, all cells, fine-tuned together.
#
# | Name | Value | Meaning |
# | --- | --- | --- |
# | `STEPS_A`, `STEPS_B`, `STEPS_C` | 300, 100, 300 | Adam steps of the three stages |
# | `A0_SCAN` | 32 values, 0.01-9.5 mm | log-spaced $A_0$ scan per cell that starts stage B |

# %%
"""Step 2 (pickup and free oscillation): definitions.

Model: one free mode x(t) = p_o + A_0 e^{-sigma_0 t} sin 2 pi f_0 t seen through the
pickup, eps = -kappa dPsi(x, p_d)/dt, with the surface fixed from photos and Psi tabulated
once.  Target and model are projected onto the fundamental and its first four harmonics in
Hann frames; the loss is the mean squared log-magnitude difference over the cells above the
noise floor, with kappa the closed-form level match.

p_o, p_d, sigma_0 are per key, A_0 per key and dynamic.  The decay of step 1
(fundamentals.npz) sets the frame span per key and the starting sigma_0.  Keys in FLAGGED
are fitted like the others and marked in the output, for exclusion downstream.


Hyperparameters
| Name | Value | Meaning |
| --- | --- | --- |
| `DYNS`, `FIT_DYN` | p, mp, mf, f; f | dynamics used; the one the geometry is fitted on |
| `VELS` | .25 .5 .75 1 | assumed velocities (synthetic test only) |
| `N_HARM` | 5 | fundamental + 4 harmonics |
| `T_START` | 10 ms | first frame starts this long after onset (past contact) |
| `ONSET_FRAC` | 0.05 | onset = first sample above this fraction of the peak |
| `N_PERIODS` | 64 | frame length in periods of f_0: the 7.1 f_0 partial and its intermodulation products at 5.1, 6.1 f_0 fall > 6 bins from a harmonic |
| `K_FRAMES` | 12 | frames per sample, spread evenly over the frame span |
| `T_FIT`, `DECAY_SPAN` | 6 s, 4 | frame span per key min(T_FIT, 4 / sigma_1) after t_0, sigma_1 from step 1 |
| `L_FRAME` | 1024 | synthetic samples per frame (16 per period, Nyquist 8 f_0) |
| `NOISE_MARGIN` | 3 | keep cells with |H_t| > margin x noise (9.5 dB; noise is Rayleigh, 3 passes ~3% of noise cells) |
| `R_FIX`, `A_FIX`, `M_FIX` | 4 mm, 1.3 mm, 1 | pickup geometry from photos (fixed) |
| `SURF_N` | 64 | surface grid |
| `DISP_MAX`, `N_TABLE` | 10 mm, 8192 | A_0 <= DISP_MAX (bass tines sweep fully past the pickup); 2.6 um table |
| `PO_MAX` | r = 4 mm | p_o bounded to (0, PO_MAX) via sigmoid: the tine rests over the pole (also the table extent) |
| `PD_RANGE`, `N_PD` | 0.5-3.5 mm, 96 | p_d bounded to this range via sigmoid, tabulated on N_PD rows |
| `FLAGGED` | F0, G0, D6 | samples detuned > 8 cents from their note (13-18, -9 cents; all others within 6.5) |
| `LR` | 0.05 | Adam learning rate (all stages) |
| `KEY_STRIDE` | 1 | fit every k-th key (1 = all 73, 6 = 13 keys, 24 = E0, E2, E4, E6); runtime scales with it |
| `CACHE` | cache/ | target features and init results |
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import scipy.io.wavfile as wavfile

jax.config.update("jax_enable_x64", True)

from model import z_trapz, make_surface, drop_masked, psi_table, psi_lookup

# ---------- hyperparameters ----------
SAMPLES      = "Samples"
FUNDAMENTALS = "results/step1_fundamentals.npz"
RESULTS      = "results/step2_pickup.npz"
CACHE        = "cache"
DYNS         = ["p", "mp", "mf", "f"]
FIT_DYN      = "f"
VELS         = np.array([0.25, 0.5, 0.75, 1.0])
N_DYN, J_FIT = len(DYNS), DYNS.index(FIT_DYN)

N_HARM       = 5
T_START      = 10e-3
ONSET_FRAC   = 0.05
N_PERIODS    = 64
K_FRAMES     = 12
T_FIT        = 6.0
DECAY_SPAN   = 4.0
L_FRAME      = 1024
NOISE_MARGIN = 3.0

R_FIX, A_FIX, M_FIX = 4e-3, 1.3e-3, 1.0
SURF_N       = 64
DISP_MAX, N_TABLE = 10.0e-3, 8192
PO_MAX       = R_FIX
PD_RANGE, N_PD = (0.5e-3, 3.5e-3), 96
FLAGGED      = ["F0", "G0", "D6"]
LR           = 0.05
KEY_STRIDE   = 1      # every k-th key (24 -> E0, E2, E4, E6; library names are one octave low)

HARM = np.arange(1, N_HARM + 1)


# ---------- framing (shared by target and synthetic) ----------

def frame_times(f0, t_fit):
    """Frame length T and the K frame starts (s after t_0), evenly over t_fit."""
    T   = N_PERIODS / f0
    hop = (t_fit - T) / (K_FRAMES - 1)
    return T, hop * np.arange(K_FRAMES)


def frame_span(f0s, sig1):
    """Per-key frame span: DECAY_SPAN decay times, at least two frames, at most T_FIT."""
    return np.clip(DECAY_SPAN / sig1, 2 * N_PERIODS / f0s, T_FIT)


def hann(n):
    return 0.5 * (1.0 - np.cos(2 * np.pi * (np.arange(n) + 0.5) / n))


# ---------- keys: fundamentals and decays of step 1 ----------

def load_keys(stride=KEY_STRIDE):
    """notes, f_0, sigma_1 (step-1 decay per key, gaps filled by log interpolation over the
    keyboard) and the per-key frame span, for every stride-th key."""
    data    = np.load(FUNDAMENTALS, allow_pickle=True)
    notes   = data["notes"]
    tab     = data["table"]
    f0_all  = np.nanmedian(tab["f"], axis=1)
    lam     = np.where(tab["ok"], tab["lam"], np.nan)
    sig_all = np.array([np.nanmedian(r) if np.isfinite(r).any() else np.nan for r in lam])
    have    = np.isfinite(sig_all)
    sig_all = np.exp(np.interp(np.arange(len(notes)), np.where(have)[0], np.log(sig_all[have])))
    sel     = np.arange(0, len(notes), stride)
    keys = dict(notes=notes[sel], f0s=f0_all[sel], sig1=sig_all[sel], tfit=frame_span(f0_all[sel], sig_all[sel]),
                flag=np.isin(notes[sel], FLAGGED))
    print(f"{len(sel)} keys: {keys['notes'][0]} ({keys['f0s'][0]:.1f} Hz) ... {keys['notes'][-1]} ({keys['f0s'][-1]:.1f} Hz)")
    print("step-1 sigma (1/s):", np.round(keys["sig1"], 3), " frame span (s):", np.round(keys["tfit"], 2))
    return keys


# ---------- target side (numpy, once, cached) ----------

def load_wave(note, dyn):
    fs, x = wavfile.read(f"{SAMPLES}/{note}-{dyn}.wav")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, float(fs)


def onset(x):
    return int(np.argmax(np.abs(x) > ONSET_FRAC * np.abs(x).max()))


def project_target(x, fs, f0, t_fit):
    """|H_t| (N_HARM, K) and the per-frame noise floor: mean magnitude at the midpoints
    (n + 1/2) f_0, broadcast over harmonics."""
    T, starts = frame_times(f0, t_fit)
    Lf = int(round(T * fs))
    w  = hann(Lf)
    n  = np.arange(Lf) / fs
    E_h = np.exp(-2j * np.pi * np.outer(HARM * f0, n))
    E_m = np.exp(-2j * np.pi * np.outer((np.arange(N_HARM) + 0.5) * f0, n))

    i0 = onset(x) + int(round(T_START * fs))
    H, mid = np.zeros((N_HARM, K_FRAMES)), np.zeros((N_HARM, K_FRAMES))
    for k, s in enumerate(starts):
        a = i0 + int(round(s * fs))
        seg = w * x[a:a + Lf]
        H[:, k]   = np.abs(E_h @ seg) / fs
        mid[:, k] = np.abs(E_m @ seg) / fs
    return H, np.repeat(mid.mean(axis=0, keepdims=True), N_HARM, axis=0)


def build_target(keys, waves):
    """waves[(i, j)] -> (x, fs).  Returns |H_t|, noise and mask, all (Mk, N_DYN, N_HARM, K)."""
    Mk = len(keys["f0s"])
    H, eta = np.zeros((Mk, N_DYN, N_HARM, K_FRAMES)), np.zeros((Mk, N_DYN, N_HARM, K_FRAMES))
    for i, f0 in enumerate(keys["f0s"]):
        for j in range(N_DYN):
            x, fs = waves[(i, j)]
            H[i, j], eta[i, j] = project_target(x, fs, f0, keys["tfit"][i])
    return H, eta, H > NOISE_MARGIN * eta


def features(tag, keys, stride=KEY_STRIDE):
    """Target features of the real samples (tag 'real') or the synthetic waves (tag 'syn'),
    cached in CACHE/features_{tag}_{stride}.npz."""
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/features_{tag}_{stride}.npz"
    if os.path.exists(path):
        z = np.load(path)
        print(f"features from {path}")
        return z["H"], z["eta"], z["mask"]
    t0 = time.time()
    if tag == "real":
        waves = {(i, j): load_wave(n, d) for i, n in enumerate(keys["notes"]) for j, d in enumerate(DYNS)}
    else:
        waves = synth_waves(theta_true(keys), KAPPA_SYN, keys, jax.random.PRNGKey(1))
    H, eta, mask = build_target(keys, waves)
    np.savez(path, H=H, eta=eta, mask=mask)
    print(f"features ({tag}) built in {time.time() - t0:.0f} s -> {path}")
    return H, eta, mask


def report_mask(mask):
    kept = mask.mean(axis=(0, 1, 3))
    print("frames kept per harmonic:", " ".join(f"H{n}:{k:.0%}" for n, k in zip(HARM, kept)))
    print(f"kept frames per key, dyn {FIT_DYN}, H1..H{N_HARM}:", mask[:, J_FIT].sum(-1).tolist())


# ---------- synthetic side (jax, differentiable) ----------

# Fixed surface -> one static Psi(x, z) table.  x spans the displacement plus
# the p_o range, z the p_d range; p_o and p_d then only enter via the lookup.
PTS, W = drop_masked(*make_surface(partial(z_trapz, a=A_FIX, m=M_FIX), N=SURF_N, r_max=R_FIX))
TABLE  = psi_table(dict(p_d=PD_RANGE[0]), PTS, W, DISP_MAX, PO_MAX,
                   pd_range=PD_RANGE, n=N_TABLE, n_pd=N_PD)


def eps_free(t, A, sig, f, p_o, p_d, table):
    """-dPsi/dt for free modes (A, sig, f arrays) at times t (any shape)."""
    def psi(tt):
        x = p_o + jnp.sum(A * jnp.exp(-sig * tt) * jnp.sin(2 * jnp.pi * f * tt))
        return psi_lookup(jnp.array([x, 0.0, p_d]), table)
    d = jax.vmap(lambda tt: jax.jvp(psi, (tt,), (1.0,))[1])(t.ravel())
    return -d.reshape(t.shape)


def project_synth(A0, sig, f0, t_fit, p_o, p_d, table, kidx=None, L=L_FRAME):
    """|H_s| (N_HARM, K): epsilon on each frame's own grid, Hann, projected.
    kidx selects a subset of the frames, L the samples per frame (reduced evaluation)."""
    T   = N_PERIODS / f0
    hop = (t_fit - T) / (K_FRAMES - 1)
    ks  = jnp.arange(K_FRAMES) if kidx is None else jnp.asarray(kidx)
    rel = (jnp.arange(L) + 0.5) / L
    tt  = hop * ks[:, None] + T * rel[None, :]                              # (K, L)
    e   = eps_free(tt, A0[None], sig[None], f0[None], p_o, p_d, table)     # (K, L)
    E   = jnp.exp(-2j * jnp.pi * jnp.outer(HARM * f0, T * rel))            # (N, L)
    H   = jnp.einsum("nl,kl->nk", E, jnp.asarray(hann(L)) * e) * (T / L)
    return jnp.abs(H)


def unpack(theta):
    """Per-key p_d, p_o, sigma (Mk,), p_d/p_o bounded to the table; A_0 (Mk, n_dyn) inside the table."""
    lo, hi = PD_RANGE
    g = dict(p_d=lo + (hi - lo) * jax.nn.sigmoid(theta["pd_raw"]),
             p_o=PO_MAX * jax.nn.sigmoid(theta["po_raw"]),
             sig=jnp.exp(theta["log_sig"]))
    return g, DISP_MAX * jax.nn.sigmoid(theta["A0_raw"])


raw_po = lambda po: jax.scipy.special.logit(jnp.asarray(po) / PO_MAX)
raw_pd = lambda pd: jax.scipy.special.logit((jnp.asarray(pd) - PD_RANGE[0]) / (PD_RANGE[1] - PD_RANGE[0]))
raw_a0 = lambda a0: jax.scipy.special.logit(jnp.asarray(a0) / DISP_MAX)


def predict(theta, keys, kidx=None, L=L_FRAME):
    """log|H_s| for kappa = 1: (Mk, n_dyn, N_HARM, K), n_dyn = columns of A0_raw."""
    g, A0 = unpack(theta)

    @jax.checkpoint                                         # recompute in backward: O(1) memory per cell
    def cell(A0, sig, f0, t_fit, p_o, p_d):
        return jnp.log(project_synth(A0, sig, f0, t_fit, p_o, p_d, TABLE, kidx, L))

    over_dyn = jax.vmap(cell, in_axes=(0, None, None, None, None, None))
    return jax.vmap(over_dyn)(A0, g["sig"], jnp.asarray(keys["f0s"]), jnp.asarray(keys["tfit"]), g["p_o"], g["p_d"])


def masked_mean(x, mask, axis=None):
    return jnp.sum(jnp.where(mask, x, 0.0), axis=axis) / jnp.maximum(mask.sum(axis=axis), 1)


def loss(theta, keys, logHt, eta, mask, log_kappa=None):
    """Mean squared log-magnitude residual over the kept cells; kappa closed-form (level
    match over all kept cells) unless given.  Returns (loss, log_kappa)."""
    logHs = predict(theta, keys)
    if log_kappa is None:
        log_kappa = masked_mean(logHt - logHs, mask)
    pred = jnp.logaddexp(log_kappa + logHs, jnp.log(eta))
    return masked_mean((logHt - pred) ** 2, mask), log_kappa


def loss_per_key(theta, keys, logHt, eta, mask, kidx=None, L=L_FRAME):
    """Same, per key with kappa closed per key: gain-free, so only the harmonic ratios and
    the decay shapes count.  Used to compare starts."""
    logHs = predict(theta, keys, kidx, L)
    ax    = (1, 2, 3)
    lk    = masked_mean(logHt - logHs, mask, ax)
    pred  = jnp.logaddexp(lk[:, None, None, None] + logHs, jnp.log(eta))
    return masked_mean((logHt - pred) ** 2, mask, ax)


def fit(params, loss_fn, steps, lr=LR, label="", state=None):
    """Adam on params for loss_fn(params) -> scalar.  Returns params, loss history, state."""
    opt   = optax.adam(lr)
    state = opt.init(params) if state is None else state

    @jax.jit
    def step(params, state):
        l, g = jax.value_and_grad(loss_fn)(params)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, l

    hist, t0 = [], time.time()
    for _ in range(steps):
        params, state, l = step(params, state)
        hist.append(float(l))
    print(f"{label}{steps} steps in {time.time() - t0:.0f} s, loss {hist[0]:.4f} -> {hist[-1]:.4f}")
    return params, np.array(hist), state


# ---------- reporting ----------

def report(theta, log_kappa, true=None):
    g, A0 = unpack(theta)
    mm = lambda v: np.array2string(np.asarray(v) * 1e3, precision=2, max_line_width=200)
    print(f"kappa={float(jnp.exp(log_kappa)):.3e}")
    print(f"  p_d (mm): {mm(g['p_d'])}\n  p_o (mm): {mm(g['p_o'])}")
    print(f"  sigma (1/s): {np.array2string(np.asarray(g['sig']), precision=3, max_line_width=200)}")
    print(f"  A_0 (mm), rows keys, cols {DYNS}:\n{mm(A0)}")
    print("  A_0 ratios to f:\n", np.round(np.asarray(A0[:, :J_FIT] / A0[:, J_FIT:J_FIT + 1]), 3))
    if true is not None:
        gt, A0t = unpack(true)
        print("  " + "  ".join(f"{k}: max rel err {float(jnp.max(jnp.abs(g[k] / gt[k] - 1))):.2%}"
                               for k in ("p_d", "p_o", "sig")))
        print(f"  A_0: max rel err {float(jnp.max(jnp.abs(A0 / A0t - 1))):.2%}")
    return g, A0


def residuals(theta, log_kappa, keys, Ht, eta, mask):
    d = jnp.where(mask, jnp.log(Ht + eta) - jnp.logaddexp(log_kappa + predict(theta, keys), jnp.log(eta)), 0.0)
    rms = lambda ax: np.asarray(jnp.sqrt(masked_mean(d**2, mask, ax)))
    print("rms log residual per harmonic:", np.round(rms((0, 1, 3)), 3))
    print("per key:", np.round(rms((1, 2, 3)), 3), " per dynamic:", np.round(rms((0, 2, 3)), 3))


def plot_fit(theta, log_kappa, keys, Ht, eta, mask, i, j):
    """Measured vs fitted harmonic magnitudes over time for one sample."""
    pred = np.asarray(jnp.logaddexp(log_kappa + predict(theta, keys), jnp.log(eta)))[i, j]
    T, starts = frame_times(keys["f0s"][i], keys["tfit"][i])
    plt.figure(figsize=(8, 4))
    for n in range(N_HARM):
        db = 20 * np.log10(Ht[i, j, n] + eta[i, j, n])
        plt.plot(starts, db, color=f"C{n}", label=f"H{n+1}")
        plt.plot(starts, 20 * pred[n] / np.log(10), "--", color=f"C{n}")
        plt.plot(starts[~mask[i, j, n]], db[~mask[i, j, n]], ".", color="0.6")
    plt.title(f"{keys['notes'][i]}-{DYNS[j]}: target (solid), fit (dashed), masked (grey)")
    plt.xlabel("t after t_0 (s)"); plt.ylabel("dB"); plt.legend(ncol=5)
    plt.tight_layout(); plt.show()


def plot_params(hists, theta, keys, true=None):
    g, A0 = unpack(theta)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].semilogy(np.concatenate(hists)); ax[0].set_xlabel("step (A | B | C)"); ax[0].set_ylabel("loss")
    for j in range(N_DYN):
        ax[1].plot(np.asarray(A0[:, j]) * 1e3, "x-", color=f"C{j}", label=DYNS[j])
    ax[2].plot(np.asarray(g["sig"]), "x-", color="k")
    if true is not None:
        gt, A0t = unpack(true)
        for j in range(N_DYN):
            ax[1].plot(np.asarray(A0t[:, j]) * 1e3, "o", color=f"C{j}", mfc="none")
        ax[2].plot(np.asarray(gt["sig"]), "o", color="k", mfc="none")
    ax[1].set_ylabel("A_0 (mm)" + ("  (o = true)" if true is not None else "")); ax[1].legend()
    ax[2].set_ylabel("sigma_0 (1/s)")
    for a_ in ax[1:]:
        a_.set_xticks(range(len(keys["notes"]))); a_.set_xticklabels(keys["notes"], rotation=60)
    plt.tight_layout(); plt.show()


# ---------- synthetic evaluation ----------
# Waves rendered at 48 kHz with known parameters, white noise and one inharmonic partial
# at 7.1 f0 (the Rhodes' first) decaying INHARM_SYN[2] times faster than the fundamental;
# then exactly the target pipeline (onset, framing, floor, mask).

FS_SYN, PRE_SYN, NOISE_SYN = 48000.0, 5e-3, 1e-5      # ~60 dB SNR like the recordings
INHARM_SYN = (7.1, 0.05, 2.0)                    # ratio to f0, amplitude / A_0, decay relative to sigma_0
KAPPA_SYN  = 2e-5


def theta_true(keys):
    """p_o and p_d vary over the keyboard so the per-key fit is exercised; sigma is the
    step-1 value so the frame span is consistent with the rendered decay."""
    Mk = len(keys["f0s"])
    return dict(pd_raw=raw_pd(np.linspace(2.4e-3, 1.8e-3, Mk)),
                po_raw=raw_po(np.linspace(0.3e-3, 1.5e-3, Mk)),
                log_sig=jnp.log(jnp.asarray(keys["sig1"])),
                A0_raw=raw_a0(1.0e-3 * VELS[None, :] * np.ones((Mk, 1))))


def theta_ref(keys):
    """The truth as the fit should recover it: A_0 refers to t_0 = onset + T_START."""
    th = theta_true(keys)
    _, A0 = unpack(th)
    return dict(th, A0_raw=raw_a0(A0 * jnp.exp(-jnp.exp(th["log_sig"])[:, None] * T_START)))


def synth_waves(theta, kappa, keys, key):
    g, A0 = unpack(theta)
    t = jnp.arange(int((T_FIT + 0.1) * FS_SYN)) / FS_SYN

    @jax.jit
    def one(A0, sig, f0, p_o, p_d, key):
        r, rel, s2 = INHARM_SYN
        A = jnp.array([A0, rel * A0]); s = jnp.array([sig, s2 * sig]); f = jnp.array([f0, r * f0])
        x = eps_free(t, A, s, f, p_o, p_d, TABLE) * kappa
        x = jnp.concatenate([jnp.zeros(int(PRE_SYN * FS_SYN)), x])
        return x + NOISE_SYN * jax.random.normal(key, x.shape)

    waves = {}
    for i, f0 in enumerate(keys["f0s"]):
        for j in range(N_DYN):
            key, sub = jax.random.split(key)
            waves[(i, j)] = (np.asarray(one(A0[i, j], g["sig"][i], f0, g["p_o"][i], g["p_d"][i], sub)), FS_SYN)
    return waves


# %% [markdown]
# ## Initial values: multi-start with successive elimination
#
# Not the optimiser -- an initial-value finder for the geometry of every key.
# The log-magnitude loss is rugged in $p_o$: wherever a harmonic of the model
# passes through zero as a function of the operating point, that harmonic's
# residual spikes, and these walls separate basins that gradient descent
# cannot cross.  $p_o$, $p_d$ and $\sigma_0$ are per key, so one cell per key
# (dynamic `FIT_DYN`) decides the basin.
#
# Every key starts from `PO_STARTS` x `A0_STARTS` starts ($p_d$ = `PD_INIT`,
# $\sigma_0$ from step 1, held).  All starts of all keys are refined in
# parallel with Adam on the gain-free per-key loss, evaluated on `K_INIT` of
# the frames with `L_INIT` samples each (a coarse view is enough to rank
# basins); after each round only the best `KEEP` of a key's starts survive.
# The result -- the surviving candidates, their Adam state and scores -- is
# cached; the best candidate of every key starts the fit (below).
#
#
# | Name | Value | Meaning |
# | --- | --- | --- |
# | `PD_INIT` | 2 mm | $p_d$ of every start |
# | `PO_STARTS`, `A0_STARTS` | 20 x 3 | starts in $p_o$ (0.2 mm apart, closer than any basin) and $A_0$ |
# | `ROUND_STEPS`, `KEEP` | (20, 20, 40), 1/3 | Adam steps per round; fraction of a key's starts kept after each round but the last: 60 -> 20 -> 7 |
# | `K_INIT`, `L_INIT` | 4, 768 | frames and samples per frame of the reduced evaluation |


# %%
PD_INIT      = 2.0e-3
PO_STARTS    = np.linspace(0.15e-3, PO_MAX - 0.15e-3, 20)
A0_STARTS    = np.array([0.1e-3, 0.5e-3, 2.5e-3])
ROUND_STEPS  = (20, 20, 40)
KEEP         = 1 / 3
K_INIT, L_INIT = 4, 768


def init_path(tag, stride=KEY_STRIDE):
    return f"{CACHE}/init_{tag}_{stride}.pkl"


def scorer(keys, Ht, eta, mask):
    """Gain-free per-key loss of the FIT_DYN cell on the reduced evaluation, for a batch of
    starts: q (S, Mk) arrays -> scores (S, Mk)."""
    kidx = np.round(np.linspace(0, K_FRAMES - 1, K_INIT)).astype(int)
    sl = np.s_[:, J_FIT:J_FIT + 1, :, kidx]
    logHt, eta_r, mask_r = jnp.log(Ht + eta)[sl], eta[sl], mask[sl]
    log_sig = jnp.log(jnp.asarray(keys["sig1"]))

    def scores(q):
        one = lambda po, pd, a0: loss_per_key(dict(po_raw=po, pd_raw=pd, log_sig=log_sig, A0_raw=a0[:, None]),
                                              keys, logHt, eta_r, mask_r, kidx, L_INIT)
        return jax.vmap(one)(q["po_raw"], q["pd_raw"], q["A0_raw"])
    return scores


def multi_start(keys, Ht, eta, mask):
    """Returns the cache dict: candidates q (S, Mk), their Adam state, scores (S, Mk), and
    the winner per key (index 0 after the last round's sort)."""
    Mk = len(keys["f0s"])
    scores = scorer(keys, Ht, eta, mask)
    po, a0 = np.meshgrid(PO_STARTS, A0_STARTS, indexing="ij")
    S = po.size
    q = dict(po_raw=raw_po(np.repeat(po.reshape(S, 1), Mk, 1)),
             pd_raw=raw_pd(np.full((S, Mk), PD_INIT)),
             A0_raw=raw_a0(np.repeat(a0.reshape(S, 1), Mk, 1)))
    state, kk = None, np.arange(Mk)
    for r, steps in enumerate(ROUND_STEPS):
        q, _, state = fit(q, lambda q: scores(q).sum(), steps, label=f"round {r + 1}, {S} starts x {Mk} keys: ", state=state)
        L = np.asarray(jax.jit(scores)(q))
        S = S if r == len(ROUND_STEPS) - 1 else max(1, int(np.ceil(S * KEEP)))
        top = np.argsort(L, axis=0)[:S]                       # (S, Mk): each key keeps its own best starts
        q = {k: v[top, kk] for k, v in q.items()}
        state = jax.tree_util.tree_map(lambda v: v[top, kk] if getattr(v, "ndim", 0) == 2 else v, state)
        L = L[top, kk]
        print("  best per key:", np.round(L[0], 4), " runner-up:", np.round(L[1], 4))
    return dict(q={k: np.asarray(v) for k, v in q.items()}, state=jax.device_get(state), scores=L,
                notes=keys["notes"], rounds=ROUND_STEPS)


def winner(cache):
    """Starting theta of the fit: the best candidate of every key (A0_raw (Mk, 1))."""
    q = cache["q"]
    return dict(po_raw=jnp.asarray(q["po_raw"][0]), pd_raw=jnp.asarray(q["pd_raw"][0]),
                A0_raw=jnp.asarray(q["A0_raw"][0])[:, None])


# %% [markdown]
# ## The fit: stages A, B, C

# %%
STEPS_A, STEPS_B, STEPS_C = 300, 100, 300
A0_SCAN = np.geomspace(0.01e-3, 0.95 * DISP_MAX, 32)     # stage B start: per-cell scan of A_0


def stage_A(theta0, keys, Ht, eta, mask):
    """p_o, p_d, sigma_0, A_0(f) from the f cells."""
    sl = np.s_[:, J_FIT:J_FIT + 1]
    logHt, eta_f, mask_f = jnp.log(Ht + eta)[sl], eta[sl], mask[sl]
    theta, hist, _ = fit(theta0, lambda th: loss(th, keys, logHt, eta_f, mask_f)[0], STEPS_A, label="A: ")
    return theta, loss(theta, keys, logHt, eta_f, mask_f)[1], hist


def stage_B(theta_A, log_kappa, keys, Ht, eta, mask):
    """A_0 of the other dynamics, one parameter per cell, geometry / sigma / kappa fixed.
    The loss of a cell is rugged in A_0 (harmonics have nulls as a function of amplitude),
    so each cell starts from the best of a log-spaced scan A0_SCAN, then Adam."""
    others = [j for j in range(N_DYN) if j != J_FIT]
    logHt, eta_o, mask_o = jnp.log(Ht + eta)[:, others], eta[:, others], mask[:, others]
    fixed = {k: v for k, v in theta_A.items() if k != "A0_raw"}
    Mk = len(keys["f0s"])

    def cell_loss(A0_raw):                                    # (Mk, n_others) per-cell losses
        pred = jnp.logaddexp(log_kappa + predict(dict(fixed, A0_raw=A0_raw), keys), jnp.log(eta_o))
        return masked_mean((logHt - pred) ** 2, mask_o, axis=(2, 3))

    # one scan value at a time (lax.map, not vmap): all 32 at once need ~11 GB
    scan = jax.jit(lambda s: jax.lax.map(lambda a0: cell_loss(jnp.full((Mk, len(others)), raw_a0(a0))), s))(A0_SCAN)
    A0 = A0_SCAN[np.asarray(scan).argmin(0)]                  # (Mk, n_others)
    loss_B = lambda q: loss(dict(fixed, **q), keys, logHt, eta_o, mask_o, log_kappa)[0]
    q, hist, _ = fit(dict(A0_raw=raw_a0(A0)), loss_B, STEPS_B, label="B: ")
    A0_raw = jnp.concatenate([q["A0_raw"][:, :J_FIT], theta_A["A0_raw"], q["A0_raw"][:, J_FIT:]], axis=1)
    return dict(fixed, A0_raw=A0_raw), hist


def stage_C(theta, keys, Ht, eta, mask):
    """All parameters on all cells."""
    logHt = jnp.log(Ht + eta)
    theta, hist, _ = fit(theta, lambda th: loss(th, keys, logHt, eta, mask)[0], STEPS_C, label="C: ")
    return theta, loss(theta, keys, logHt, eta, mask)[1], hist


def fit_all(theta0, keys, Ht, eta, mask):
    t0 = time.time()
    theta_A, lk_A, hA = stage_A(theta0, keys, Ht, eta, mask)
    theta_B, hB       = stage_B(theta_A, lk_A, keys, Ht, eta, mask)
    theta_C, lk_C, hC = stage_C(theta_B, keys, Ht, eta, mask)
    print(f"total {time.time() - t0:.0f} s")
    return theta_C, lk_C, (hA, hB, hC)


# %%
# Initial values: multi-start on the target features, cached for the fit.
# From the terminal: python step2_pickup.py [real|syn] [stride].  Run cell by cell, the
# command line is the kernel's, not ours: set TAG and STRIDE here instead.
if __name__ == "__main__":
    args   = [] if "ipykernel" in sys.argv[0] else sys.argv[1:]
    TAG    = args[0] if len(args) > 0 else "syn"
    STRIDE = int(args[1]) if len(args) > 1 else KEY_STRIDE
    keys = load_keys(STRIDE)
    Ht, eta, mask = features(TAG, keys, STRIDE)
    report_mask(mask)
    t0 = time.time()
    cache = multi_start(keys, Ht, eta, mask)
    print(f"multi-start in {time.time() - t0:.0f} s")
    os.makedirs(CACHE, exist_ok=True)
    with open(init_path(TAG, STRIDE), "wb") as f:
        pickle.dump(cache, f)
    print(f"-> {init_path(TAG, STRIDE)}")
    g, A0 = unpack(dict(winner(cache), log_sig=jnp.log(jnp.asarray(keys["sig1"]))))
    print("winner p_o (mm):", np.round(np.asarray(g["p_o"]) * 1e3, 2), " p_d (mm):", np.round(np.asarray(g["p_d"]) * 1e3, 2),
          " A_0(f) (mm):", np.round(np.asarray(A0[:, 0]) * 1e3, 2))

# %%
# The fit, from the best start of every key.
if __name__ == "__main__":
    with open(init_path(TAG, STRIDE), "rb") as f:
        cache = pickle.load(f)
    assert (cache["notes"] == keys["notes"]).all(), "init cache is for other keys"
    theta0 = dict(winner(cache), log_sig=jnp.log(jnp.asarray(keys["sig1"])))

    true = theta_ref(keys) if TAG == "syn" else None
    if true is not None:
        print(f"loss at truth: {float(loss(true, keys, jnp.log(Ht + eta), eta, mask)[0]):.4f}")

    theta, lk, hists = fit_all(theta0, keys, Ht, eta, mask)
    print("fit:"); g, A0 = report(theta, lk, true)
    residuals(theta, lk, keys, Ht, eta, mask)

# %%
if __name__ == "__main__":
    plot_params(hists, theta, keys, true)
    plot_fit(theta, lk, keys, Ht, eta, mask, len(keys["f0s"]) // 2, J_FIT)

# %%
# Hand-over to step 3: A_0 (Mk, N_DYN) and sigma_0 (Mk,) with the geometry (full keyboard only).
if __name__ == "__main__" and TAG == "real" and STRIDE == 1:
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    np.savez(RESULTS, notes=keys["notes"], f0=keys["f0s"], A0=np.asarray(A0),
             sigma=np.asarray(g["sig"]), p_d=np.asarray(g["p_d"]), p_o=np.asarray(g["p_o"]),
             kappa=float(jnp.exp(lk)), tfit=keys["tfit"], ok=mask[:, :, 0].sum(-1) >= 3, flag=keys["flag"])
    print(f"-> {RESULTS}")
