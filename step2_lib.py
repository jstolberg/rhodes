"""Step 2 of fitting.py (pickup and fundamental): shared definitions.

Model: one free mode x(t) = p_o + A_0 e^{-sigma_0 t} sin 2 pi f_0 t seen through the
pickup, eps = -kappa dPsi(x, p_d)/dt, with the surface fixed from photos and Psi tabulated
once.  Target and model are projected onto the fundamental and its first four harmonics in
Hann frames; the loss is the mean squared log-magnitude difference over the cells above the
noise floor, with kappa the closed-form level match.

p_o, p_d, sigma_0 are per key, A_0 per key and dynamic.  The decay of step 1
(fundamentals.npz) sets the frame span per key and the starting sigma_0.

Used by step2_init.py (initial-value finder) and step2_fft.py (the fit).

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
| `T_FIT`, `DECAY_SPAN` | 3 s, 4 | frame span per key min(T_FIT, 4 / sigma_1) after t_0, sigma_1 from step 1 |
| `L_FRAME` | 1024 | synthetic samples per frame (16 per period, Nyquist 8 f_0) |
| `NOISE_MARGIN` | 3 | keep cells with |H_t| > margin x noise (9.5 dB; noise is Rayleigh, 3 passes ~3% of noise cells) |
| `R_FIX`, `A_FIX`, `M_FIX` | 4 mm, 1.3 mm, 1 | pickup geometry from photos (fixed) |
| `SURF_N` | 64 | surface grid |
| `DISP_MAX`, `N_TABLE` | 6 mm, 4096 | A_0 <= DISP_MAX (bass tines sweep fully past the pickup); 3 um table |
| `PO_MAX` | 3/2 r = 6 mm | p_o bounded to (0, PO_MAX) via sigmoid (also the table extent) |
| `PD_RANGE`, `N_PD` | 0.5-3.5 mm, 96 | p_d bounded to this range via sigmoid, tabulated on N_PD rows |
| `LR` | 0.05 | Adam learning rate (all stages) |
| `KEY_STRIDE` | 1 | fit every k-th key (1 = all 73, 24 = E0, E2, E4, E6); runtime scales with it |
| `CACHE` | cache/ | target features and init results |
"""
from __future__ import annotations

import os
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
FUNDAMENTALS = "fundamentals.npz"
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
T_FIT        = 3.0
DECAY_SPAN   = 4.0
L_FRAME      = 1024
NOISE_MARGIN = 3.0

R_FIX, A_FIX, M_FIX = 4e-3, 1.3e-3, 1.0
SURF_N       = 64
DISP_MAX, N_TABLE = 6.0e-3, 4096
PO_MAX       = 3.0 / 2.0 * R_FIX
PD_RANGE, N_PD = (0.5e-3, 3.5e-3), 96
LR           = 0.05
KEY_STRIDE   = 1       # every k-th key (24 -> E0, E2, E4, E6; library names are one octave low)

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
    keys = dict(notes=notes[sel], f0s=f0_all[sel], sig1=sig_all[sel], tfit=frame_span(f0_all[sel], sig_all[sel]))
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

    @jax.checkpoint                                           # recompute in backward: O(1) memory per cell
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
