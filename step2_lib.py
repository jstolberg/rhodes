"""Step 2 of fitting.py (pickup and fundamental): shared definitions.

What step 2 does
----------------
The model of this step is one freely decaying oscillation of the tine tip,

    x(t) = p_o + A_0 e^{-sigma_0 t} sin(2 pi f_0 t),

seen through the pickup: the voltage is eps = -kappa dPsi(x, p_d)/dt.  The pickup is
nonlinear, so eps contains the fundamental f_0 *and* its harmonics 2 f_0, 3 f_0, ...
How strong the harmonics are, and how that changes while the tone decays, depends on
where the tine sits in front of the pickup (p_o, p_d).  That is what step 2 fits.

How recording and model are compared
------------------------------------
Both are cut into K_FRAMES short frames (N_PERIODS periods of f_0 each, Hann window).
In every frame we measure the magnitude at f_0 ... 5 f_0 ("projection onto the
harmonics").  That gives a small table |H[harmonic, frame]| per recording.  The loss is
the mean squared difference of log|H| between recording and model, over the cells that
stand clearly above the noise.  kappa (overall level) is not fitted but computed
directly as the mean log difference.

What is fitted
--------------
    p_o, p_d, sigma_0   per key
    A_0                 per key and dynamic (p, mp, mf, f)
    kappa               one number, computed in closed form

The pickup surface is fixed from photos, so Psi(x, z) is computed once as a lookup table
(TABLE) and p_o, p_d only enter through the lookup.

Used by step2_init.py (start values), step2_fft.py (the fit), and for their shared
helpers (load_wave, onset, hann, fit, TABLE, ...) by steps 3, 5 and 6.
"""
from __future__ import annotations

import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.io.wavfile as wavfile

jax.config.update("jax_enable_x64", True)

from model import z_trapz, make_surface, drop_masked, psi_table, psi_lookup


# =====================================================================================
# 1. Settings
# =====================================================================================

# Files
SAMPLES      = "Samples"            # folder with the recordings {note}-{dyn}.wav
FUNDAMENTALS = "fundamentals.npz"   # result of step 1 (f_0 and decay per recording)
CACHE        = "cache"              # target features and start values are cached here

# Dynamics
DYNS         = ["p", "mp", "mf", "f"]
FIT_DYN      = "f"                  # the dynamic the geometry is fitted on first
VELS         = np.array([0.25, 0.5, 0.75, 1.0])   # assumed velocities of p, mp, mf, f
N_DYN, J_FIT = len(DYNS), DYNS.index(FIT_DYN)

# Frames
N_HARM       = 5                    # compared lines: f_0 and 4 harmonics
HARM         = np.arange(1, N_HARM + 1)            # 1, 2, 3, 4, 5 (multiples of f_0)
T_START      = 10e-3                # first frame starts 10 ms after the onset (hammer gone)
ONSET_FRAC   = 0.05                 # onset = first sample above 5 % of the peak
N_PERIODS    = 64                   # frame length in periods of f_0; long enough that the
                                    # 7.1 f_0 mode and its mixing products at 5.1, 6.1 f_0
                                    # stay > 6 frequency bins away from a harmonic
K_FRAMES     = 12                   # frames per recording, spread evenly over the span
T_FIT        = 3.0                  # frames cover at most 3 s ...
DECAY_SPAN   = 4.0                  # ... or 4 decay times 4 / sigma_1, whichever is shorter
L_FRAME      = 1024                 # model samples per frame (16 per period)
NOISE_MARGIN = 3.0                  # a cell counts if |H| > 3 x noise (9.5 dB)

# Pickup (fixed) and parameter ranges
R_FIX, A_FIX, M_FIX = 4e-3, 1.3e-3, 1.0            # surface radius, plateau half width, slope
SURF_N       = 64                   # surface grid points per side
DISP_MAX     = 6.0e-3               # largest amplitude A_0 (bass tines swing fully past the pickup)
N_TABLE      = 4096                 # table points over the displacement range (~3 um apart)
PO_MAX       = 3.0 / 2.0 * R_FIX    # p_o is kept in (0, PO_MAX)
PD_RANGE     = (0.5e-3, 3.5e-3)     # p_d is kept in this range ...
N_PD         = 96                   # ... and tabulated on this many rows

# Optimiser
LR           = 0.05                 # Adam learning rate
KEY_STRIDE   = 1                    # fit every k-th key (24 -> E0, E2, E4, E6 only; faster)


# =====================================================================================
# 2. Keys and recordings
# =====================================================================================

def load_keys(stride=KEY_STRIDE):
    """Keys to fit, with f_0 and the decay sigma_1 of step 1.

    Step 1 stores one row per recording.  Here f_0 and sigma_1 are the medians over the
    four dynamics; keys whose decay step 1 could not measure get it by log-linear
    interpolation between their neighbours.  Returns a dict with notes, f0s, sig1 and
    tfit (the time span the frames cover), for every stride-th key.
    """
    data   = np.load(FUNDAMENTALS, allow_pickle=True)
    notes  = data["notes"]
    tab    = data["table"]                              # (73 keys, 4 dynamics)
    f0_all = np.nanmedian(tab["f"], axis=1)

    # decay: median over the dynamics where step 1 succeeded, gaps interpolated
    lam     = np.where(tab["ok"], tab["lam"], np.nan)
    sig_all = np.array([np.nanmedian(r) if np.isfinite(r).any() else np.nan for r in lam])
    have    = np.isfinite(sig_all)
    sig_all = np.exp(np.interp(np.arange(len(notes)), np.where(have)[0], np.log(sig_all[have])))

    sel  = np.arange(0, len(notes), stride)
    keys = dict(notes=notes[sel], f0s=f0_all[sel], sig1=sig_all[sel],
                tfit=frame_span(f0_all[sel], sig_all[sel]))
    print(f"{len(sel)} keys: {keys['notes'][0]} ({keys['f0s'][0]:.1f} Hz) ... "
          f"{keys['notes'][-1]} ({keys['f0s'][-1]:.1f} Hz)")
    print("step-1 sigma (1/s):", np.round(keys["sig1"], 3), " frame span (s):", np.round(keys["tfit"], 2))
    return keys


def load_wave(note, dyn):
    """One recording as float64 mono, and its sample rate."""
    fs, x = wavfile.read(f"{SAMPLES}/{note}-{dyn}.wav")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)                              # stereo -> mono
    return x, float(fs)


def onset(x):
    """Index of the first sample above ONSET_FRAC of the peak."""
    return int(np.argmax(np.abs(x) > ONSET_FRAC * np.abs(x).max()))


# =====================================================================================
# 3. Frames (the same for recording and model)
# =====================================================================================

def frame_span(f0s, sig1):
    """Time the frames of a key cover: DECAY_SPAN decay times, but at least two frame
    lengths and at most T_FIT."""
    return np.clip(DECAY_SPAN / sig1, 2 * N_PERIODS / f0s, T_FIT)


def frame_times(f0, t_fit):
    """Frame length T and the K_FRAMES frame starts (s after t_0), evenly over t_fit."""
    T   = N_PERIODS / f0
    hop = (t_fit - T) / (K_FRAMES - 1)
    return T, hop * np.arange(K_FRAMES)


def hann(n):
    """Hann window of n samples, sampled at the sample centres."""
    return 0.5 * (1.0 - np.cos(2 * np.pi * (np.arange(n) + 0.5) / n))


# =====================================================================================
# 4. Target: the harmonic magnitudes of the recordings (numpy, computed once, cached)
# =====================================================================================

def project_target(x, fs, f0, t_fit):
    """|H_t| (N_HARM, K_FRAMES) of one recording, and its noise floor.

    Each frame is windowed and multiplied with e^{-2 pi i n f_0 t} for n = 1..5: that is
    the Fourier coefficient exactly at the harmonic.  The noise floor of a frame is the
    mean magnitude halfway between the harmonics, at (n + 1/2) f_0, where the signal has
    no energy; it is the same for all harmonics of that frame.
    """
    T, starts = frame_times(f0, t_fit)
    Lf  = int(round(T * fs))                            # frame length in samples
    w   = hann(Lf)
    n   = np.arange(Lf) / fs                            # time within a frame
    E_h = np.exp(-2j * np.pi * np.outer(HARM * f0, n))                        # at n f_0
    E_m = np.exp(-2j * np.pi * np.outer((np.arange(N_HARM) + 0.5) * f0, n))   # in between

    i0 = onset(x) + int(round(T_START * fs))            # t_0: start of the first frame
    H, mid = np.zeros((N_HARM, K_FRAMES)), np.zeros((N_HARM, K_FRAMES))
    for k, s in enumerate(starts):
        a = i0 + int(round(s * fs))
        seg = w * x[a:a + Lf]
        H[:, k]   = np.abs(E_h @ seg) / fs
        mid[:, k] = np.abs(E_m @ seg) / fs
    return H, np.repeat(mid.mean(axis=0, keepdims=True), N_HARM, axis=0)


def build_target(keys, waves):
    """|H_t|, noise floor eta and mask for every key and dynamic, all (Mk, N_DYN, N_HARM, K).
    waves[(i, j)] = (x, fs) of key i, dynamic j.  mask: the cell stands above the noise."""
    shape = (len(keys["f0s"]), N_DYN, N_HARM, K_FRAMES)
    H, eta = np.zeros(shape), np.zeros(shape)
    for i, f0 in enumerate(keys["f0s"]):
        for j in range(N_DYN):
            x, fs = waves[(i, j)]
            H[i, j], eta[i, j] = project_target(x, fs, f0, keys["tfit"][i])
    return H, eta, H > NOISE_MARGIN * eta


def features(tag, keys, stride=KEY_STRIDE):
    """build_target of the real recordings (tag "real") or of synthetic ones (tag "syn",
    see section 7), cached in CACHE/features_{tag}_{stride}.npz."""
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
    """Print how many cells stand above the noise."""
    kept = mask.mean(axis=(0, 1, 3))
    print("frames kept per harmonic:", " ".join(f"H{n}:{k:.0%}" for n, k in zip(HARM, kept)))
    print(f"kept frames per key, dyn {FIT_DYN}, H1..H{N_HARM}:", mask[:, J_FIT].sum(-1).tolist())


# =====================================================================================
# 5. Model: the same magnitudes, predicted (jax, differentiable)
# =====================================================================================

# Psi(x, z) of the fixed surface, tabulated once: x covers displacement plus p_o range,
# z the p_d range.
PTS, W = drop_masked(*make_surface(partial(z_trapz, a=A_FIX, m=M_FIX), N=SURF_N, r_max=R_FIX))
TABLE  = psi_table(dict(p_d=PD_RANGE[0]), PTS, W, DISP_MAX, PO_MAX,
                   pd_range=PD_RANGE, n=N_TABLE, n_pd=N_PD)


def eps_free(t, A, sig, f, p_o, p_d, table):
    """Pickup voltage -dPsi/dt of freely decaying modes (arrays A, sig, f) at times t
    (any shape).  dPsi/dt is taken exactly by forward-mode differentiation (jvp)."""
    def psi(tt):
        x = p_o + jnp.sum(A * jnp.exp(-sig * tt) * jnp.sin(2 * jnp.pi * f * tt))
        return psi_lookup(jnp.array([x, 0.0, p_d]), table)
    d = jax.vmap(lambda tt: jax.jvp(psi, (tt,), (1.0,))[1])(t.ravel())
    return -d.reshape(t.shape)


def project_synth(A0, sig, f0, t_fit, p_o, p_d, table, kidx=None, L=L_FRAME):
    """|H_s| (N_HARM, K) of the model: like project_target, but the voltage is computed
    only at L points inside each frame (kidx: only these frames)."""
    T, starts = frame_times(f0, t_fit)
    ks  = jnp.arange(K_FRAMES) if kidx is None else jnp.asarray(kidx)
    rel = (jnp.arange(L) + 0.5) / L                                          # 0..1 in a frame
    tt  = jnp.asarray(starts)[ks][:, None] + T * rel[None, :]                # (K, L) times
    e   = eps_free(tt, A0[None], sig[None], f0[None], p_o, p_d, table)      # (K, L)
    E   = jnp.exp(-2j * jnp.pi * jnp.outer(HARM * f0, T * rel))             # (N, L)
    H   = jnp.einsum("nl,kl->nk", E, jnp.asarray(hann(L)) * e) * (T / L)
    return jnp.abs(H)


# --- parameters ---
# The optimiser works on unbounded "raw" numbers.  unpack maps them into the allowed
# ranges with a sigmoid (p_d in PD_RANGE, p_o in (0, PO_MAX), A_0 in (0, DISP_MAX)) and
# sigma through exp (always > 0); raw_* go the other way.
#   theta = dict(po_raw (Mk,), pd_raw (Mk,), log_sig (Mk,), A0_raw (Mk, n_dyn))

def unpack(theta):
    """raw theta -> (dict(p_d, p_o, sig) per key, A_0 per key and dynamic)."""
    lo, hi = PD_RANGE
    g = dict(p_d=lo + (hi - lo) * jax.nn.sigmoid(theta["pd_raw"]),
             p_o=PO_MAX * jax.nn.sigmoid(theta["po_raw"]),
             sig=jnp.exp(theta["log_sig"]))
    return g, DISP_MAX * jax.nn.sigmoid(theta["A0_raw"])


def raw_po(po):
    return jax.scipy.special.logit(jnp.asarray(po) / PO_MAX)


def raw_pd(pd):
    return jax.scipy.special.logit((jnp.asarray(pd) - PD_RANGE[0]) / (PD_RANGE[1] - PD_RANGE[0]))


def raw_a0(a0):
    return jax.scipy.special.logit(jnp.asarray(a0) / DISP_MAX)


def predict(theta, keys, kidx=None, L=L_FRAME):
    """log|H_s| for kappa = 1 of every key and dynamic: (Mk, n_dyn, N_HARM, K)."""
    g, A0 = unpack(theta)

    @jax.checkpoint                         # recompute in the backward pass: saves memory
    def cell(A0, sig, f0, t_fit, p_o, p_d):
        return jnp.log(project_synth(A0, sig, f0, t_fit, p_o, p_d, TABLE, kidx, L))

    over_dyn = jax.vmap(cell, in_axes=(0, None, None, None, None, None))    # dynamics share geometry
    return jax.vmap(over_dyn)(A0, g["sig"], jnp.asarray(keys["f0s"]), jnp.asarray(keys["tfit"]),
                              g["p_o"], g["p_d"])


# =====================================================================================
# 6. Loss and optimiser
# =====================================================================================

def masked_mean(x, mask, axis=None):
    """Mean of x over the cells where mask is True."""
    return jnp.sum(jnp.where(mask, x, 0.0), axis=axis) / jnp.maximum(mask.sum(axis=axis), 1)


def loss(theta, keys, logHt, eta, mask, log_kappa=None):
    """Mean squared log difference over the kept cells.  Returns (loss, log_kappa).

    log_kappa (overall level) is the mean log difference unless given.  The model
    magnitude is combined with the noise floor (logaddexp = log(kappa |H_s| + eta)), just
    as the target is log(|H_t| + eta), so a model line below the noise costs nothing.
    """
    logHs = predict(theta, keys)
    if log_kappa is None:
        log_kappa = masked_mean(logHt - logHs, mask)
    pred = jnp.logaddexp(log_kappa + logHs, jnp.log(eta))
    return masked_mean((logHt - pred) ** 2, mask), log_kappa


def loss_per_key(theta, keys, logHt, eta, mask, kidx=None, L=L_FRAME):
    """Loss per key (Mk,), with the level fitted per key: only the harmonic ratios and the
    decay shape count.  Used to compare start values."""
    logHs = predict(theta, keys, kidx, L)
    ax    = (1, 2, 3)
    lk    = masked_mean(logHt - logHs, mask, ax)
    pred  = jnp.logaddexp(lk[:, None, None, None] + logHs, jnp.log(eta))
    return masked_mean((logHt - pred) ** 2, mask, ax)


def fit(params, loss_fn, steps, lr=LR, label="", state=None):
    """Adam on params for loss_fn(params) -> scalar.  Returns params, loss history and the
    optimiser state (pass it back in to continue)."""
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


# =====================================================================================
# 7. Synthetic test data (TAG = "syn")
# =====================================================================================
# Recordings rendered by the model itself with known parameters, plus white noise and
# one inharmonic mode at 7.1 f_0 (the Rhodes' first) that the fit has to ignore.  They
# go through exactly the same target pipeline (onset, frames, noise floor, mask).

FS_SYN, PRE_SYN, NOISE_SYN = 48000.0, 5e-3, 1e-5    # sample rate, silence before, noise (~60 dB SNR)
INHARM_SYN = (7.1, 0.05, 2.0)       # ratio to f_0, amplitude / A_0, decay / sigma_0
KAPPA_SYN  = 2e-5                   # level of the synthetic recordings


def theta_true(keys):
    """The known parameters: p_o and p_d vary over the keyboard, sigma is the step-1
    value, A_0 = 1 mm x the assumed velocity."""
    Mk = len(keys["f0s"])
    return dict(pd_raw=raw_pd(np.linspace(2.4e-3, 1.8e-3, Mk)),
                po_raw=raw_po(np.linspace(0.3e-3, 1.5e-3, Mk)),
                log_sig=jnp.log(jnp.asarray(keys["sig1"])),
                A0_raw=raw_a0(1.0e-3 * VELS[None, :] * np.ones((Mk, 1))))


def theta_ref(keys):
    """The truth as the fit should find it: A_0 is measured at t_0 = onset + T_START, so it
    has already decayed by e^{-sigma T_START}."""
    th = theta_true(keys)
    _, A0 = unpack(th)
    return dict(th, A0_raw=raw_a0(A0 * jnp.exp(-jnp.exp(th["log_sig"])[:, None] * T_START)))


def synth_waves(theta, kappa, keys, key):
    """waves[(i, j)] = (x, FS_SYN): synthetic recording of key i, dynamic j."""
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
