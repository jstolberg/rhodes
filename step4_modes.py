# %% [markdown]
# # Step 4: Inharmonic modes
#
#     python step4_modes.py      results/step1..3 -> results/step4_modes.npz
#
# ## A. Frequencies and decay ratios of the clearest tine modes
#
# For every key and mode: find the mode in the spectrum, keep it only if it is clear,
# measure how fast it dies, and divide by how fast the fundamental dies.
#
# 1. **Find.** In each of the four recordings: the tallest spectrum peak within
#    $\mu \pm 3\sigma$ times $f_0$ (Gabrielli's ratios), away from the harmonics.
# 2. **Keep only the clear ones.** At least `MIN_AGREE` recordings put the peak at the
#    same frequency (within `AGREE`).
# 3. **Decay.** The line's level over time in short frames, a straight line through the
#    dB values, the slope is the decay. Only from p and mp (at loud dynamics the pickup
#    bends the curve). Both must fall at least `FALL_DB` while still 10 dB over the noise,
#    lie on a straight line (`RESID_DB`) and agree with each other (`DECAY_AGREE`).
# 4. **Ratio.** $\sigma_n / \sigma_0$, with $\sigma_0$ of the same recordings from
#    step 1 (`results/step1_fundamentals.npz`). One number per mode: the median
#    over the keys.
#
# %%
import warnings

import jax.numpy as jnp
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal.windows import blackmanharris

from model import calc_tau, epsilon, hammer2free
from step2_pickup import T_START, TABLE, VELS

RESULTS = "results/step4_modes.npz"

MODES = {"m2": (7.1, 0.3), "m3": (20.4, 0.4), "m4": (39.7, 0.9),    # Gabrielli 2020,
         "m5": (62.9, 1.9), "m6": (93.1, 2.7)}                        # Fig. 20: mu, sigma
DYNS = ("p", "mp", "mf", "f")

T_FFT       = 0.3      # s of recording in the spectrum that finds the peak
HARM_HZ     = 15.0     # no search closer than this to a harmonic (the window's main lobe)
MIN_AGREE   = 3        # recordings that must find the same peak ...
AGREE       = 0.003    # ... within this fraction of its frequency
FRAME_K     = 6.0      # decay frames are FRAME_K / (distance to the nearest harmonic) long
FALL_DB     = 10.0     # the line must fall this far during the fit ...
RESID_DB    = 1.0      # ... with the dB values this close (rms) to a straight line
DECAY_AGREE = 0.25     # p and mp decays within this fraction of each other

fund = np.load("results/step1_fundamentals.npz", allow_pickle=True)
NOTES, FUND = list(fund["notes"]), fund["table"]


def load(note, dyn):
    """Mono recording from 10 ms after the hammer to 0.7 s before the end (fade-out)."""
    fs, x = wavfile.read(f"Samples/{note}-{dyn}.wav")
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.mean(axis=1)
    start = np.argmax(np.abs(x) > 0.1 * np.abs(x).max()) + int(0.01 * fs)
    return x[start:-int(0.7 * fs)], float(fs)


def spectrum(x, fs):
    """Frequencies and dB spectrum of the first T_FFT s."""
    n, nfft = int(T_FFT * fs), 1 << 20
    X = np.fft.rfft(x[:n] * blackmanharris(n), nfft)
    return np.fft.rfftfreq(nfft, 1 / fs), 20 * np.log10(np.abs(X) + 1e-12)


def find_peak(f, db, f0, mu, sigma):
    """Frequency of the tallest local maximum within mu +- 3 sigma times f0, harmonics
    excluded; nan if there is none."""
    window = (np.abs(f - mu * f0) <= 3 * sigma * f0) & (np.abs(f - f0 * np.round(f / f0)) > HARM_HZ)
    is_max = np.r_[False, (db[1:-1] > db[:-2]) & (db[1:-1] > db[2:]), False]
    cand = np.flatnonzero(window & is_max)
    if cand.size == 0:
        return np.nan
    return f[cand[np.argmax(db[cand])]]


def levels(x, fs, f, T):
    """Level (dB) of the line at f in frames of T s, and the frame centres (s)."""
    L = int(T * fs)
    w = blackmanharris(L) * np.exp(-2j * np.pi * f * np.arange(L) / fs)
    starts = np.arange(0, x.size - L, L // 4)
    db = 20 * np.log10(np.abs([x[s:s + L] @ w for s in starts]) + 1e-12)
    return (starts + L / 2) / fs, db


def fit_range(t, db):
    """From the loudest frame until the level is 10 dB over the noise (the level in the
    last second, when the mode is long gone)."""
    noise = np.median(db[t > t[-1] - 1.0])
    i0 = int(np.argmax(db))
    below = np.flatnonzero(db[i0:] < noise + 10)
    return i0, i0 + (int(below[0]) if below.size else db.size - i0)


def decay(x, fs, f, T):
    """Decay (1/s) of the line at f: a straight line through its dB levels over the fit
    range. Returns sigma, how far it fell (dB) and the rms distance from the line (dB)."""
    t, db = levels(x, fs, f, T)
    i0, i1 = fit_range(t, db)
    if i1 - i0 < 4:
        return np.nan, 0.0, np.nan
    slope, icpt = np.polyfit(t[i0:i1], db[i0:i1], 1)
    resid = np.sqrt(np.mean((db[i0:i1] - icpt - slope * t[i0:i1]) ** 2))
    return -slope / 8.686, -slope * (t[i1 - 1] - t[i0]), resid


# %%
# ---------- all keys ----------
rows = []
for i, note in enumerate(NOTES):
    f0 = {d: FUND[i, j]["f"] for j, d in enumerate(DYNS)}
    rec = {d: load(note, d) for d in DYNS}
    spec = {d: spectrum(*rec[d]) for d in DYNS}
    for mode, (mu, sigma) in MODES.items():
        # 1. the peak in every recording
        peak = {d: find_peak(*spec[d], f0[d], mu, sigma) for d in DYNS}
        f_p = peak["p"]
        if not np.isfinite(f_p):
            continue
        # 2. clear: the recordings agree on it
        if sum(abs(peak[d] - f_p) <= AGREE * f_p for d in DYNS) < MIN_AGREE:
            continue
        # 3. decay in p and mp, frames long enough to keep the nearest harmonic out
        dist = abs(f_p - f0["p"] * round(f_p / f0["p"]))
        dec = {d: decay(*rec[d], f_p, FRAME_K / dist) for d in ("p", "mp")}
        clean = all(fell >= FALL_DB and resid <= RESID_DB for _, fell, resid in dec.values())
        s_p, s_mp = dec["p"][0], dec["mp"][0]
        if not clean or abs(s_p - s_mp) > DECAY_AGREE * (s_p + s_mp) / 2:
            continue
        # 4. ratio to the fundamental of the same recordings
        s0 = [FUND[i, j]["lam"] for j in (0, 1) if FUND[i, j]["ok"]]
        if len(s0) < 2:
            continue
        sig_n, sig_0 = (s_p + s_mp) / 2, float(np.mean(s0))
        rows.append(dict(note=note, mode=mode, f=f_p, f0=f0["p"], T=FRAME_K / dist,
                         ratio_f=f_p / f0["p"], sig_n=sig_n, sig_0=sig_0, ratio=sig_n / sig_0))

# %%
# ---------- result ----------
print("decay ratio sigma_n / sigma_0 per mode (median over the clear keys)")
for mode in MODES:
    r = [x["ratio"] for x in rows if x["mode"] == mode]
    print(f"  {mode}: {np.median(r):5.1f}   ({len(r)} keys)" if r else f"  {mode}:     -   (no clear key)")


# %% [markdown]
# ## B. The modes on every key
#
# Part A measures each mode only on the keys where it is clear.  Part B puts the modes on
# **all** keys: every quantity is one number per mode, the median over the clear keys, as
# for the decays in part A.  (No trend over the keyboard: the scatter between neighbouring
# keys hides any.)
#
# - **Frequency**: $f_n = \mu_n f_0$, $\mu_n$ Gabrielli's mean ratio (`MODES`), $f_0$ from
#   step 2.  Not part A's measured ratios: they rest on few keys (m4: 3), and part A searches
#   only within $\mu_n \pm 3 \sigma_n$, so a ratio near the edge of that window (m4) is not
#   certainly the mode.  Gabrielli's ratios are measured on the tine itself (laser vibrometer,
#   no pickup) over more keys.  The measured median is printed for comparison.  Modes above
#   `F_MAX` = 10 kHz are left out (Gabrielli found none).
# - **Decay**: $\sigma_n = r_n \sigma_0$, $r_n$ the median of $\sigma_n / \sigma_0$ with part A's
#   $\sigma_n$ and $\sigma_0$ of step 2, the one the model uses.  (Part A prints the ratio to
#   $\sigma_0$ of step 1.)  Modes without a clear key have no $r_n$ and are left out.
# - **Amplitude**: $A_n = a_n A_0$, one ratio $a_n$ per mode for all keys **and all
#   dynamics**.  $A_n$ and $A_0$ are the free amplitudes of the tine at $t_0$ = onset +
#   `T_START`.  $A_n / A_0$ hardly changes with the dynamic, so the modes simply follow the
#   fundamental's velocity law of step 3.  Measured on the clear keys at every dynamic: the
#   level of the mode in the recording (part A's spectrum of the first `T_FFT` s after $t_0$,
#   at part A's peak) against the model of step 2 (free fundamental $A_0$ through the pickup)
#   with the mode added at a trial amplitude.  A small mode passes the pickup linearly, so one
#   scaling matches the two.  A key's ratio is the mean of $\log(A_n / A_0)$ over its dynamics.
# - **Excitation** $c_n$: not measured, only converted.  The model starts every mode through
#   the hammer (`hammer2free`), so per key and dynamic $c_n$ is the excitation for which the
#   hammer hands over $A_n(\tau) = a_n A_0(t_0)\, e^{\sigma_n (t_0 - \tau)}$, with $A_0(t_0)$ the
#   fundamental the hammer of step 3 leaves.  The hammer model thus sets how the fundamental
#   grows with velocity, but not how the modes grow relative to it: its pulse spectrum at
#   $f_n \tau$ = 3 - 44 is not checked against the data (step 3 sees only $f_0 \tau \le 2$).
#   Near the nulls of the pulse spectrum ($f_n \tau$ = 2, 3, ...) $c_n$ is capped (`G_FLOOR`),
#   and the mode there gets less than $a_n A_0$.
#
# Velocities: `vels` of step 3 (from the bass), not the assumed `VELS` of step 2.  Everything
# else is fixed from steps 2 and 3.  Output `results/step4_modes.npz` for `synth.py` (`c`: $c_n$
# per key and dynamic; `a`: $a_n$; `a_key`, `a_dyn`: the measured $A_n / A_0$ per clear key,
# and per clear key and dynamic, nan elsewhere).


# %%
import jax

from step2_pickup import eps_free, load_wave, onset

F_MAX   = 10e3     # Hz, no modes above (Gabrielli 2020)
FS      = 48000.0  # model sample rate (= the recordings')
A_TRIAL = 1e-3     # trial amplitude A_n / A_0 (small: the pickup is linear for it)

fit2 = np.load("results/step2_pickup.npz", allow_pickle=True)
fit3 = np.load("results/step3_hammer.npz", allow_pickle=True)
assert (fit2["notes"] == fit3["notes"]).all()              # fit3 is indexed with fit2's keys
notes = list(fit2["notes"])
kappa, beta, vels = float(fit2["kappa"]), float(fit3["beta"]), fit3["vels"]
f0, sig0, c0 = fit2["f0"], fit2["sigma"], fit3["c0"]
names = list(MODES)


def median_clear(value):
    """One number per mode: the median of value(row) over part A's clear keys, nan if none."""
    return np.array([np.median([value(x) for x in rows if x["mode"] == m])
                     if any(x["mode"] == m for x in rows) else np.nan for m in names])


mu    = np.array([MODES[m][0] for m in names])                         # f_n / f_0 (Gabrielli)
rho   = median_clear(lambda x: x["ratio_f"])                            # measured, for comparison
r     = median_clear(lambda x: x["sig_n"] / sig0[notes.index(x["note"])])  # sigma_n / sigma_0
f_n   = mu[None, :] * f0[:, None]                                       # (keys, N)
sig_n = r[None, :] * sig0[:, None]
for n, m in enumerate(names):
    if np.isfinite(r[n]):
        print(f"{m}: f_n/f_0 = {mu[n]} (measured {rho[n]:.3f}), sigma_n/sigma_0 = {r[n]:.1f}")


# %%
t0_grid = jnp.arange(int(T_FFT * FS)) / FS                 # s after t_0 = onset + T_START
free = jax.jit(lambda A, s, f, p_o, p_d: kappa * eps_free(t0_grid, A, s, f, p_o, p_d, TABLE))


def ratio_a(x, j, load_rec=load_wave, A0=None):
    """A_n / A_0 at t_0 of part A's clear mode x in the recording of dynamic j; nan if that
    recording's peak is not part A's.  The mode's part of the model is the model with the
    mode (at the key's own f_n and sigma_n of part A) minus the model without it: on low keys
    the window's main lobe of a harmonic would otherwise be read.  load_rec and A0 (default:
    the recordings and A_0 of step 2) are swapped by the synthetic check (part C)."""
    i, i5 = notes.index(x["note"]), NOTES.index(x["note"])
    w, fs = load_rec(x["note"], DYNS[j])
    assert fs == FS
    f, db = spectrum(w[onset(w) + int(round(T_START * fs)):], fs)
    fp = find_peak(f, db, FUND[i5, j]["f"], *MODES[x["mode"]])
    if not abs(fp - x["f"]) <= AGREE * x["f"]:                # also nan: no peak
        return np.nan
    A0 = float(fit2["A0"][i, j]) if A0 is None else A0
    args = (jnp.array([sig0[i], x["sig_n"]]), jnp.array([f0[i], x["f"]]),
            float(fit2["p_o"][i]), float(fit2["p_d"][i]))
    mode = np.asarray(free(jnp.array([A0, A_TRIAL * A0]), *args) - free(jnp.array([A0, 0.0]), *args))
    fm, dbm = spectrum(mode, FS)
    return A_TRIAL * 10 ** ((np.interp(fp, f, db) - np.interp(x["f"], fm, dbm)) / 20)


# %%
a_dyn = np.full((len(notes), len(DYNS), len(names)), np.nan)     # A_n / A_0, clear keys only
for x in rows:
    i, n = notes.index(x["note"]), names.index(x["mode"])
    a_dyn[i, :, n] = [ratio_a(x, j) for j in range(len(DYNS))]
    print(f"{x['note']}/{x['mode']}", end=" ", flush=True)
print()

# %%
DB = 20 / np.log(10)
with warnings.catch_warnings():                                            # keys not clear: all nan
    warnings.simplefilter("ignore", RuntimeWarning)
    log_a = np.log(a_dyn)
    a_key = np.exp(np.nanmean(log_a, axis=1))                              # (keys, N)
    spread = DB * np.nanstd(log_a, axis=1)                                 # dB, (keys, N)

# One a_n per mode: the median over the clear keys, as the decays in part A.  Printed with it:
# the spread over the dynamics of a key, and how A_n / A_0 changes from p to f (median over the
# keys of the slope of log(A_n / A_0) against log v, times log(v_f / v_p))
a = np.full(len(names), np.nan)
for n, m in enumerate(names):
    k = np.isfinite(a_key[:, n])
    if not k.any():
        print(f"{m}: no clear key with a measurement")
        continue
    a[n] = np.median(a_key[k, n])
    slope = [np.polyfit(np.log(vels[u]), log_a[i, u, n], 1)[0]
             for i in np.flatnonzero(k) for u in [np.isfinite(log_a[i, :, n])] if u.sum() >= 3]
    print(f"{m}: A_n/A_0 = {DB * np.log(a[n]):+.1f} dB over {k.sum()} clear keys, spread over the "
          f"dynamics {np.median(spread[k, n]):.1f} dB, p -> f {DB * np.median(slope) * np.log(vels[-1] / vels[0]):+.1f} dB")

ok = np.isfinite(r * a)[None, :] & (f_n < F_MAX)                           # modes in the model
print("modes in the model, keys:", dict(zip(names, ok.sum(0))))


# %%
G_FLOOR = 0.1      # |sin(w tau / 2)| of the pulse spectrum floored here (see c_of)


def c_of(i, j, a_n=None):
    """c_n (N,) of key i at dynamic j: the excitation for which the hammer hands over
    A_n(tau) = a_n A_0(t_0) e^{sigma_n (t_0 - tau)}; A_0(t_0) is the fundamental the hammer of
    step 3 leaves (c_0), carried to t_0 = T_START after contact.  0 for modes not in the model.
    a_n (N,): a by default; the synthetic check (part C) sets its own per key.

    The hammer hands over |A_n| = c_n 2|K| |sin(w tau / 2)| (`hammer2free`), with nulls at
    f_n tau = 2, 3, ...  Near a null c_n would have to grow without bound, and the stiff mode
    would push the tine far out during contact.  So |sin| is floored at G_FLOOR: there the mode
    gets less than a_n A_0 (`short`: the fraction it gets, 1 elsewhere)."""
    a_n = a if a_n is None else a_n
    v = float(vels[j])
    tau = float(calc_tau(v, dict(tau_0=fit3["tau0"][i], beta=beta)))
    A0 = abs(float(hammer2free(f0[i], c0[i], v, tau)[0])) * np.exp(-sig0[i] * (T_START - tau))
    g, phi = hammer2free(jnp.asarray(f_n[i]), 1.0, v, tau)                # A_n per unit c_n, w tau / 2
    s = np.abs(np.sin(np.asarray(phi)))
    env = np.abs(np.asarray(g)) / np.maximum(s, 1e-300)                   # 2|K|: the pulse spectrum without nulls
    with np.errstate(invalid="ignore"):
        c_n = np.where(ok[i], a_n * A0 * np.exp(sig_n[i] * (T_START - tau)) / (env * np.maximum(s, G_FLOOR)), 0.0)
    return c_n, np.where(ok[i], s / np.maximum(s, G_FLOOR), 1.0)


c, short = (np.array(x) for x in zip(*[c_of(i, j) for i in range(len(notes)) for j in range(len(DYNS))]))
c, short = c.reshape(len(notes), len(DYNS), -1), short.reshape(len(notes), len(DYNS), -1)   # (keys, dyn, N)
print(f"A_n below a_n A_0 (pulse null): {(short < 1).sum()} of {ok.sum() * len(DYNS)} key x dynamic x mode, "
      f"median {DB * np.log(np.median(short[short < 1])):+.1f} dB")
np.savez(RESULTS, notes=np.array(notes), dyns=np.array(DYNS), modes=names, ratios=mu, ratios_meas=rho, vels=vels,
         f_modes=f_n, sig=np.where(ok, sig_n, np.nan), ok=ok, c=c, short=short,
         a=a, a_key=a_key, a_dyn=a_dyn, sig_ratio=r)
print(f"-> {RESULTS}")


# %% [markdown]
# ## C. Synthetic check
#
#     python step4_modes.py syn      -> cache/step4_synthetic_check.npz, plots/step4_synthetic_check.png
#
# Does part B's measurement find a known $A_n / A_0$?  Only with `syn` on the command line (as
# step 2); run cell by cell, set `SYN = True`.  Parts A and B run on **synthetic recordings
# with a known truth**:
#
# - rendered by the full model (hammer, fundamental, modes, pickup; as `synth.py`) for every
#   key and dynamic, 8 s like the recordings, with $f_n$ and $\sigma_n$ of part B;
# - a **known $A_n / A_0$ per key**: $a_n$ times a random factor within $\pm$`SYN_SPREAD`
#   (about the scatter between the real keys), the same for every dynamic, turned into $c_n$ by
#   `c_of`.  Where `c_of` caps $c_n$ at a pulse null, the truth is what the mode then gets;
# - white noise as strong as the noise floor of the matching recording (median of its spectrum
#   between $5 f_0$ and $45 f_0$ in the first 0.3 s after $t_0$).
#
# Part A runs unchanged, only the sample folder swapped; part B's `ratio_a` gets the true
# $A_0$ of the synthetic fundamental in place of step 2's fit (step 2 has its own check), so
# this checks the mode measurement alone.  Reported: how often part A finds the modes, the
# error of $A_n / A_0$ below and above `SYN_BASS` = 150 Hz, and whether part A's clear keys are
# the ones with the stronger modes (the selection behind the median).  The synthetic wavs are
# kept in `cache/syn4/` and rendered again only when the truth changes.

# %%
import os
import sys

SYN        = "syn" in sys.argv[1:]
SYN_DIR    = "cache/syn4"
SYN_SPREAD = 15.0       # dB: true A_n / A_0 per key is a_n times up to +- this
SYN_BASS   = 150.0      # Hz: below, 7.1 f_0 lies within HARM_HZ of 7 f_0
SYN_DUR, SYN_LEAD = 8.0, 10e-3                              # as synth.py


def noise_std(x, fs, f_0):
    """Std of white noise with the same spectral floor as x: median |X|^2 between 5 f_0 and
    45 f_0 (at most 20 kHz) in the first 0.3 s after t_0.  For white noise of std s, |X|^2 is
    exponential with mean s^2 sum(w^2), so its median is ln 2 times that."""
    a0 = onset(x) + int(T_START * fs)
    seg = x[a0:a0 + int(0.3 * fs)]
    w = blackmanharris(len(seg))
    P = np.abs(np.fft.rfft(seg * w)) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / fs)
    band = (f > 5 * f_0) & (f < min(45 * f_0, 20e3))
    return np.sqrt(np.median(P[band]) / (np.log(2) * np.sum(w ** 2)))


def A0_syn(i, j):
    """Free amplitude of the synthetic fundamental at t_0 = T_START after contact."""
    v = float(vels[j])
    tau = float(calc_tau(v, dict(tau_0=fit3["tau0"][i], beta=beta)))
    return abs(float(hammer2free(f0[i], c0[i], v, tau)[0])) * np.exp(-sig0[i] * (T_START - tau))


def load_syn(note, dyn):
    fs, x = wavfile.read(f"{SYN_DIR}/{note}-{dyn}.wav")
    return np.asarray(x, dtype=float), float(fs)


# %%
# ---------- 1. the truth and the synthetic recordings ----------
if SYN:
    M, N = ok.shape
    rng = np.random.default_rng(0)
    a_true = np.where(ok, a[None, :] * 10 ** (rng.uniform(-SYN_SPREAD, SYN_SPREAD, ok.shape) / 20), np.nan)
    c_syn, short_syn = (np.array(v) for v in zip(*[c_of(i, j, a_true[i]) for i in range(M) for j in range(len(DYNS))]))
    c_syn, short_syn = c_syn.reshape(M, len(DYNS), N), short_syn.reshape(M, len(DYNS), N)
    truth = np.where(ok[:, None, :], a_true[:, None, :] * short_syn, np.nan)   # A_n / A_0 in the wavs

    os.makedirs(SYN_DIR, exist_ok=True)
    stamp = f"{SYN_DIR}/truth.npz"
    fresh = not (os.path.exists(stamp) and np.allclose(np.load(stamp)["c"], c_syn))
    t_syn = jnp.arange(int(SYN_DUR * FS)) / FS
    for i, note in enumerate(notes):
        k = ok[i]
        for j, dyn in enumerate(DYNS):
            path = f"{SYN_DIR}/{note}-{dyn}.wav"
            if os.path.exists(path) and not fresh:
                continue
            p = dict(c=jnp.r_[c0[i], c_syn[i, j][k]], lam=jnp.r_[sig0[i], sig_n[i][k]],
                     f_modes=jnp.r_[f0[i], f_n[i][k]], tau_0=float(fit3["tau0"][i]), beta=beta,
                     p_d=float(fit2["p_d"][i]), p_o=float(fit2["p_o"][i]))
            x = kappa * np.asarray(epsilon(t_syn, p, TABLE, vel=float(vels[j])))
            x = np.concatenate([np.zeros(int(SYN_LEAD * FS)), x])[:int(SYN_DUR * FS)]
            s = noise_std(*load_wave(note, dyn), float(f0[i]))
            wavfile.write(path, int(FS), (x + s * rng.standard_normal(len(x))).astype(np.float32))
        print(f"{note:>4}", end=" ", flush=True)
    np.savez(stamp, c=c_syn)
    print(f"\nsynthetic recordings in {SYN_DIR}/")

# %%
# ---------- 2. part A on the synthetic recordings ----------
if SYN:
    src = open("step4_modes.py", encoding="utf-8").read().split("# ## B.")[0]
    assert 'f"Samples/{note}-{dyn}.wav"' in src
    syn = {}
    exec(src.replace('f"Samples/{note}-{dyn}.wav"', f'f"{SYN_DIR}/{{note}}-{{dyn}}.wav"'), syn)
    rows_syn = syn["rows"]
    clear_syn = np.zeros(ok.shape, bool)
    for x in rows_syn:
        clear_syn[notes.index(x["note"]), names.index(x["mode"])] = True
    print("\npart A on synthetic data:")
    for n, m in enumerate(names):
        if not ok[:, n].any():
            continue
        R = [x for x in rows_syn if x["mode"] == m]
        rf = np.median([x["ratio_f"] for x in R]) if R else np.nan
        rs = np.median([x["sig_n"] / sig0[notes.index(x["note"])] for x in R]) if R else np.nan
        print(f"  {m}: clear on {len(R):2d} keys (real: {sum(x['mode'] == m for x in rows)}, in the model: "
              f"{int(ok[:, n].sum())});  f_n/f_0 found {rf:.3f}, true {mu[n]};  "
              f"sigma_n/sigma_0 found {rs:.1f}, true {r[n]:.1f}")

# %%
# ---------- 3. part B's A_n / A_0 on the clear synthetic modes ----------
if SYN:
    a_meas = np.full(truth.shape, np.nan)
    for x in rows_syn:
        i, n = notes.index(x["note"]), names.index(x["mode"])
        a_meas[i, :, n] = [ratio_a(x, j, load_syn, A0_syn(i, j)) for j in range(len(DYNS))]
    with np.errstate(divide="ignore", invalid="ignore"):
        err = DB * np.log(a_meas / truth)                                      # (keys, dyn, N), dB
    bass = f0 < SYN_BASS
    iqr = lambda v: np.nanpercentile(v, 75) - np.nanpercentile(v, 25)
    print(f"\nA_n/A_0 measured - true (dB), clear synthetic modes x dynamics:")
    print(f"  all: median {np.nanmedian(err):+.1f}, IQR {iqr(err):.1f}, |error| median "
          f"{np.nanmedian(np.abs(err)):.1f} ({np.isfinite(err).sum()} cells)")
    for label, sel in ((f"below {SYN_BASS:.0f} Hz", bass), (f"{SYN_BASS:.0f} Hz and up", ~bass)):
        e = err[sel]
        print(f"  {label}: " + (f"median {np.nanmedian(e):+.1f}, |error| median {np.nanmedian(np.abs(e)):.1f} "
                                f"({np.isfinite(e).sum()} cells)" if np.isfinite(e).any() else "no clear synthetic mode"))

    # per mode as in part B (log mean over the dynamics, median over the clear keys); the true
    # median on the clear keys against the one on all keys shows the selection of the clear keys
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        key_meas = np.exp(np.nanmean(np.log(a_meas), axis=1))
        key_true = np.exp(np.nanmean(np.log(truth), axis=1))
    print("\nper mode, median (dB):  measured on the clear keys | true on the clear keys | true on all keys in the model")
    for n, m in enumerate(names):
        k = clear_syn[:, n] & np.isfinite(key_meas[:, n])
        if not k.any():
            continue
        slope = [np.polyfit(np.log(vels[u]), np.log(a_meas[i, u, n] / truth[i, u, n]), 1)[0]
                 for i in np.flatnonzero(k) for u in [np.isfinite(a_meas[i, :, n])] if u.sum() >= 3]
        print(f"  {m}: {DB * np.log(np.median(key_meas[k, n])):+6.1f} | {DB * np.log(np.median(key_true[k, n])):+6.1f} | "
              f"{DB * np.log(np.nanmedian(key_true[ok[:, n], n])):+6.1f}   (error p -> f "
              f"{DB * np.median(slope) * np.log(vels[-1] / vels[0]):+.1f} dB)")
    np.savez("cache/step4_synthetic_check.npz", a_true=a_true, truth=truth, a_meas=a_meas, err=err, bass=bass,
             clear=clear_syn, key_meas=key_meas, key_true=key_true, spread_db=SYN_SPREAD)
    print("-> cache/step4_synthetic_check.npz")

# %%
# ---------- 4. figure ----------
if SYN:
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axs[0]
    for n, m in enumerate(names):
        u = np.isfinite(err[:, :, n])
        if not u.any():
            continue
        b = np.broadcast_to(bass[:, None], u.shape)
        for sel, face in ((u & ~b, f"C{n}"), (u & b, "none")):
            ax.plot(DB * np.log(truth[:, :, n][sel]), DB * np.log(a_meas[:, :, n][sel]), "o", ms=3,
                    color=f"C{n}", mfc=face, label=m if face != "none" else None)
    lim = ax.get_xlim()
    ax.plot(lim, lim, "k-", lw=0.8)
    ax.set_xlabel("true $A_n/A_0$ (dB)"); ax.set_ylabel("measured $A_n/A_0$ (dB)"); ax.legend(fontsize=8)
    ax.set_title(f"measured against true (open: below {SYN_BASS:.0f} Hz)")
    ax = axs[1]
    kk = np.arange(len(notes))
    for n, m in enumerate(names):
        if np.isfinite(err[:, :, n]).any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ax.semilogy(kk, np.nanmedian(np.abs(err[:, :, n]), axis=1) + 0.1, "o", ms=3, color=f"C{n}", label=m)
    ax.set_xticks(kk[::6]); ax.set_xticklabels(notes[::6], rotation=90, fontsize=8)
    ax.set_ylabel("|error| (dB, median over dyn)"); ax.legend(fontsize=8)
    ax.set_title("error of the measured A_n/A_0 per key")
    plt.tight_layout()
    plt.savefig("plots/step4_synthetic_check.png", dpi=100)
    plt.show()
