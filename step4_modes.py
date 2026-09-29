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


def ratio_a(x, j):
    """A_n / A_0 at t_0 of part A's clear mode x in the recording of dynamic j; nan if that
    recording's peak is not part A's.  The mode's part of the model is the model with the
    mode (at the key's own f_n and sigma_n of part A) minus the model without it: on low keys
    the window's main lobe of a harmonic would otherwise be read."""
    i, i5 = notes.index(x["note"]), NOTES.index(x["note"])
    w, fs = load_wave(x["note"], DYNS[j])
    assert fs == FS
    f, db = spectrum(w[onset(w) + int(round(T_START * fs)):], fs)
    fp = find_peak(f, db, FUND[i5, j]["f"], *MODES[x["mode"]])
    if not abs(fp - x["f"]) <= AGREE * x["f"]:                # also nan: no peak
        return np.nan
    A0 = float(fit2["A0"][i, j])
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


def c_of(i, j):
    """c_n (N,) of key i at dynamic j: the excitation for which the hammer hands over
    A_n(tau) = a_n A_0(t_0) e^{sigma_n (t_0 - tau)}; A_0(t_0) is the fundamental the hammer of
    step 3 leaves (c_0), carried to t_0 = T_START after contact.  0 for modes not in the model.

    The hammer hands over |A_n| = c_n 2|K| |sin(w tau / 2)| (`hammer2free`), with nulls at
    f_n tau = 2, 3, ...  Near a null c_n would have to grow without bound, and the stiff mode
    would push the tine far out during contact.  So |sin| is floored at G_FLOOR: there the mode
    gets less than a_n A_0 (`short`: the fraction it gets, 1 elsewhere)."""
    v = float(vels[j])
    tau = float(calc_tau(v, dict(tau_0=fit3["tau0"][i], beta=beta)))
    A0 = abs(float(hammer2free(f0[i], c0[i], v, tau)[0])) * np.exp(-sig0[i] * (T_START - tau))
    g, phi = hammer2free(jnp.asarray(f_n[i]), 1.0, v, tau)                # A_n per unit c_n, w tau / 2
    s = np.abs(np.sin(np.asarray(phi)))
    env = np.abs(np.asarray(g)) / np.maximum(s, 1e-300)                   # 2|K|: the pulse spectrum without nulls
    with np.errstate(invalid="ignore"):
        c_n = np.where(ok[i], a * A0 * np.exp(sig_n[i] * (T_START - tau)) / (env * np.maximum(s, G_FLOOR)), 0.0)
    return c_n, np.where(ok[i], s / np.maximum(s, G_FLOOR), 1.0)


c, short = (np.array(x) for x in zip(*[c_of(i, j) for i in range(len(notes)) for j in range(len(DYNS))]))
c, short = c.reshape(len(notes), len(DYNS), -1), short.reshape(len(notes), len(DYNS), -1)   # (keys, dyn, N)
print(f"A_n below a_n A_0 (pulse null): {(short < 1).sum()} of {ok.sum() * len(DYNS)} key x dynamic x mode, "
      f"median {DB * np.log(np.median(short[short < 1])):+.1f} dB")
np.savez(RESULTS, notes=np.array(notes), dyns=np.array(DYNS), modes=names, ratios=mu, ratios_meas=rho, vels=vels,
         f_modes=f_n, sig=np.where(ok, sig_n, np.nan), ok=ok, c=c, short=short,
         a=a, a_key=a_key, a_dyn=a_dyn, sig_ratio=r)
print(f"-> {RESULTS}")
