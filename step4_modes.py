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
# ## B. Excitation of the modes on every key
#
# Part A measures each mode's decay only on the keys where the mode is clear, as a
# multiple of the fundamental's decay.  Part B puts the modes on **all** keys, using only
# what part A gives:
#
# - **Frequency**: $f_n = \mu_n f_0$, Gabrielli's mean ratio (`MODES` of part A) times the
#   key's $f_0$ (step 2).  Modes above `F_MAX` = 10 kHz are left out (Gabrielli found none).
# - **Decay**: $\sigma_n = r_n \sigma_0$, $r_n$ the median of part A's ratios over the clear
#   keys.  Modes without a clear key have no $r_n$ and are left out.
# - **Excitation** $c_n$, the only thing set here, treated like the decays: measured only on
#   the keys where part A found the mode clearly, as a ratio $c_n / c_0$, and the median over
#   those keys applied to all keys ($c_n = (c_n/c_0) \cdot c_0$).
#   The measurement: part A's spectrum of the first `T_FFT` s and its peak search near
#   $\mu_n f_0$ give the mode's level in the recording.  The model's spectrum over the same
#   span, read at $f_n$, is proportional to $c_n$ (a small mode passes the pickup linearly),
#   so one render with a trial $c_n$ and one scaling match the two.  This is done per
#   dynamic; a key's ratio is the mean of $\log(c_n / c_0)$ over the dynamics.
#   Keys below `F0_MIN` = 150 Hz are not measured: there $7.1 f_0$ lies within `HARM_HZ` of
#   $7 f_0$, which the peak search excludes.
#
# Everything else is fixed from steps 2 and 3.  Output `results/step4_modes.npz` for `synth.py`
# (`c`: $c_n$ per key, repeated per dynamic; `c_dyn`: the measured value of each dynamic, 0
# where not measured; `measured`: keys at or above `F0_MIN`).


# %%
F_MAX   = 10e3     # Hz, no modes above (Gabrielli 2020)
FS      = 48000.0  # model sample rate
A_TRIAL = 1e-3     # trial amplitude A_n / A_0 (small: the pickup is linear for it)
F0_MIN  = 150.0    # Hz, below: c_n not measurable (see above)

fit2 = np.load("results/step2_pickup.npz", allow_pickle=True)
fit3 = np.load("results/step3_hammer.npz", allow_pickle=True)
assert (fit2["notes"] == fit3["notes"]).all()              # fit3 is indexed with fit2's keys
kappa, beta = float(fit2["kappa"]), float(fit3["beta"])
names = list(MODES)
mu, sd = np.array([MODES[m] for m in names]).T
r = np.array([np.median([x["ratio"] for x in rows if x["mode"] == m])
              if any(x["mode"] == m for x in rows) else np.nan for m in names])
notes = fit2["notes"]
f_n   = mu[None, :] * fit2["f0"][:, None]                  # (keys, N)
ok    = np.isfinite(r)[None, :] & (f_n < F_MAX)             # modes in the model
sig_n = r[None, :] * fit2["sigma"][:, None]
print("modes in the model, keys:", dict(zip(names, ok.sum(0))))


# %%
def model_db(i, c, vel):
    """Part A's spectrum (dB) of the modes' part of the model over the first T_FFT s after
    contact, read at the kept f_n (-inf for the others).  The modes' part is the model with
    the kept modes (excitation c) minus the model without them: on low keys 7.1 f_0 lies
    within the window's main lobe of 7 f_0, whose leakage would otherwise be read."""
    k = ok[i]
    t = jnp.asarray(T_START + np.arange(int(T_FFT * FS)) / FS)

    def eps(c_n):
        p = dict(c=jnp.r_[fit3["c0"][i], c_n], lam=jnp.r_[fit2["sigma"][i], sig_n[i][k]],
                 f_modes=jnp.r_[fit2["f0"][i], f_n[i][k]], tau_0=float(fit3["tau0"][i]),
                 beta=beta, p_d=float(fit2["p_d"][i]), p_o=float(fit2["p_o"][i]))
        return kappa * np.asarray(epsilon(t, p, TABLE, vel=vel))

    f, db = spectrum(eps(c[k]) - eps(0.0 * c[k]), FS)
    return np.where(k, np.interp(f_n[i], f, db), -np.inf)


def amps(i, vel):
    """Free amplitude A_0 of the fundamental and |A_n| per unit c_n that the hammer hands over."""
    tau = calc_tau(vel, dict(tau_0=fit3["tau0"][i], beta=beta))
    A0 = abs(float(hammer2free(fit2["f0"][i], fit3["c0"][i], vel, tau)[0]))
    g = np.abs(np.asarray(hammer2free(jnp.asarray(f_n[i]), 1.0, vel, tau)[0]))
    return A0, np.maximum(g, 1e-30)


# %%
measured = fit2["f0"] >= F0_MIN                             # keys whose c_n is measured
c = np.zeros((len(notes), len(DYNS), len(names)))
for i, note in enumerate(notes):
    if not ok[i].any() or not measured[i]:
        continue
    i5 = NOTES.index(note)
    for j, dyn in enumerate(DYNS):
        # level in the recording: part A's peak near mu_n f_0 (-inf where there is none)
        f, db = spectrum(*load(note, dyn))
        fp = [find_peak(f, db, FUND[i5, j]["f"], m, s) for m, s in zip(mu, sd)]
        db_rec = np.array([np.interp(x, f, db) if np.isfinite(x) else -np.inf for x in fp])
        # trial c_n: free amplitude A_TRIAL * A_0, then scaled to the recording's level
        A0, g = amps(i, VELS[j])
        c_try = np.where(ok[i], A_TRIAL * A0 / g, 0.0)
        with np.errstate(invalid="ignore"):
            c[i, j] = np.where(ok[i], c_try * 10 ** ((db_rec - model_db(i, c_try, VELS[j])) / 20), 0.0)
    print(f"{note:>4}", end=" ", flush=True)

# %%
c = c_dyn = np.nan_to_num(c)                                              # (keys, dyn, N)
used = ok[:, None, :] & (c > 0)
print("\nkeys x dynamics with a c_n per mode:", dict(zip(names, used.sum((0, 1)))))

# Keys where part A found the mode clearly
clear = np.zeros(ok.shape, bool)
for x in rows:
    clear[list(notes).index(x["note"]), names.index(x["mode"])] = True
clear &= ok

# c_n / c_0 per key: mean of the log over the dynamics; spread over the dynamics in dB
c0 = fit3["c0"]
log_rel = np.where(used, np.log(np.where(used, c, 1.0)) - np.log(c0)[:, None, None], np.nan)
with warnings.catch_warnings():                                            # unmeasured keys: all nan
    warnings.simplefilter("ignore", RuntimeWarning)
    key_rel = np.nanmean(log_rel, axis=1)                                  # (keys, N)
    spread = 20 / np.log(10) * np.nanstd(log_rel, axis=1)                 # dB, (keys, N)

# One c_n / c_0 per mode: the median over the clear keys, as the decays in part A
rel = np.full(len(names), np.nan)
for n, m in enumerate(names):
    k = clear[:, n] & np.isfinite(key_rel[:, n])                          # clear and measured
    if k.any():
        rel[n] = np.median(key_rel[k, n])
        print(f"{m}: c_n/c_0 = {20 / np.log(10) * rel[n]:+.1f} dB over {k.sum()} clear keys "
              f"({clear[:, n].sum()} clear), spread over the dynamics {np.median(spread[k, n]):.1f} dB")
    else:
        print(f"{m}: no clear key with a measurement")
c = np.where(ok & np.isfinite(rel)[None, :], np.exp(rel)[None, :] * c0[:, None], 0.0)   # (keys, N)
c = np.repeat(c[:, None, :], len(DYNS), axis=1)                            # same for every dynamic
np.savez(RESULTS, notes=notes, dyns=np.array(DYNS), ratios=mu, modes=names,
         f_modes=f_n, ok=ok, used=used, measured=measured, c=c, c_dyn=c_dyn, sig=np.where(ok, sig_n, np.nan))
print(f"-> {RESULTS}")
