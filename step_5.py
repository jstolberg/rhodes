# %% [markdown]
# # Decay ratios of the clearest tine modes
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
#    `fundamentals.npz` (`fundamental_measurement.py`). One number per mode: the median
#    over the keys.
#
# %%
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal.windows import blackmanharris

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

fund = np.load("fundamentals.npz", allow_pickle=True)
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

#%%