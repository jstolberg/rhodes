# %% [markdown]
# # Mode frequencies near the Gabrielli ratios (step 5 of fitting.py)
#
# Free extraction of the inharmonic modes from the recordings fails: the pickup's
# intermodulation fills the spectrum with lines at $r f_0 \pm k f_0$ that look like
# modes.  Here the search is anchored instead.  Gabrielli et al. (2020) measured the
# tine modes by laser vibrometry at consistent ratios $r_n$ to $f_0$ over the whole
# keyboard; the spring position shifts them by a few percent.  So for every mode only
# the tallest peak within $\pm$`SEARCH_REL` of $r_n f_0$ is taken, with the harmonics
# $n f_0$ excluded.  The sidebands $r_n f_0 \pm f_0$ lie outside the window for
# $r = 7.1$ and $20.4$; at $39.7$ they fall inside, so check the plot there.
#
# A line counts only if it is a true local maximum (not the flank of a peak cut off by
# the harmonic guard), stands `PEAK_DB` above the median of its window and lies below
# `F_MAX_FRAC` of the sample rate.  Result: `step5_lines.npz`, to be checked by
# eye in `plots/step5_lines.png` before step 6 uses it.
#
# | Name | Value | Meaning |
# | --- | --- | --- |
# | `MODE_KEYS` | E3 ... E4 | keys to use: middle register, clean in step 2 (see `plot_params.py`) |
# | `RATIOS` | 7.1, 20.4, 39.7 | Gabrielli ratios; the sub mode 0.51 is left out for now (mains hum) |
# | `SEARCH_REL` | 0.06, 0.04, 0.04 | search window around $r_n f_0$ (7.1: wider, measured ratios reach 7.4) |
# | `HARM_GUARD` | 0.03 | excluded around every harmonic, in units of $f_0$ |
# | `T_SEARCH` | 0.5, 0.05, 0.05 s | analysed span from onset + `T_START`: long to split 7.1 $f_0$ from 7 $f_0$, short for the fast high modes |
# | `PEAK_DB`, `F_MAX_FRAC` | 12 dB, 0.45 | acceptance: prominence over the window median; highest line / fs |

# %%
import os

import matplotlib.pyplot as plt
import numpy as np

from step2_lib import FIT_DYN, T_START, load_wave, onset

MODE_KEYS  = ["E3", "G3", "A#3", "C#4", "E4"]      # library names, one octave low
RATIOS     = np.array([7.1, 20.4, 39.7])
SEARCH_REL = np.array([0.06, 0.04, 0.04])        # one entry per ratio
HARM_GUARD = 0.03
T_SEARCH   = np.array([0.5, 0.05, 0.05])
PEAK_DB, F_MAX_FRAC = 12.0, 0.45
NFFT       = 2**20
EXCLUDE = [("E3", 7.1)]

def spectrum(note, T, dyn=FIT_DYN):
    """Frequencies, |X| in dB of T s of the sample from onset + T_START, and fs."""
    x, fs = load_wave(note, dyn)
    a = onset(x) + int(round(T_START * fs))
    seg = x[a:a + int(round(T * fs))]
    X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), NFFT))
    return np.fft.rfftfreq(NFFT, 1 / fs), 20 * np.log10(X + 1e-20), fs


def find_line(f, X_db, f_nom, f0, rel):
    """Tallest point within ±rel of f_nom, harmonics excluded -> (f_peak, prominence dB,
    whether it is a local maximum of the full spectrum)."""
    band = (np.abs(f - f_nom) < rel * f_nom) & (np.abs(f / f0 - np.round(f / f0)) > HARM_GUARD)
    k = np.flatnonzero(band)[np.argmax(X_db[band])]
    peak = X_db[k] >= X_db[k - 1] and X_db[k] >= X_db[k + 1]
    return f[k], X_db[k] - np.median(X_db[band]), peak


# %%
fit2 = np.load("step2_fit.npz", allow_pickle=True)
idx  = [list(fit2["notes"]).index(n) for n in MODE_KEYS]
f0   = fit2["f0"][idx]                                  # the f_0 steps 2 and 3 used

f_modes = np.full((len(idx), len(RATIOS)), np.nan)
prom    = np.full_like(f_modes, np.nan)
peak    = np.zeros(f_modes.shape, bool)
specs   = {}
for i, note in enumerate(MODE_KEYS):
    for n, r in enumerate(RATIOS):
        f, X_db, fs = spectrum(note, T_SEARCH[n])
        specs[i, n] = (f, X_db)
        if r * f0[i] * (1 + SEARCH_REL[n]) < F_MAX_FRAC * fs:
            f_modes[i, n], prom[i, n], peak[i, n] = find_line(f, X_db, r * f0[i], f0[i], SEARCH_REL[n])
ok = (prom > PEAK_DB) & peak                            # nan compares False
for note, r in EXCLUDE:
    ok[MODE_KEYS.index(note), list(RATIOS).index(r)] = False
    
print("f_n / f_0 found (*: rejected, nan: above F_MAX_FRAC, prominence in brackets if no local max):")
for i, note in enumerate(MODE_KEYS):
    row = "  ".join(f"{f_modes[i, n] / f0[i]:6.3f}{' ' if ok[i, n] else '*'}" for n in range(len(RATIOS)))
    pr = "  ".join(f"{p:5.1f}" if k else f"({p:.1f})" for p, k in zip(prom[i], peak[i]))
    print(f"  {note:>4} ({f0[i]:6.1f} Hz): {row}   prominence (dB): {pr}")

np.savez("step5_lines.npz", notes=np.array(MODE_KEYS), f0=f0, ratios=RATIOS,
         f_modes=f_modes, prom_db=prom, local_max=peak, ok=ok)
print("-> step5_lines.npz")

# %%
# One panel per key and mode: search window, Gabrielli ratio (dotted), harmonics (light
# grey), found line (red: accepted, grey: rejected).
fig, ax = plt.subplots(len(MODE_KEYS), len(RATIOS), figsize=(4 * len(RATIOS), 2.4 * len(MODE_KEYS)),
                       squeeze=False)
for i, note in enumerate(MODE_KEYS):
    for n, r in enumerate(RATIOS):
        f, X_db = specs[i, n]
        a, lo, hi = ax[i, n], r * (1 - 1.5 * SEARCH_REL[n]), r * (1 + 1.5 * SEARCH_REL[n])
        win = (f / f0[i] > lo) & (f / f0[i] < hi)
        a.plot(f[win] / f0[i], X_db[win], lw=0.7)
        a.axvline(r, ls=":", color="k")
        for h in np.arange(np.ceil(lo), hi):
            a.axvline(h, color="0.85", lw=0.8)
        if np.isfinite(f_modes[i, n]):
            a.axvline(f_modes[i, n] / f0[i], color="r" if ok[i, n] else "0.5")
        a.set_title(f"{note}, r = {r}, {T_SEARCH[n] * 1e3:.0f} ms", fontsize=9)
ax[-1, 0].set_xlabel("f / f_0"); ax[0, 0].set_ylabel("dB")
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/step5_lines.png", dpi=90); plt.show()
