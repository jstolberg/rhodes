# %% [markdown]
# # Mode frequencies near the Gabrielli ratios (step 5 of fitting.py)
#
# **Why anchored.**  A free search for the inharmonic modes fails: the pickup mixes
# every mode with the harmonics, so the spectrum is full of lines at $r f_0 \pm k f_0$
# that look like modes.  Gabrielli et al. (2020) measured the tine modes by laser
# vibrometry at consistent ratios $r_n$ to $f_0$ over the whole keyboard (the spring
# position shifts them by a few percent).  So for every mode only the tallest peak
# within $\pm$`SEARCH_REL` of $r_n f_0$ is taken, with the harmonics $n f_0$ excluded.
# The sidebands $r_n f_0 \pm f_0$ lie outside the window for $r = 7.1$ and $20.4$; at
# $39.7$ they fall inside.
#
# **Per recording.**  Every key and every dynamic (292 recordings) is searched.  A
# line is accepted if it is a true local maximum (not the flank of a peak cut off by
# the harmonic guard), stands `PEAK_DB` above the median of its window and lies below
# `F_MAX_FRAC` of the sample rate.
#
# **Per key (robust lines).**  A single accepted peak can still be chance.  A line of
# a key is used in step 6 only if at least `LINE_MIN_DYN` of the four dynamics accept
# it and their frequencies agree within `LINE_SPREAD`; its frequency is then the
# median over those dynamics.
#
# **Output.**  A table (one row per recording), a summary, the plots
# `plots/step5_lines.png` and `plots/step5_lines_combined.png`, and `step5_lines.npz`.
#
# | Name | Value | Meaning |
# | --- | --- | --- |
# | `RATIOS` | 7.1, 20.4, 39.7 | Gabrielli ratios; the sub mode 0.51 is left out (mains hum) |
# | `SEARCH_REL` | 0.06, 0.04, 0.04 | search window around $r_n f_0$ (7.1: wider, measured ratios reach 7.4) |
# | `HARM_GUARD` | 0.03 | excluded around every harmonic, in units of $f_0$ |
# | `T_SEARCH` | 0.5, 0.05, 0.05 s | analysed span from onset + `T_START`: long to split 7.1 $f_0$ from 7 $f_0$, short for the fast high modes |
# | `PEAK_DB`, `F_MAX_FRAC` | 12 dB, 0.45 | acceptance: prominence over the window median; highest line / fs |
# | `LINE_MIN_DYN`, `LINE_SPREAD` | 3, 0.5 % | robust line: accepted in this many dynamics, frequencies agreeing this well |
# | `SHOW_KEYS` | [] | keys whose spectra are plotted in detail |

# %%
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from step2_lib import DYNS, T_START, load_wave, onset

RATIOS       = np.array([7.1, 20.4, 39.7])
SEARCH_REL   = np.array([0.06, 0.04, 0.04])     # one entry per ratio
HARM_GUARD   = 0.03
T_SEARCH     = np.array([0.5, 0.05, 0.05])      # one entry per ratio
PEAK_DB      = 12.0
F_MAX_FRAC   = 0.45
NFFT         = 2**20                            # FFT length (zero padding: fine frequency steps)
LINE_MIN_DYN = 3
LINE_SPREAD  = 0.005
SHOW_KEYS    = []                               # e.g. ["C2", "E3", "C5"]


def spectrum(note, dyn, T):
    """Frequencies (Hz), magnitude (dB) and sample rate of T s of a recording,
    starting T_START after the onset."""
    x, fs = load_wave(note, dyn)
    a = onset(x) + int(round(T_START * fs))
    seg = x[a:a + int(round(T * fs))]
    X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), NFFT))
    return np.fft.rfftfreq(NFFT, 1 / fs), 20 * np.log10(X + 1e-20), fs     # +1e-20: no log(0)


def find_line(f, X_db, f_nom, f0, rel):
    """Tallest point within ±rel of f_nom, harmonics excluded.
    Returns (frequency, prominence in dB over the window median, local maximum?)."""
    near_nom = np.abs(f - f_nom) < rel * f_nom                      # search window ...
    off_harm = np.abs(f / f0 - np.round(f / f0)) > HARM_GUARD       # ... without the harmonics
    band = near_nom & off_harm
    k = np.flatnonzero(band)[np.argmax(X_db[band])]
    # a local maximum is not smaller than its neighbours; otherwise it is only the flank of
    # a peak cut off at the edge of the window or of a harmonic guard
    is_peak = X_db[k] >= X_db[k - 1] and X_db[k] >= X_db[k + 1]
    return f[k], X_db[k] - np.median(X_db[band]), is_peak


# %%
# ---------- search every recording ----------
fit2  = np.load("step2_fit.npz", allow_pickle=True)
notes = [str(n) for n in fit2["notes"]]
f0    = fit2["f0"]                                   # the f_0 steps 2 and 3 used

# results per key, dynamic and ratio; nan = not searched (window above F_MAX_FRAC)
shape   = (len(notes), len(DYNS), len(RATIOS))
f_modes = np.full(shape, np.nan)                     # found frequency (Hz)
prom    = np.full(shape, np.nan)                     # prominence (dB)
peak    = np.zeros(shape, bool)                      # local maximum?

for i, note in enumerate(notes):
    for j, dyn in enumerate(DYNS):
        for n, r in enumerate(RATIOS):
            f, X_db, fs = spectrum(note, dyn, T_SEARCH[n])
            if r * f0[i] * (1 + SEARCH_REL[n]) < F_MAX_FRAC * fs:
                f_modes[i, j, n], prom[i, j, n], peak[i, j, n] = find_line(
                    f, X_db, r * f0[i], f0[i], SEARCH_REL[n])
    print(f"{note} done", end="  ", flush=True)
print()

ok       = (prom > PEAK_DB) & peak                   # accepted (nan compares False)
searched = np.isfinite(f_modes)
ratio    = f_modes / f0[:, None, None]

# ---------- robust lines per key ----------
f_acc = np.where(ok, f_modes, np.nan)                # frequencies of accepted lines only
n_acc = ok.sum(1)                                    # (keys, ratios): how many dynamics accepted
with warnings.catch_warnings():                      # all-nan columns warn; the result is nan
    warnings.simplefilter("ignore", RuntimeWarning)
    spread  = np.nanmax(f_acc, 1) / np.nanmin(f_acc, 1) - 1      # relative spread over dynamics
    f_lines = np.nanmedian(f_acc, 1)                             # median over dynamics
robust = (n_acc >= LINE_MIN_DYN) & (spread < LINE_SPREAD)       # nan compares False

# %%
# ---------- report ----------
# Table: found f/f0, "*" if rejected, prominence; "-" = not searched.
print("\nfound f/f0 (* = rejected) and prominence in dB")
print(f"{'key':>4} {'dyn':>3} {'f0 (Hz)':>8}  " + "  ".join(f"{'r=' + str(r):>15}" for r in RATIOS))
for i, note in enumerate(notes):
    for j, dyn in enumerate(DYNS):
        cells = []
        for n in range(len(RATIOS)):
            if not searched[i, j, n]:
                cells.append(f"{'-':>15}")
            else:
                mark = " " if ok[i, j, n] else "*"
                cells.append(f"{ratio[i, j, n]:6.3f}{mark} {prom[i, j, n]:5.1f}dB")
        print(f"{note:>4} {dyn:>3} {f0[i]:8.1f}  " + "  ".join(cells))

print("\naccepted / searched")
print(f"{'':>6}" + "".join(f"{'r=' + str(r):>12}" for r in RATIOS))
for j, dyn in enumerate(DYNS):
    row = "".join(f"{ok[:, j, n].sum():>6} / {searched[:, j, n].sum():<3}" for n in range(len(RATIOS)))
    print(f"{dyn:>6}{row}")
row = "".join(f"{ok[:, :, n].sum():>6} / {searched[:, :, n].sum():<3}" for n in range(len(RATIOS)))
print(f"{'all':>6}{row}")
print("robust lines per ratio:", dict(zip(RATIOS.tolist(), robust.sum(0).tolist())))

np.savez("step5_lines.npz", notes=np.array(notes), dyns=np.array(DYNS), f0=f0, ratios=RATIOS,
         f_modes=f_modes, prom_db=prom, local_max=peak, ok=ok,          # per recording
         robust=robust, f_lines=f_lines)                                # per key
print("-> step5_lines.npz")

# %%
# ---------- plots ----------
# Overview.  Top: found f/f0 over the keyboard, one colour per dynamic; filled =
# accepted, open = rejected; dotted = Gabrielli ratio, grey = search window.
# Bottom: prominence, dashed = PEAK_DB.
x = np.arange(len(notes))
fig, ax = plt.subplots(2, len(RATIOS), figsize=(6 * len(RATIOS), 7), sharex=True)
for n, r in enumerate(RATIOS):
    top, bot = ax[0, n], ax[1, n]
    top.axhspan(r * (1 - SEARCH_REL[n]), r * (1 + SEARCH_REL[n]), color="0.92")
    top.axhline(r, ls=":", color="k")
    bot.axhline(PEAK_DB, ls="--", color="k", lw=0.8)
    for j, dyn in enumerate(DYNS):
        c = f"C{j}"
        acc, rej = ok[:, j, n], searched[:, j, n] & ~ok[:, j, n]
        top.plot(x[acc], ratio[acc, j, n], "o", color=c, ms=4, label=dyn)
        bot.plot(x[acc], prom[acc, j, n], "o", color=c, ms=4)
        top.plot(x[rej], ratio[rej, j, n], "o", mfc="none", color=c, ms=4)
        bot.plot(x[rej], prom[rej, j, n], "o", mfc="none", color=c, ms=4)
    top.set_title(f"r = {r}  ({T_SEARCH[n] * 1e3:.0f} ms)")
    top.set_ylabel("f / f0"); bot.set_ylabel("prominence (dB)")
    bot.set_xticks(x[::6]); bot.set_xticklabels([notes[k] for k in x[::6]], rotation=90)
ax[0, 0].legend(title="dynamic", fontsize=8)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/step5_lines.png", dpi=90); plt.show()

# All three modes in one plot, log y axis (a relative deviation looks the same size at
# every ratio).
fig, a = plt.subplots(figsize=(14, 8))
for n, r in enumerate(RATIOS):
    a.axhspan(r * (1 - SEARCH_REL[n]), r * (1 + SEARCH_REL[n]), color="0.92")
    a.axhline(r, ls=":", color="k")
    for j, dyn in enumerate(DYNS):
        c = f"C{j}"
        acc, rej = ok[:, j, n], searched[:, j, n] & ~ok[:, j, n]
        a.plot(x[acc], ratio[acc, j, n], "o", color=c, ms=4, label=dyn if n == 0 else None)
        a.plot(x[rej], ratio[rej, j, n], "o", mfc="none", color=c, ms=4)
a.set_yscale("log"); a.minorticks_off()
a.set_yticks(RATIOS); a.set_yticklabels([str(r) for r in RATIOS]); a.set_ylabel("f / f0 (log)")
a.set_xticks(x[::3]); a.set_xticklabels([notes[k] for k in x[::3]], rotation=90)
a.set_xlim(-1, len(notes))
a.legend(title="dynamic", fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
a.set_title("found modes, all recordings (filled: accepted, open: rejected)")
plt.tight_layout()
plt.savefig("plots/step5_lines_combined.png", dpi=90); plt.show()

# Detail for SHOW_KEYS: spectrum around each search window, one line per dynamic.
# Dotted = Gabrielli ratio, light grey = harmonics, coloured = found line (dashed: rejected).
for note in SHOW_KEYS:
    i = notes.index(note)
    fig, ax = plt.subplots(1, len(RATIOS), figsize=(5 * len(RATIOS), 3))
    for n, r in enumerate(RATIOS):
        a = ax[n]
        lo, hi = r * (1 - 1.5 * SEARCH_REL[n]), r * (1 + 1.5 * SEARCH_REL[n])
        for j, dyn in enumerate(DYNS):
            f, X_db, fs = spectrum(note, dyn, T_SEARCH[n])
            win = (f / f0[i] > lo) & (f / f0[i] < hi)
            a.plot(f[win] / f0[i], X_db[win], lw=0.7, color=f"C{j}", label=dyn)
            if searched[i, j, n]:
                a.axvline(ratio[i, j, n], color=f"C{j}", ls="-" if ok[i, j, n] else "--")
        a.axvline(r, ls=":", color="k")
        for h in np.arange(np.ceil(lo), hi):
            a.axvline(h, color="0.85", lw=0.8)
        a.set_title(f"{note}, r = {r}", fontsize=9); a.set_xlabel("f / f0")
    ax[0].set_ylabel("dB"); ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"plots/step5_lines_{note}.png", dpi=90); plt.show()
