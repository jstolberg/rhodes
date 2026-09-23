"""Decay of the inharmonic lines found by inharmonic_scan.py, over the whole recording.

Per key and dynamic, the strongest sub-fundamental, 7.x and 20.x line of the 64-period scan is
demodulated over T_MAX seconds (mixed to 0 Hz, Kaiser low-pass), and a straight line is fitted
to its log envelope from 50 ms until it stays within 10 dB of the noise.  The low-pass width is
set by the nearest neighbour: harmonics n f_0, 0 Hz, and the recordings' 60 Hz mains hum (a hum
line inside the band shows up as a flat, "non-decaying" plateau).  The noise is the same filter
half-way to that neighbour, the quieter side, over the last 1.5 s.

Reported per line: level over the noise at 0.1/1/3/6 s, decay rate over the whole fit and over
its early and late halves (equal means exponential), rms fit residual, and the ratio to the
fundamental's decay sigma_1 of step 1; per class and key range the medians of lambda,
lambda / sigma_1 and Q = pi f / lambda.

    python line_decay.py          (needs inharmonic_scan_all_np64-{f,p}.npy)

Hyperparameters
| Name | Value | Meaning |
| --- | --- | --- |
| `CLASSES` | sub < 1, 7.x 5.5-7.6, 20.x 18.5-22.5 | ratio bands; the strongest parent line of each per key |
| `T_MAX`, `DEC` | 7.5 s, 48 | analysed length after onset; envelope decimation (1 kHz at 48 kHz) |
| `BW_FRAC`, `BW_RANGE` | 0.3, 2-40 Hz | low-pass width: fraction of the distance to the nearest neighbour, clipped |
| `STOPBAND_DB` | 80 dB | Kaiser rejection |
| `HUM` | 60 Hz x k | mains hum in every recording, treated as a neighbour |
| `FIT_FROM`, `FIT_SNR` | 50 ms, 10 dB | fit window start; envelope samples used must stand this far over the noise |
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg

import inharmonic_scan as scan

CLASSES = [("sub", 0.0, 1.0), ("7.x", 5.5, 7.6), ("20.x", 18.5, 22.5)]
RANGES = [(0, 24, "E0-D#2"), (24, 48, "E2-D#4"), (48, 73, "E4-E6")]
T_MAX, DEC = 7.5, 48
BW_FRAC, BW_RANGE = 0.3, (2.0, 40.0)
STOPBAND_DB = 80.0
HUM = 60.0 * np.arange(1, 40)
FIT_FROM, FIT_SNR = 0.05, 10.0
OUT = "plots"


def load_fundamentals():
    """notes, f_0 and the step-1 decay sigma_1 per key (median over the fitted dynamics)."""
    f1 = np.load("fundamentals.npz", allow_pickle=True)
    lam = np.where(f1["table"]["ok"], f1["table"]["lam"], np.nan)
    sig1 = np.array([np.nanmedian(r) if np.isfinite(r).any() else np.nan for r in lam])
    return list(f1["notes"]), np.nanmedian(f1["table"]["f"], axis=1), sig1


def demod(x, fs, f, bw):
    """|complex envelope| of the line at f, decimated; the filter's edge samples are dropped."""
    n, beta = sg.kaiserord(STOPBAND_DB, bw / (fs / 2))
    taps = sg.firwin(n | 1, bw, fs=fs, window=("kaiser", beta))
    e = sg.oaconvolve(x * np.exp(-2j * np.pi * f * np.arange(x.size) / fs), taps, mode="same")
    return np.abs(e[:x.size - taps.size // 2:DEC])


def envelope(note, dyn, f, f0):
    """Time axis, envelope (dB), noise level (dB) and filter width for the line at f."""
    x, fs = scan.load_wave(note, dyn)
    i0 = np.argmax(np.abs(x) > scan.ONSET_FRAC * np.abs(x).max())
    x = x[i0:i0 + int(T_MAX * fs)]
    others = np.concatenate([np.arange(1, 60) * f0, [0.0], HUM])
    gap = np.min(np.abs(others - f))
    bw = float(np.clip(BW_FRAC * gap, *BW_RANGE))
    env = demod(x, fs, f, bw)
    guards = [g for g in (f - 0.5 * gap, f + 0.5 * gap) if g > 3 * bw]
    noise = min(np.median(demod(x, fs, g, bw)[-int(1.5 * fs / DEC):]) for g in guards) if guards else np.nan
    t = np.arange(env.size) * DEC / fs
    return t, 20 * np.log10(env + 1e-12), 20 * np.log10(noise + 1e-12), bw


def fit(t, L, noise_db):
    """Decay rate over the leading run above noise + FIT_SNR, over its early and late halves,
    rms residual (dB) and the end of the fitted window."""
    use = (t > FIT_FROM) & (L > noise_db + FIT_SNR)
    if use.sum() < 20:
        return np.nan, np.nan, np.nan, np.nan, 0.0
    idx = np.flatnonzero(use)
    gaps = np.flatnonzero(np.diff(idx) > 50)                    # a break of > 50 ms ends the run
    last = idx[gaps[0]] if gaps.size else idx[-1]
    sel = use & (t <= t[last])
    tt, yy = t[sel], L[sel] / 8.686                             # dB -> neper
    p = np.polyfit(tt, yy, 1)
    res = 8.686 * np.sqrt(np.mean((yy - np.polyval(p, tt)) ** 2))
    h = tt.size // 2
    early, late = -np.polyfit(tt[:h], yy[:h], 1)[0], -np.polyfit(tt[h:], yy[h:], 1)[0]
    return -p[0], early, late, res, tt[-1]


def measure_all(notes, f0s, sig1):
    rows = []
    for dyn in ("f", "p"):
        for k in np.load(f"inharmonic_scan_all_np64-{dyn}.npy", allow_pickle=True):
            i = notes.index(k["note"])
            for c, lo, hi in CLASSES:
                ps = [p for p in k["peaks"] if p["im_of"] is None and lo <= p["ratio"] < hi]
                if not ps:
                    continue
                p = max(ps, key=lambda p: p["mean_db"])
                t, L, nz, bw = envelope(k["note"], dyn, p["f"], f0s[i])
                lam, early, late, res, tend = fit(t, L, nz)
                at = {tq: L[np.argmin(np.abs(t - tq))] - nz for tq in (0.1, 1.0, 3.0, 6.0)}
                rows.append(dict(dyn=dyn, note=k["note"], i=i, cls=c, ratio=p["ratio"], f=p["f"], bw=bw,
                                 lam=lam, early=early, late=late, res=res, tend=tend, sig1=sig1[i],
                                 at=at, t=t, L=L, nz=nz))
    return rows


def print_lines(rows):
    for c, _, _ in CLASSES:
        print(f"\n=== {c}   dB over noise at 0.1/1/3/6 s | lambda (early, late) | rms resid dB | "
              f"fitted until | lambda / sigma_1")
        for r in (r for r in rows if r["cls"] == c):
            a = r["at"]
            print(f"{r['dyn']} {r['note']:4s} {r['ratio']:6.2f} {r['f']:7.0f} Hz | "
                  f"{a[0.1]:4.0f} {a[1.0]:4.0f} {a[3.0]:4.0f} {a[6.0]:4.0f} | "
                  f"{r['lam']:6.2f} ({r['early']:6.2f}, {r['late']:6.2f}) | {r['res']:4.1f} | "
                  f"{r['tend']:4.1f} s | {r['lam'] / r['sig1']:6.1f}")


def print_summary(rows, f0s):
    """Medians per class and key range, over lines fitted for at least 0.3 s."""
    print(f"\n{'class':5s} {'keys':8s} {'n':>4s} {'lambda':>7s} {'lambda/sigma_1 (IQR)':>24s} "
          f"{'Q line':>7s} {'Q fund':>7s} {'resid dB':>9s} {'early/late':>11s}")
    for c, _, _ in CLASSES:
        for lo, hi, name in RANGES:
            rr = [r for r in rows if r["cls"] == c and lo <= r["i"] < hi
                  and np.isfinite(r["lam"]) and r["lam"] > 0.3 and r["tend"] >= 0.3]
            if len(rr) < 2:
                continue
            lam = np.array([r["lam"] for r in rr])
            q = lam / np.array([r["sig1"] for r in rr])
            Q = np.pi * np.array([r["f"] for r in rr]) / lam
            Qf = np.array([np.pi * f0s[r["i"]] / r["sig1"] for r in rr])
            el = [r["early"] / r["late"] for r in rr if r["late"] > 0.3]
            print(f"{c:5s} {name:8s} {len(rr):4d} {np.median(lam):7.1f} {np.nanmedian(q):8.1f} "
                  f"({np.nanpercentile(q, 25):5.1f}-{np.nanpercentile(q, 75):5.1f}) {np.median(Q):12.0f} "
                  f"{np.nanmedian(Qf):7.0f} {np.median([r['res'] for r in rr]):9.1f} {np.median(el):11.2f}")


def plot_sub_envelopes(rows, path):
    """Envelopes of twelve sub-fundamental lines at p, spread over the keyboard, with the fit."""
    sub = [r for r in rows if r["cls"] == "sub" and r["dyn"] == "p"]
    fig, ax = plt.subplots(3, 4, figsize=(18, 10), sharex=True)
    for a, r in zip(ax.ravel(), sub[::max(1, len(sub) // 12)][:12]):
        a.plot(r["t"], r["L"], lw=0.6); a.axhline(r["nz"], color="0.5", ls=":")
        if np.isfinite(r["lam"]):
            L0 = r["L"][np.argmin(np.abs(r["t"] - 0.1))]
            a.plot(r["t"], L0 - 8.686 * r["lam"] * (r["t"] - 0.1), "C1", lw=0.8)
        a.set_ylim(r["nz"] - 10, r["L"].max() + 5)
        a.set_title(f"{r['dyn']} {r['note']} {r['f']:.0f} Hz ({r['ratio']:.2f} f0) lam {r['lam']:.1f}", fontsize=9)
    for a in ax[-1]:
        a.set_xlabel("t after onset (s)")
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close(fig)


def plot_decay_vs_key(rows, notes, f0s, sig1, path):
    """Decay rates of the lines against the key and against their frequency, with sigma_1."""
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    x = np.arange(len(notes))
    ax[0].semilogy(x, sig1, "k-", lw=1.5, label="fundamental sigma_1 (step 1)")
    ax[1].loglog(f0s, sig1, "k-", lw=1.5, label="fundamental: sigma_1 vs f_0")
    mk = {"sub": "o", "7.x": "s", "20.x": "^"}
    for c, _, _ in CLASSES:
        for dyn, col in (("f", "C0"), ("p", "C3")):
            rr = [r for r in rows if r["cls"] == c and r["dyn"] == dyn and np.isfinite(r["lam"]) and r["lam"] > 0]
            kw = dict(color=col, mfc="none", label=f"{c} ({dyn})")
            ax[0].semilogy([r["i"] for r in rr], [r["lam"] for r in rr], mk[c], **kw)
            ax[1].loglog([r["f"] for r in rr], [r["lam"] for r in rr], mk[c], **kw)
    for a in ax:
        a.grid(alpha=0.3, which="both"); a.legend(fontsize=8, ncol=2); a.set_ylabel("decay rate (1/s)")
    ax[0].set_xticks(x[::3]); ax[0].set_xticklabels(np.array(notes)[::3], rotation=90, fontsize=8)
    ax[1].set_xlabel("line frequency (Hz)")
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close(fig)


if __name__ == "__main__":
    notes, f0s, sig1 = load_fundamentals()
    rows = measure_all(notes, f0s, sig1)
    print_lines(rows)
    print_summary(rows, f0s)
    np.save("line_decay.npy", np.array([{k: v for k, v in r.items() if k not in ("t", "L")} for r in rows],
                                       dtype=object))
    os.makedirs(OUT, exist_ok=True)
    plot_sub_envelopes(rows, f"{OUT}/sub_envelopes_p.png")
    plot_decay_vs_key(rows, notes, f0s, sig1, f"{OUT}/line_decay_vs_key.png")
    print(f"-> line_decay.npy, {OUT}/sub_envelopes_p.png, {OUT}/line_decay_vs_key.png")
