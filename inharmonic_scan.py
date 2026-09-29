"""Scan for inharmonic energy (step 5 of fitting.py): which non-harmonic frequency bands carry
the most energy right after the hammer contact, per key.

Short-time spectra with frames of a fixed number of periods of f_0: what an inharmonic line
must be separated from is the nearest harmonic, f_0 away, so the resolution needed scales with
f_0 and a top key gets short frames.  Blackman-Harris frames (sidelobes -92 dB) so that the
harmonics, 60-80 dB over the noise, do not leak between each other; their main lobes are masked.
Every bin is normalised to the same bin late in the note (the recording's last second), so the
level reads as "dB above what is left at the end".  The axis is f / f_0, so keys line up.

Per key: the frame-averaged level over the analysis span, peaks on it, and for each peak its
level over time (decay rate, lifetime) and whether it is an intermodulation product
k f_0 +- r f_0 of a stronger peak (the pickup is nonlinear, so a mode at r f_0 comes with lines
at every integer distance from it, and at the mirror 1 - r).

    python inharmonic_scan.py [E0 E2 E4 E6 | --all] [--dyn f] [--np 16]

Hyperparameters
| Name | Value | Meaning |
| --- | --- | --- |
| `N_P` | 16 (`--np`) | frame length in periods of f_0; one bin = f_0 / N_P.  16 favours time resolution, 64 frequency resolution (lines 0.1 f_0 from a harmonic) |
| `HOP` | 0.5 | frame hop, fraction of a frame |
| `ZP` | 8 | zero padding factor (grid 1/8 bin) |
| `MASK_BINS` | 5 | masked +-5 bins around every n f_0: Blackman-Harris main lobe (+-4) + 1 (16 periods: +-0.31 f_0, 64: +-0.08 f_0) |
| `N_SPAN`, `T_SPAN`, `MIN_FRAMES` | 200 periods, 1 s, 3 | analysis span after contact: constant Q -> fixed number of periods, capped, but at least 3 frames |
| `T_NOISE` | 1 s | reference: the last second of the recording |
| `MIN_LEVEL`, `MIN_PROM` | 10 dB, 6 dB | peak height over the reference and prominence on the averaged curve; a peak needs unmasked samples for one bin on both sides (a skirt running into the mask is not a located line) |
| `N_PEAKS` | 12 | strongest peaks kept per key |
| `LIFE_DB` | 10 dB | a frame counts as "line present" above this |
| `IM_TOL`, `IM_LAM` | 0.02, 0.7 | fractional ratios matching within IM_TOL are one intermodulation family, unless the weaker decays slower than IM_LAM x the stronger |
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sg

SAMPLES, OUT = "Samples", "plots"
DYNS, VELS = ["p", "mp", "mf", "f"], np.array([0.25, 0.5, 0.75, 1.0])
ONSET_FRAC = 0.05

N_P, HOP, ZP = 16, 0.5, 8
MASK_BINS = 5
N_SPAN, T_SPAN, MIN_FRAMES = 200, 1.0, 3
T_NOISE = 1.0
MIN_LEVEL, MIN_PROM, N_PEAKS = 10.0, 6.0, 12
LIFE_DB = 10.0
IM_TOL, IM_LAM = 0.02, 0.7
R_MIN, R_MAX = 0.3, 120.0                        # ratio range shown and searched
NOMINAL = np.array([0.48, 7.1, 20.4, 39.7, 62.9, 93.1])   # mode_measurement.py, for reference only


# ---------- data ----------

def load_keys():
    """notes, f_0 (step 1), contact time tau_0 and beta (step 3) for all keys."""
    f1 = np.load("fundamentals.npz", allow_pickle=True)
    f3 = np.load("step3_fit.npz", allow_pickle=True)
    assert (f1["notes"] == f3["notes"]).all()
    return f1["notes"], np.nanmedian(f1["table"]["f"], axis=1), f3["tau0"], float(f3["beta"])


def load_wave(note, dyn):
    fs, x = wavfile.read(f"{SAMPLES}/{note}-{dyn}.wav")
    x = np.asarray(x, dtype=np.float64)
    return (x.mean(axis=1) if x.ndim > 1 else x), float(fs)


# ---------- spectra ----------

def frames(x, fs, f0, t_from, t_to):
    """(K, B) power spectra of Blackman-Harris frames of N_P periods between t_from and t_to,
    their centre times (K,) and the bin ratios f / f_0 (B,)."""
    L    = int(round(N_P / f0 * fs))
    hop  = max(1, int(HOP * L))
    i0   = int(t_from * fs)
    i1   = min(int(t_to * fs), x.size) - L
    starts = np.arange(i0, max(i1, i0) + 1, hop)
    w    = sg.windows.blackmanharris(L, sym=False)
    nfft = 1 << int(np.ceil(np.log2(ZP * L)))
    seg  = np.stack([x[s:s + L] for s in starts]) * w
    P    = np.abs(np.fft.rfft(seg, nfft)) ** 2 / w.sum() ** 2
    return P, (starts + L / 2) / fs, np.fft.rfftfreq(nfft, 1 / fs) / f0


def harmonic_mask(r):
    n = np.round(r)
    return (n >= 1) & (np.abs(r - n) < MASK_BINS / N_P)


# ---------- one key ----------

def scan_key(note, f0, tau, dyn):
    """Level over the reference per frame and bin, its frame average, and the peaks."""
    x, fs = load_wave(note, dyn)
    t_on  = np.argmax(np.abs(x) > ONSET_FRAC * np.abs(x).max()) / fs
    t0    = t_on + tau                                          # contact over
    T_frame = N_P / f0
    span  = max(min(N_SPAN / f0, T_SPAN), (1 + HOP * (MIN_FRAMES - 1)) * T_frame)
    P, t, r = frames(x, fs, f0, t0, t0 + span)
    Pn, _, _ = frames(x, fs, f0, x.size / fs - max(T_NOISE, 2.5 * T_frame), x.size / fs)
    ref   = Pn.mean(0) + 1e-30

    keep  = (r >= R_MIN) & (r <= min(R_MAX, 0.45 * fs / f0))
    mask  = harmonic_mask(r) | ~keep
    S     = 10 * np.log10(P / ref)                              # (K, B) dB over the reference
    S[:, mask] = np.nan
    mean  = np.full(r.size, np.nan)                             # frame-averaged power
    mean[~mask] = 10 * np.log10(np.mean(10 ** (S[:, ~mask] / 10), axis=0))

    curve = np.where(np.isnan(mean), -60.0, mean)
    idx, prop = sg.find_peaks(curve, height=MIN_LEVEL, prominence=MIN_PROM, distance=4 * ZP)
    inner = np.convolve(mask, np.ones(2 * ZP + 1), mode="same") == 0     # one bin clear each side
    good  = inner[idx]
    idx, h = idx[good], prop["peak_heights"][good]
    idx   = idx[np.argsort(-h)][:N_PEAKS]
    peaks = [measure_peak(S, t - t0, r, i, f0) for i in idx]
    level = 10 * np.log10(P.mean(0) + 1e-30)                    # absolute frame average, every bin
    for p, i in zip(peaks, idx):
        p["abs_db"] = float(level[i])
    flag_intermodulation(peaks)
    return dict(note=note, f0=f0, t=t - t0, r=r, S=S, mean=mean, level=level, peaks=peaks, span=span,
                n_frames=len(t), T_frame=T_frame)


def measure_peak(S, t, r, i, f0):
    """Level of one peak over time (max over +-1 original bin), its lifetime above LIFE_DB and
    decay rate from a straight line through the frames where it is present."""
    lvl = np.nanmax(S[:, max(i - ZP, 0):i + ZP + 1], axis=1)
    on  = lvl > LIFE_DB
    a   = int(np.argmax(on))                                    # first frame present
    off = np.flatnonzero(~on[a:])
    b   = a + (int(off[0]) if off.size else on.size - a)        # end of that run
    lam = -np.polyfit(t[a:b], lvl[a:b] / 8.686, 1)[0] if b - a >= 3 else np.nan   # dB -> neper
    return dict(ratio=float(r[i]), f=float(r[i] * f0),
                mean_db=float(10 * np.log10(np.nanmean(10 ** (S[:, i] / 10)))),
                max_db=float(lvl.max()), life=float(t[b - 1]) if b > a else 0.0,
                n_on=b - a, lam=float(lam), im_of=None)


def flag_intermodulation(peaks):
    """A weaker peak whose fractional ratio equals that of a stronger one (k f_0 + r f_0) or its
    mirror (k f_0 - r f_0) belongs to that one's family, unless it decays slower: a product
    carries the parent's decay plus that of the fundamental, so it cannot outlive the parent."""
    for j, p in enumerate(peaks):                               # peaks are sorted by strength
        fp = p["ratio"] % 1.0
        for q in peaks[:j]:
            fq = q["ratio"] % 1.0
            d  = min(abs(fp - fq), 1 - abs(fp - fq), abs(fp - (1 - fq)), 1 - abs(fp - (1 - fq)))
            slower = p["lam"] < IM_LAM * q["lam"]               # False if either is nan
            if d < IM_TOL and q["im_of"] is None and not slower:
                p["im_of"] = q["ratio"]
                break


# ---------- output ----------

def print_key(k):
    print(f"\n{k['note']}  f0 {k['f0']:.1f} Hz  span {k['span'] * 1e3:.0f} ms, {k['n_frames']} frames "
          f"of {k['T_frame'] * 1e3:.1f} ms")
    print(f"{'ratio':>8} {'f (Hz)':>9} {'mean dB':>8} {'max dB':>7} {'life ms':>8} {'frames':>6} "
          f"{'lam 1/s':>8} {'nominal':>8}  family")
    for p in k["peaks"]:
        nom = NOMINAL[np.argmin(np.abs(np.log(NOMINAL / p["ratio"])))]
        near = f"{nom:g}" if abs(np.log(nom / p["ratio"])) < 0.06 else ""
        fam  = f"IM of {p['im_of']:.3f}" if p["im_of"] is not None else "parent"
        print(f"{p['ratio']:8.3f} {p['f']:9.1f} {p['mean_db']:8.1f} {p['max_db']:7.1f} "
              f"{p['life'] * 1e3:8.1f} {p['n_on']:6d} {p['lam']:8.1f} {near:>8}  {fam}")


def plot_keys(res, dyn, path):
    """Per key: frame-averaged level over the ratio axis with peaks, and level over time."""
    fig, ax = plt.subplots(len(res), 2, figsize=(18, 3.6 * len(res)), squeeze=False,
                           gridspec_kw=dict(width_ratios=[1.3, 1]))
    for a, b, k in zip(ax[:, 0], ax[:, 1], res):
        a.semilogx(k["r"], k["mean"], lw=0.7)
        for p in k["peaks"]:
            c = "C3" if p["im_of"] is None else "C1"
            a.plot(p["ratio"], p["mean_db"], "v", color=c, ms=6)
            a.annotate(f"{p['ratio']:.2f}", (p["ratio"], p["mean_db"]), textcoords="offset points",
                       xytext=(0, 6), ha="center", fontsize=7, color=c)
        for nom in NOMINAL:
            a.axvline(nom, color="0.6", ls=":", lw=0.8)
        a.axhline(MIN_LEVEL, color="0.3", ls="--", lw=0.6)
        a.set_xlim(R_MIN, np.nanmax(k["r"][np.isfinite(k["mean"])]) * 1.05)
        a.set_ylabel("dB over last second"); a.grid(alpha=0.3, which="both")
        a.set_title(f"{k['note']}-{dyn} (f0 {k['f0']:.0f} Hz): frame average, harmonics masked; "
                    f"red parent, orange IM, dotted nominal", fontsize=9)
        im = b.pcolormesh(k["r"], k["t"] * 1e3, np.clip(k["S"], -10, 60), shading="nearest", cmap="magma")
        b.set_xscale("log"); b.set_xlim(a.get_xlim())
        b.set_ylabel("t after contact (ms)"); plt.colorbar(im, ax=b, label="dB")
    ax[-1, 0].set_xlabel("f / f0"); ax[-1, 1].set_xlabel("f / f0")
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close(fig)


def plot_heatmap(res, dyn, path):
    """Keys x frequency of the frame-averaged level, once over f / f_0 (lines that scale with
    the tine) and once over Hz (lines at a fixed frequency).  Masked harmonics are white."""
    fig, ax = plt.subplots(1, 2, figsize=(22, 2 + 0.2 * len(res)), sharey=True)
    for a, hz in zip(ax, (False, True)):
        grid = np.geomspace(20.0, 20e3, 2000) if hz else np.geomspace(R_MIN, R_MAX, 2000)
        Z = np.array([np.interp(np.log(grid), np.log(k["r"][1:] * (k["f0"] if hz else 1.0)),
                                k["mean"][1:], left=np.nan, right=np.nan) for k in res])
        im = a.pcolormesh(grid, np.arange(len(res)), np.clip(Z, -10, 50), shading="nearest", cmap="magma")
        a.set_xscale("log"); a.set_xlabel("f (Hz)" if hz else "f / f0")
        if not hz:
            for nom in NOMINAL:
                a.axvline(nom, color="c", ls=":", lw=0.8)
    ax[0].set_yticks(np.arange(len(res))); ax[0].set_yticklabels([k["note"] for k in res], fontsize=7)
    plt.colorbar(im, ax=ax, label="dB over last second", fraction=0.02)
    fig.suptitle(f"inharmonic scan ({dyn}): frame-averaged level after contact, harmonics masked "
                 f"(dotted: nominal ratios)")
    plt.savefig(path, dpi=90, bbox_inches="tight"); plt.close(fig)


def plot_trends(res, dyn, path):
    """Picked peaks per key, over f / f_0 and over Hz: marker area ~ level, colour = decay
    rate, parents filled, intermodulation products as small grey dots."""
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    for a, hz in zip(ax, (False, True)):
        for i, k in enumerate(res):
            for p in k["peaks"]:
                y = p["f"] if hz else p["ratio"]
                if p["im_of"] is not None:
                    a.plot(i, y, ".", color="0.7", ms=3)
                    continue
                sc = a.scatter(i, y, s=4 * max(p["mean_db"] - 5, 1), c=[np.log10(max(p["lam"], 1.0))
                               if np.isfinite(p["lam"]) else 3.0], cmap="viridis", vmin=0, vmax=3,
                               edgecolors="k", linewidths=0.3)
            a.plot(i, k["f0"] if hz else 1.0, "_", color="r", ms=6)
        a.set_yscale("log"); a.grid(alpha=0.3, which="both")
        a.set_ylabel("f (Hz)" if hz else "f / f0")
        a.set_xticks(range(0, len(res), 3))
        a.set_xticklabels([k["note"] for k in res][::3], rotation=90, fontsize=8)
        if not hz:
            for nom in NOMINAL:
                a.axhline(nom, color="c", ls=":", lw=0.8)
    plt.colorbar(sc, ax=ax, label="log10 decay rate (1/s); yellow = unmeasured/fast", fraction=0.02)
    fig.suptitle(f"inharmonic peaks ({dyn}): parents (size ~ dB over last second), IM products grey, "
                 f"red dash = f0, dotted = nominal ratios")
    plt.savefig(path, dpi=90, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:]
    dyn  = args[args.index("--dyn") + 1] if "--dyn" in args else "f"
    N_P  = int(args[args.index("--np") + 1]) if "--np" in args else N_P
    notes, f0s, tau0, beta = load_keys()
    sel  = [a for a in args if a in notes]
    sel  = list(notes) if "--all" in args else (sel or ["E0", "E2", "E4", "E6"])
    vel  = VELS[DYNS.index(dyn)]

    res = []
    for note in sel:
        i = list(notes).index(note)
        res.append(scan_key(note, f0s[i], tau0[i] * vel ** -beta, dyn))
        print_key(res[-1])

    os.makedirs(OUT, exist_ok=True)
    tag = ("all" if "--all" in args else "_".join(sel)) + f"_np{N_P}"
    if len(res) <= 8:
        plot_keys(res, dyn, f"{OUT}/inharmonic_scan_{tag}-{dyn}.png")
    plot_heatmap(res, dyn, f"{OUT}/inharmonic_heatmap_{tag}-{dyn}.png")
    plot_trends(res, dyn, f"{OUT}/inharmonic_trends_{tag}-{dyn}.png")
    np.save(f"inharmonic_scan_{tag}-{dyn}.npy",
            np.array([dict(note=k["note"], f0=k["f0"], peaks=k["peaks"]) for k in res], dtype=object))
    print(f"\n-> {OUT}/inharmonic_{{scan,heatmap,trends}}_{tag}-{dyn}.png, inharmonic_scan_{tag}-{dyn}.npy")
