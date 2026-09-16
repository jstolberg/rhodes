# %% [markdown]
# # Measuring the Fundamental Frequency $f_0$ and Its Decay Rate $\lambda_0$
#
# The theoretical frequency of a given musical note is known. However, real-world instruments are not perfectly tuned, meaning that the actual frequency of a recorded sample can differ slightly from its theoretical value. To determine the fundamental frequency and its decay rate from the recorded signal, the following procedure is applied.
#
# 1. **Compute the Fourier Transform**
#
# First, the Fourier Transform (FFT) is computed over the recorded sample to obtain its frequency spectrum.
#
# 2. **Identify the fundamental frequency**
#
# A small frequency window is selected around the theoretically expected fundamental frequency. Within this window, the frequency with the highest energy is identified as the measured fundamental frequency $f_0$.
#
# 3. **Demodulate the signal**
#
# To isolate the fundamental component, the signal is shifted to 0 Hz by multiplying it with a complex exponential at the measured frequency:
#
# $$ e(t) = \operatorname{lowpass}\left(x(t)e^{-2\pi i f_0t}\right) $$
#
# After this frequency shift, the fundamental component is centered around 0 Hz. Applying a low-pass filter suppresses the remaining harmonics and other frequency components.
#
# 4. **Determine the decay rate**
#
# The resulting signal $e(t)$ is complex-valued. Its magnitude represents the amplitude envelope of the fundamental component. Assuming an exponential decay:
#
# $$ |e(t)| = A e^{-\lambda_0 t} $$
#
# Taking the logarithm gives a linear relationship:
#
# $$ \log |e(t)| = \log A - \lambda_0 t $$
#
# Therefore, the magnitude of $e(t)$ is calculated for every sample, and a linear regression is performed on its logarithm. The negative slope of the fitted line corresponds to the decay rate $\lambda_0$.
#
# 5. **Determine the frequency offset**
#
# The phase of the complex signal can be analyzed in the same way. Any residual linear change in phase over time indicates that the actual frequency differs slightly from the frequency used for demodulation. This phase evolution can therefore be used to estimate the frequency offset and refine the measured fundamental frequency.
#
# More precisely, if $\phi(t)$ denotes the unwrapped phase, the slope of the phase is related to the frequency difference by:
#
# $$ \frac{d\phi(t)}{dt} = 2\pi(f_{\text{actual}}-f_0) $$
#
# ## Dataset
#
# The dataset consists of four individual recordings for each piano key, corresponding to four different playing velocities. Having multiple recordings per note provides additional measurements for estimating the fundamental frequency and decay rate. It also allows the consistency of the measurements to be evaluated across different playing dynamics.
#
# ## (Pre-)Processing of the Samples
#
# Each sample is approximately 8 seconds long. The recordings contain a fading tail beginning at approximately 7.3 seconds. Including this section in the decay estimation would introduce an artificial change in the measured decay rate. Therefore, the samples are truncated before this point.
#
# The initial attack of the sound is also excluded from the analysis. Since the goal is to measure the decay of the fundamental frequency, the transient attack phase is not representative of the steady-state decay and would distort the exponential fit.
#
# ## What Is Saved, and What Is Not
#
# The result is one row per recording: every note at every velocity, exactly as measured. Nothing is averaged across velocities, nothing is filled in, and nothing is fitted. Those are modelling decisions, and they belong to the model rather than to the measurement.
#
# The velocities are kept separate rather than combined into a single value per note. Averaging them would assume that the four recordings are repeated measurements of one quantity, and that assumption is not made here. Whether they can be collapsed, and how, is left to be decided from the measurements themselves.
#
# The lowest notes have particularly long decay times. Within the approximately 7.3-second measurement window their amplitude decreases too little to time the decay at all, and for the loudest recordings of those notes it barely decreases. Where the decay cannot be established the decay rate is left empty. Each row still carries the evidence behind it: how far the line was watched to fall, how straight it was while falling, and a flag recording whether that was sufficient. The frequency is reported either way, since a line too slow to time is still well located.
# %%
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sg
from scipy.ndimage import uniform_filter1d

SAMPLES, DYNS = "Samples", ("p", "mp", "mf", "f")
RESULTS = "fundamentals.npz"

TAIL_TRIM = 0.70   # s cut off the end, clearing the library's fade-out
FALL_MAX = 40.0    # dB below the envelope peak the fit runs to.  A fixed depth
                   # fixes what lam means: every note is averaged over the
                   # same part of its own decay.
FALL_MIN = 8.0     # a fit that watched less fall than this is not a decay

_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_of(note):
    """Sample-library note name -> MIDI number.  The files are labelled an
    octave low, hence the +2 rather than the usual +1."""
    return (int(note[-1]) + 2) * 12 + _NAMES.index(note[:-1])


def f0_of(note):
    return 440.0 * 2.0 ** ((midi_of(note) - 69) / 12.0)


# %%
# ---------- one recording ----------

FIELDS = ("f", "lam", "amp0", "fell_db", "resid_db", "ok")
DTYPE = np.dtype([(k, "?" if k == "ok" else "f8") for k in FIELDS])


def measure(note, dyn):
    """Measure one recording.  Returns a tuple in FIELDS order."""
    fs, x = wavfile.read(f"{SAMPLES}/{note}-{dyn}.wav")
    fs = float(fs)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    # cut the trailing tail and the attack from the sample 
    x = x[int(0.05 * fs):-int(TAIL_TRIM * fs)]

    # f0 from one big FFT of the loud opening: the largest bin within 3% of
    # nominal, at 0.023 Hz a bin.
    n = min(int(0.8 * fs), x.size)
    spec = np.abs(np.fft.rfft(x[:n] * np.hanning(n), 1 << 21))
    freq = np.fft.rfftfreq(1 << 21, 1 / fs)
    near = np.abs(freq - f0_of(note)) < 0.03 * f0_of(note)
    f0 = float(freq[near][np.argmax(spec[near])])

    # Mix to 0 Hz and low-pass.  The Kaiser FIR is symmetric and applied
    # mode="same", so its group delay cancels; only the incomplete-convolution
    # edges are trimmed.
    bw = max(5.0, min(60.0, 0.15 * f0))
    n_taps, beta = sg.kaiserord(100.0, bw / (fs / 2))
    n_taps = min(n_taps | 1, (x.size // 3) | 1)
    taps = sg.firwin(n_taps, bw, fs=fs, window=("kaiser", beta))
    env = sg.fftconvolve(x * np.exp(-2j * np.pi * f0 * np.arange(x.size) / fs),
                         taps, mode="same")[n_taps // 2: x.size - n_taps // 2]
    t = (np.arange(env.size) + n_taps // 2) / fs

    # Peak of the envelope, down to FALL_MAX dB below it.
    y = 20 * np.log10(np.abs(env) + 1e-30)
    smooth = uniform_filter1d(y, size=max(3, int(0.05 * fs) | 1))
    i0 = int(np.argmax(smooth))
    below = np.flatnonzero(smooth[i0:] < smooth[i0] - FALL_MAX)
    i1 = i0 + (int(below[0]) if below.size else y.size - i0)
    if i1 - i0 < 64:
        # No window to fit.  f comes from the spectrum and still stands;
        # the four fit values are left empty.
        return (f0, np.nan, np.nan, np.nan, np.nan, False)

    tt, yy = t[i0:i1], y[i0:i1]
    slope, intercept = np.polyfit(tt, yy, 1)
    # arg(env) turns at 2*pi*(f_true - f0): the phase slope is the frequency
    # error, far sharper than the peak pick.
    turn = np.polyfit(tt, np.unwrap(np.angle(env[i0:i1])), 1)[0]

    lam = -slope / 8.686
    fell = -slope * (tt[-1] - tt[0])
    resid = float(np.sqrt(np.mean((yy - (intercept + slope * tt)) ** 2)))

    # lam is reported only when the line was watched to fall FALL_MIN.  Across
    # a shorter fall the slope describes whatever else moved the envelope, so
    # it is left empty.  fell_db and resid_db are kept either way and record
    # how far the line fell and how straight it was.
    ok = fell >= FALL_MIN
    return (f0 + turn / (2 * np.pi), lam if ok else np.nan,
            10 ** (intercept / 20), fell, resid, ok)


# %%
# ---------- the whole library ----------
# One structured array of shape (n_notes, n_dyns).  t["lam"] is the (73, 4)
# grid, t["lam"][i, j] one recording, and notes[i] / DYNS[j] name it.


def measure_library(notes):
    t = np.zeros((len(notes), len(DYNS)), DTYPE)
    for i, note in enumerate(notes):
        for j, dyn in enumerate(DYNS):
            t[i, j] = measure(note, dyn)
        print(f"  {i + 1:3d}/{len(notes)}  {note}", end="\r")
    return t


NOTES = sorted({p.rsplit("-", 1)[0] for p in os.listdir(SAMPLES)
                if p.endswith(".wav")}, key=midi_of)

if __name__ == "__main__":
    t = measure_library(NOTES)
    np.savez_compressed(RESULTS, table=t, notes=np.array(NOTES),
                        dyns=np.array(DYNS))
    usable = t["ok"].any(axis=1)
    print(f"\nwrote {RESULTS}: {t['ok'].sum()}/{t.size} recordings measured, "
          f"{usable.sum()}/{len(NOTES)} notes with at least one usable take"
          f" (none on {', '.join(np.array(NOTES)[~usable])})")


def load(path=RESULTS):
    """-> (table, notes, dyns).

    table is the (n_notes, n_dyns) structured array of measurements: one row
    per recording, table["lam"][i, j] belonging to notes[i] struck at dyns[j].

    Each row is one recording as measured.  lam is empty where ok is False;
    f, fell_db and resid_db are reported either way.  Collapsing the four
    velocities into one number, and deciding what to do where lam is empty,
    is the model's job:

        t, notes, dyns = load()
        good = t["ok"]                      # what is worth using
        t["lam"][i][good[i]].mean()         # if one value per note is wanted
    """
    z = np.load(path, allow_pickle=False)
    return z["table"], z["notes"], z["dyns"]


# %% [markdown]
# ## What came out
#
# One line per note showing all four velocities, so the velocity trend is
# visible directly. `lam` is empty where that take did not pass the `FALL_MIN`
# test. The `fell` column gives the range across the four velocities: how far
# the line was watched to fall, which is the evidence behind each `lam`.

# %%
if __name__ == "__main__":
    print(f"\n{'note':>5} {'f (Hz)':>9} {'cents':>6}   "
          + "  ".join(f"{'lam ' + str(d):>10}" for d in DYNS)
          + f"  {'fell min..max dB':>16}  {'f/p':>5}")
    for i, note in enumerate(NOTES):
        cents = 1200 * np.log2(np.median(t["f"][i]) / f0_of(note))
        cells = []
        for j in range(len(DYNS)):
            lam = t["lam"][i, j]
            cells.append(f"{lam:10.3f}" if np.isfinite(lam) else f"{'--':>10}")
        both = t["ok"][i, 0] and t["ok"][i, -1]
        ratio = (f"{t['lam'][i, -1] / t['lam'][i, 0]:5.2f}" if both
                 else f"{'':>5}")
        print(f"{note:>5} {np.median(t['f'][i]):9.3f} {cents:+6.1f}   "
              + "  ".join(cells)
              + f"  {np.nanmin(t['fell_db'][i]):7.1f}"
                f"..{np.nanmax(t['fell_db'][i]):<7.1f}"
              + f"  {ratio}")

    good = t["ok"]
    full = good.all(axis=1)
    rel = t["lam"][full] / np.median(t["lam"][full], axis=1, keepdims=True)
    ratios = t["lam"][full][:, -1] / t["lam"][full][:, 0]
    print(f"""
recordings   {good.sum()}/{t.size} usable;  {good.any(axis=1).sum()}/{len(NOTES)} notes have at least one
             (no usable take at all: {', '.join(np.array(NOTES)[~good.any(axis=1)])})
tuning       {np.median([1200 * np.log2(np.median(t['f'][i]) / f0_of(n))
                        for i, n in enumerate(NOTES)]):+.1f} cents median
residual     {np.median(t['resid_db'][good]):.2f} dB median, \
{t['resid_db'][good].max():.2f} dB worst

velocity     lam relative to each note's own median, over the {full.sum()} notes with all four:
             """ + "   ".join(f"{d}={rel[:, j].mean():.3f}"
                              for j, d in enumerate(DYNS)) + f"""
             loudest/softest = {np.median(ratios):.2f} median, \
faster when loud on {(ratios > 1).sum()}/{full.sum()} notes

Nothing above is averaged into the saved file.  Collapsing the velocities, and
deciding what to do about the notes with no usable take, is the model's job.""")

# %%
if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(12, 5))
    f_note = np.median(t["f"], axis=1)

    # Every usable recording, coloured by velocity.  The four series separate
    # vertically by the velocity dependence of the decay.
    for j, d in enumerate(DYNS):
        m = t["ok"][:, j]
        ax.plot(f_note[m], 6.91 / t["lam"][m, j], "o", ms=3.5, alpha=.8,
                label=str(d))
    none_ok = ~t["ok"].any(axis=1)
    if none_ok.any():
        ax.plot(f_note[none_ok], np.full(none_ok.sum(), 60.0), "x", color="C3",
                label="no usable take (decay exceeds the recording)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("f (Hz)"); ax.set_ylabel("t60 (s)")
    ax.legend(fontsize=8, ncol=5); ax.grid(alpha=.3, which="both")
    ax.set_title("decay across the keyboard, every velocity measured")
    fig.tight_layout(); plt.show()
