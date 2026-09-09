# %% [markdown]
# # Mode frequencies and decay rates, measured
#
# For one note, read each mode's frequency and decay rate straight off the
# recording. Nothing is fitted.
#
# ![how f and lambda are measured](images/freq_measurement.png)
#
# ## How
#
# **Coarse guess.** One $2^{21}$-point FFT of the loud opening. Take the
# tallest bin within $\pm 6\%$ of the mode's nominal ratio. A starting point,
# nothing more.
#
# **Isolate the line.** Multiply the signal by a complex exponential at the
# guessed frequency $f_g$, which slides the spectrum so that line sits at
# 0 Hz, then low-pass:
#
# $$ e(t) = \mathrm{lowpass}\left( x(t)\, e^{-2\pi i f_g t} \right) $$
#
# What survives is one complex number per sample, the line's *envelope*. Its
# neighbours, meaning the other modes and every $n f_0$, end up about 100 dB
# down.
#
# **Frequency, from the phase.** If the line really sits $\delta$ Hz off the
# guess, the envelope's phase turns at $2 \pi \delta$. So fit a straight line
# to the unwrapped phase and take its slope:
#
# $$ f = f_g + \frac{1}{2\pi} \frac{\mathrm{d}}{\mathrm{d}t} \arg e(t) $$
#
# **Decay, from the magnitude.** A decaying sinusoid has
# $|e(t)| = A \mathrm{e}^{-\lambda t}$, so its log magnitude is a straight line
# whose slope is the decay rate:
#
# $$ \log |e(t)| = \log A - \lambda t $$
#
# Two straight-line fits on the one envelope: the phase gives $f$, the
# magnitude gives $\lambda$ and $A$.
#
# ## Deciding whether a mode is there at all
#
# Three tests:
#
# 1. **SNR.** Something has to stand above the noise measured beside the line.
# 2. **Decay.** Noise has no slope, so compare the fitted $\lambda$ against its
#    own standard error, $\lambda / \mathrm{se}(\lambda)$.
# 3. **Harmonic.** The pickup is a static nonlinearity, so it never moves a
#    line, it only adds new ones at $n f_0$. A candidate landing on $n f_0$ may
#    therefore be distortion rather than a mode, and distortion there decays at
#    roughly $n \lambda_1$.
#
# The residual of the magnitude fit is printed but never acted on. A mode whose
# decay is not a straight line is still a mode.

# %%
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sg
from scipy.ndimage import uniform_filter1d

# %%
# ---------- configuration ----------

SAMPLES = "Samples"
NOTE = "D3"
DYN = "f"

RATIOS_NOMINAL = np.array([1.0, 0.48, 7.1, 20.4, 39.7, 62.9, 93.1])
FUNDAMENTAL = 0        # the reference mode: the harmonic test needs its lam

MEASURE_START = 0.10   # s, past the attack
MEASURE_LEN = 6.00     # s

FIND_TOL = 0.06        # peak search: this fraction either side of nominal

BW_FRACTION = 0.4      # low-pass width, as a fraction of the distance to the
BW_MIN_HZ = 5.0        # nearest other line.  At 0.4 that line falls in the
BW_MAX_HZ = 60.0       # stopband rather than on the filter's skirt.
STOPBAND_DB = 100.0    # rejection needed.  f0 stands ~80 dB over the noise, so
                       # a Hamming design (~53 dB) leaks it into other bands.
FIT_SNR_DB = 10.0      # skip envelope samples below this: the log of a noisy
                       # amplitude drags the slope toward zero

MIN_SNR_DB = 12.0      # a line must stand this far over its local noise
MIN_DECAY_T = 5.0      # lam / se(lam), since noise has no slope
HARM_NEAR_HZ = 2.0     # a candidate this close to n*f0 is distortion, if
HARM_LAM_TOL = 0.35    # its lam also lands within this fraction of n*lam_1

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def f0_of(name: str) -> float:
    """Sample-library note name -> Hz.  The files are labelled an octave low."""
    midi = (int(name[-1]) + 2) * 12 + _NOTE_NAMES.index(name[:-1])
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


# %%
# ---------- target ----------

fs_int, x_file = wavfile.read(f"{SAMPLES}/{NOTE}-{DYN}.wav")
fs = float(fs_int)
F_NYQ = 0.45 * fs
x_file = np.asarray(x_file, dtype=np.float64)
if x_file.ndim > 1:
    x_file = x_file.mean(axis=1)
x_file = x_file / np.abs(x_file).max()

x_meas = x_file[int(MEASURE_START * fs):
                min(int((MEASURE_START + MEASURE_LEN) * fs), x_file.size)]

# One high-resolution spectrum of the loud opening serves twice: it fixes f0,
# and it gives every candidate a starting frequency.
_hires_n = 1 << 21
_n_open = int(0.8 * fs)
hires_mag = np.abs(np.fft.rfft(x_meas[:_n_open] * np.hanning(_n_open), _hires_n))
hires_f = np.fft.rfftfreq(_hires_n, 1 / fs)

f0_nominal = f0_of(NOTE)
f0 = float(hires_f[np.argmax(
    np.where(np.abs(hires_f - f0_nominal) < 0.03 * f0_nominal, hires_mag, 0.0))])
print(f"{NOTE}-{DYN}: f0 {f0:.3f} Hz (nominal {f0_nominal:.2f}), "
      f"{x_meas.size/fs:.2f} s, bin {hires_f[1]:.4f} Hz")


def peak_near(f_nominal):
    """Tallest bin within FIND_TOL of f_nominal.  The phase slope refines it."""
    band = np.abs(hires_f - f_nominal) < FIND_TOL * f_nominal
    return (float(hires_f[int(np.argmax(np.where(band, hires_mag, 0.0)))])
            if band.any() else f_nominal)


CAND_F = np.array([peak_near(r * f0) for r in RATIOS_NOMINAL])
HARM_N = np.arange(2, int(F_NYQ / f0) + 1)
HARM_F = HARM_N * f0
print(f"  {int((CAND_F < F_NYQ).sum())} of {len(RATIOS_NOMINAL)} candidates "
      f"under {F_NYQ:.0f} Hz;  {HARM_N.size} harmonics to reject against")

# %%
# ---------- isolate one line ----------
# Mix the line down to 0 Hz and low-pass.  The result is its complex envelope:
# |env| carries the amplitude over time, arg(env) the frequency error.


def demod_bandwidth(f_line, f_others):
    """Low-pass width for one line, set by its distance to the nearest other."""
    gaps = np.abs(f_others - f_line)
    gaps = gaps[gaps > 1e-6]
    if gaps.size == 0:
        return BW_MAX_HZ
    return float(np.clip(BW_FRACTION * gaps.min(), BW_MIN_HZ, BW_MAX_HZ))


def demodulate(sig, f_line, bandwidth):
    """(signal, line frequency) -> (complex envelope, its time axis)

    The Kaiser FIR is symmetric and applied with mode="same", so its group
    delay cancels and the returned time axis is the true one.  Only the
    incomplete-convolution edges are trimmed off.
    """
    n_taps, beta = sg.kaiserord(STOPBAND_DB, bandwidth / (fs / 2))
    n_taps = min(n_taps | 1, (sig.size // 3) | 1)
    taps = sg.firwin(n_taps, bandwidth, fs=fs, window=("kaiser", beta))
    mixed = sig * np.exp(-2j * np.pi * f_line * (np.arange(sig.size) / fs))
    env = np.convolve(mixed, taps, mode="same")
    edge = n_taps // 2
    return env[edge:sig.size - edge], np.arange(edge, sig.size - edge) / fs


def guard_noise(sig, f_line, bandwidth, f_others):
    """Noise level beside a line, measured through the same filter.

    The guard band must hold no known line, or it measures a neighbour and the
    "noise" lands above the signal.  So scan either side, keep the spot with the
    most clearance, and take the quieter of the two.
    """
    offsets = np.linspace(2.5, 25.0, 180) * bandwidth
    picked = []
    for side in (-1.0, +1.0):
        f_guards = f_line + side * offsets
        f_guards = f_guards[(f_guards > 2 * bandwidth)
                            & (f_guards < fs / 2 - 2 * bandwidth)]
        if f_guards.size == 0:
            continue
        clearance = np.abs(f_others[None, :] - f_guards[:, None]).min(axis=1)
        j = int(np.argmax(clearance))
        if clearance[j] >= 1.5 * bandwidth:
            picked.append(float(f_guards[j]))
    if not picked:
        return np.nan
    return min(float(np.median(np.abs(demodulate(sig, f, bandwidth)[0])))
               for f in picked)


def fit_window(mag, noise, bandwidth):
    """Which envelope samples are usable: from the first one over FIT_SNR_DB
    until the line dies.

    Dips shorter than the filter can resolve are bridged over, since they are
    nulls rather than the end of the line.  Then only the leading run counts:
    once a line has died, later excursions over the threshold are noise.
    """
    usable = mag > noise * 10 ** (FIT_SNR_DB / 20)
    bridge = max(3, int(3.0 / (2.0 * np.pi * bandwidth) * fs) | 1)
    usable = uniform_filter1d(usable.astype(float), size=bridge) > 0.4
    if not usable.any():
        return 0, 0
    first = int(np.argmax(usable))
    after = np.flatnonzero(~usable[first:])
    return first, first + (int(after[0]) if after.size else usable.size - first)


# %%
# ---------- read off f, lam and A ----------


@dataclass
class Line:
    f_guess: float         # where we looked
    f_meas: float          # where it is, from the phase slope
    amp0: float            # envelope amplitude extrapolated to t = 0
    lam: float             # decay rate, 1/s
    lam_se: float          # its standard error
    resid_db: float        # rms of the log-linear fit
    snr_db: float          # measured envelope over the local noise
    n_fit: int
    i0: int                # fit window, as indices into the envelope
    i1: int
    noise: float
    bandwidth: float

    @property
    def decay_t(self):
        """lam / se(lam).  Large means the decay is real, ~0 means flat hiss."""
        return self.lam / self.lam_se if self.lam_se > 0 else 0.0

    @property
    def t60(self):
        return 6.91 / self.lam if self.lam > 1e-9 else np.inf


def measure_line(sig, f_guess, f_others):
    """Demodulate one candidate, then fit both straight lines: log|env| gives
    lam and A, unwrap(angle(env)) gives the true frequency."""
    bandwidth = demod_bandwidth(f_guess, f_others)
    env, t_env = demodulate(sig, f_guess, bandwidth)
    noise = guard_noise(sig, f_guess, bandwidth, f_others)
    mag = np.abs(env)

    blank = Line(f_guess, f_guess, float(mag.max()), 0.0, np.inf, np.inf,
                 np.nan, 0, 0, 0, noise, bandwidth)
    if not np.isfinite(noise):
        return blank
    first, last = fit_window(mag, noise, bandwidth)
    if last - first < 8:
        return blank

    tt, yy = t_env[first:last], np.log(mag[first:last])
    slope, intercept = np.polyfit(tt, yy, 1)
    resid = yy - (intercept + slope * tt)
    slope_se = np.sqrt((resid @ resid / max(tt.size - 2, 1))
                       / ((tt - tt.mean()) ** 2).sum())

    # arg(env) turns at 2*pi*(f_true - f_guess), so the phase slope gives the
    # frequency error, far sharper than the coarse peak pick.
    phase = np.unwrap(np.angle(env[first:last]))
    f_meas = f_guess + float(np.polyfit(tt, phase, 1)[0]) / (2 * np.pi)

    # A is extrapolated back to t = 0, which is what a model wants but makes a
    # useless SNR (a steep slope on a short window extrapolates absurdly high),
    # so the SNR below uses the envelope as measured.
    return Line(f_guess, f_meas, float(np.exp(intercept)), float(-slope),
                float(slope_se), float(8.686 * np.sqrt(resid @ resid / tt.size)),
                float(20 * np.log10(np.percentile(mag[first:last], 90)
                                    / max(noise, 1e-300))),
                tt.size, first, last, noise, bandwidth)


# %%
# ---------- presence tests ----------
# The fundamental is measured first, because the harmonic test needs its lam.


def nearest_harmonic(f):
    """(n, distance in Hz) for the closest n*f0, n >= 2"""
    if HARM_N.size == 0:
        return 0, np.inf
    j = int(np.argmin(np.abs(HARM_F - f)))
    return int(HARM_N[j]), float(abs(HARM_F[j] - f))


def judge(line, lam_ref, is_reference):
    """Line -> (present, reason).  A rejection always names what it failed."""
    if line.f_guess >= F_NYQ:
        return False, "above Nyquist"
    if not np.isfinite(line.snr_db):
        return False, "nothing measurable above the noise"
    if line.snr_db < MIN_SNR_DB:
        return False, f"SNR {line.snr_db:.1f} dB"
    if line.decay_t < MIN_DECAY_T:
        return False, f"no decay (t={line.decay_t:.1f})"
    if not is_reference and lam_ref > 0:
        n, dist = nearest_harmonic(line.f_meas)
        if dist < HARM_NEAR_HZ and \
                abs(line.lam - n * lam_ref) < HARM_LAM_TOL * n * lam_ref:
            return False, f"distortion at {n}*f0 (lam ~ {n}*lam_1)"
    return True, "ok"


lines, verdicts, lam_reference = [], [], 0.0
for i in range(len(RATIOS_NOMINAL)):
    if CAND_F[i] >= F_NYQ:
        lines.append(None)
        verdicts.append((False, "above Nyquist"))
        continue
    others = np.concatenate([np.delete(CAND_F, i), HARM_F])
    line = measure_line(x_meas, CAND_F[i], others)
    if i == FUNDAMENTAL:
        lam_reference = line.lam
    lines.append(line)
    verdicts.append(judge(line, lam_reference, i == FUNDAMENTAL))


print(f"\n{'ratio':>7} {'f_meas':>10} {'off nom':>8} {'n*f0':>11} {'SNR':>6} "
      f"{'lam':>7} {'t60':>7} {'decay_t':>8} {'resid':>6} {'pts':>7}  verdict")
for i, ratio in enumerate(RATIOS_NOMINAL):
    present, reason = verdicts[i]
    line = lines[i]
    if line is None:
        print(f"{ratio:7g} {'--':>10} {'':>8} {'':>11} {'':>6} {'':>7} {'':>7} "
              f"{'':>8} {'':>6} {'':>7}  {reason}")
        continue
    n, dist = nearest_harmonic(line.f_meas)
    snr = "   n/a" if not np.isfinite(line.snr_db) else f"{line.snr_db:6.1f}"
    print(f"{ratio:7g} {line.f_meas:10.2f} {line.f_meas/(ratio*f0)-1:+8.2%} "
          f"{f'{n}:{dist:+.2f}':>11} {snr} {line.lam:7.2f} {line.t60:7.2f} "
          f"{line.decay_t:8.1f} {line.resid_db:6.2f} {line.n_fit:7d}  "
          f"{'PRESENT' if present else reason}")

PRESENT = [i for i, (ok, _) in enumerate(verdicts) if ok]
print(f"\nmeasured modes: {[f'{RATIOS_NOMINAL[i]:g}' for i in PRESENT]}")
print(f"{'ratio':>7} {'f (Hz)':>10} {'lam':>8} {'t60 (s)':>9}")
for i in PRESENT:
    print(f"{RATIOS_NOMINAL[i]:7g} {lines[i].f_meas:10.2f} "
          f"{lines[i].lam:8.2f} {6.91/lines[i].lam:9.2f}")

# %%
# ---------- envelopes ----------
# One panel per candidate: a mode is a straight line here, noise is flat.  The
# fit is drawn only across the samples it was fitted on, since extrapolating a
# steep slope over the whole axis flattens everything else out of view.

n_panel = int((CAND_F < F_NYQ).sum())
n_row = (n_panel + 1) // 2
fig, axes = plt.subplots(n_row, 2, figsize=(13, 3.4 * n_row))
flat_axes = np.atleast_1d(axes).ravel()
for ax, i in zip(flat_axes, range(n_panel)):
    line = lines[i]
    env, t_env = demodulate(x_meas, line.f_guess, line.bandwidth)
    mag_db = 20 * np.log10(np.abs(env) + 1e-30)
    ax.plot(t_env, mag_db, lw=.5, label="|envelope|")
    if np.isfinite(line.noise):
        noise_db = 20 * np.log10(max(line.noise, 1e-300))
        ax.axhline(noise_db, color="0.6", ls=":", label="local noise")
        ax.set_ylim(noise_db - 15, mag_db.max() + 8)
    if line.i1 - line.i0 > 8:
        t_win = t_env[line.i0:line.i1]
        ax.plot(t_win, 20 * np.log10(line.amp0) - 8.686 * line.lam * t_win,
                "C1", lw=1.4, label=f"fit lam={line.lam:.2f}")
        ax.axvspan(t_win[0], t_win[-1], color="C1", alpha=.07)
    present, reason = verdicts[i]
    ax.set_title(f"ratio {RATIOS_NOMINAL[i]:g} @ {line.f_guess:.1f} Hz  "
                 f"[{'PRESENT' if present else reason}]", fontsize=9)
    ax.set_xlabel("t (s)"); ax.set_ylabel("dB")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
for ax in flat_axes[n_panel:]:
    ax.set_axis_off()
fig.suptitle(f"{NOTE}-{DYN}  demodulated envelopes  "
             f"(shaded = the samples actually regressed)")
fig.tight_layout(); plt.show()

# %%
