# %% [markdown]
# # Multi-resolution STFT loss
#
# reusable function, `mrstft_loss(y_hat, y_target, fs, ...)`, for
# comparing a synthetic signal against a target signal in any fitting
# pipeline:
#
# 1. `stft_magnitude`: turn one signal into a spectrogram
# 2. `build_mask`: mark which time/frequency cells count, from `t_range`/`f_range`
# 3. `resolution_loss`: compare two spectrograms within that mask
# 4. `mrstft_loss`: run 1-3 at several FFT sizes and average

# %%
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

DEFAULT_N_FFTS: tuple[int, ...] = (2048, 1024, 512, 256, 128, 64)


# %% [markdown]
# ## Parameter guide
#
# | Argument | What it is | What changes if you change it |
# |---|---|---|
# | `y_hat`, `y_target` | your two signals, same length, real-valued | `y_target` is the fixed reference; `y_hat` is what you're fitting (see "not symmetric" below) |
# | `fs` | sample rate in Hz | only used to turn `t_range`/`f_range` (seconds/Hz) into frame/bin indices |
# | `n_ffts` | tuple of FFT window sizes | more/larger sizes = finer frequency resolution but coarser time resolution, and vice versa. Each value must be `<= len(y_hat)`. **For a narrow `f_range`, make sure at least one `n_fft` is large enough to have bins inside it** -- bin spacing is `fs/n_fft`, so isolating a 20 Hz-wide band needs `n_fft > fs/20` |
# | `overlap` | frame overlap fraction (0-1), same for every resolution | higher = more frames = smoother but slower |
# | `t_range` | `(t_start, t_end)` in seconds, or `None` | restricts the comparison to that time window; `None` = whole signal |
# | `f_range` | `(f_lo, f_hi)` in Hz, or `None` | restricts the comparison to that frequency band (e.g. one isolated mode); `None` = whole spectrum |
# | `log_eps` | small positive float | floor before every `log()`, and the denominator floor in the norm ratio |
# | `sc_weight` | float, default 1.0 | weight on the spectral-convergence (linear, pitch-sensitive) term |
# | `log_weight` | float, default 1.0 | weight on the log-magnitude (level-sensitive) term |
#
#
# The function is **not symmetric**: `mrstft_loss(a, b) != mrstft_loss(b, a)`
# `y_hat` is normalised by `y_target`.
#
# See "Examples" below for what actually happens to the loss when you
# change each of these, with real numbers.
#
# ## Tuning guide
#
# | Symptom | Try |
# |---|---|
# | Pitch/frequency parameters barely move, or move in the wrong direction | raise `sc_weight` (or lower `log_weight`)  |
# | Amplitude/decay parameters barely move | raise `log_weight` -- linear magnitude alone is close to random for level |
# | Loss/gradient looks noisy, jumps around step to step | increase `overlap` (more, smoother frames), or drop the smallest `n_ffts` (they're the least frequency-selective and most prone to noise) |
# | Fitting one specific mode and it won't move | check bin coverage first -- `n_fft > fs / (bandwidth of your f_range)` -- then narrow `f_range` further around just that mode so neighbouring modes/noise don't drag the gradient elsewhere |
# | Loss dominated by quiet noise floor instead of the signal you care about | raise `log_eps` -> fewer quiet bins count as real differences (log-vs-linear amplification) |
# | Very slow (fitting many samples at once, e.g. `fitting.py` step 2's "train against all samples simultaneously") | fewer/smaller `n_ffts`, lower `overlap`, or a tighter `t_range`/`f_range` -- all directly cut the number of FFT frames computed |
#

# %% [markdown]
# ## 1. Signal to magnitude spectrogram

# %%
def hann_window(n_fft: int) -> jnp.ndarray:
    n = jnp.arange(n_fft)
    return 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * n / (n_fft - 1))


def safe_magnitude(z: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """|z| with a gradient-safe floor at z = 0 (jnp.abs's gradient is
    undefined at zero)."""
    return jnp.sqrt(z.real**2 + z.imag**2 + eps**2)


def safe_norm(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """same gradient-safe floor as safe_magnitude."""
    return jnp.sqrt(jnp.sum(x**2) + eps**2)


def stft_magnitude(x: jnp.ndarray, n_fft: int, hop_length: int) -> jnp.ndarray:
    """Magnitude STFT of a 1-D real signal -> shape (n_frames, n_fft//2+1)."""
    assert x.shape[0] >= n_fft, f"signal has {x.shape[0]} samples, shorter than n_fft={n_fft}"
    window = hann_window(n_fft)
    n_frames = 1 + (x.shape[0] - n_fft) // hop_length
    frame_starts = jnp.arange(n_frames) * hop_length
    idx = frame_starts[:, None] + jnp.arange(n_fft)[None, :]  # (n_frames, n_fft)
    frames = x[idx] * window[None, :]
    return safe_magnitude(jnp.fft.rfft(frames, axis=-1))


# %% [markdown]
# ## 2. time/frequency masking

# %%
def build_mask(
    n_frames: int,
    n_fft: int,
    hop: int,
    fs: float,
    t_range: tuple[float, float] | None,
    f_range: tuple[float, float] | None,
) -> jnp.ndarray:
    """1 where a frame/ bin is inside t_range & f_range, else 0."""
    n_bins = n_fft // 2 + 1
    mask = jnp.ones((n_frames, n_bins))

    if t_range is not None:
        frame_times = (jnp.arange(n_frames) * hop + n_fft / 2) / fs
        in_time = (frame_times >= t_range[0]) & (frame_times < t_range[1])
        mask = mask * in_time[:, None]

    if f_range is not None:
        bin_freqs = jnp.fft.rfftfreq(n_fft, d=1.0 / fs)
        in_freq = (bin_freqs >= f_range[0]) & (bin_freqs < f_range[1])
        mask = mask * in_freq[None, :]

    return mask


# %% [markdown]
# ## 3. Compare two spectrograms within a mask
#
# Two terms, following Yamamoto et al. (2019) / the MSSTFT setup evaluated
# by Turian & Henry (2020, arXiv:2012.04572): **spectral convergence**
# (linear magnitude, mainly pitch-sensitive) and **log-magnitude L1**
# (mainly level-sensitive). Both are averaged only where
# `mask == 1`

# %%
def resolution_loss(
    mag_hat: jnp.ndarray,
    mag_tgt: jnp.ndarray,
    mask: jnp.ndarray,
    log_eps: float,
    sc_weight: float,
    log_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (loss for this one resolution, whether the mask hit anything)."""
    mag_hat, mag_tgt = mag_hat * mask, mag_tgt * mask
    n_active = jnp.sum(mask)

    spectral_convergence = safe_norm(mag_tgt - mag_hat) / (safe_norm(mag_tgt) + log_eps)
    log_diff = jnp.abs(jnp.log(mag_hat + log_eps) - jnp.log(mag_tgt + log_eps)) * mask
    log_l1 = jnp.sum(log_diff) / jnp.maximum(n_active, 1.0)

    loss = sc_weight * spectral_convergence + log_weight * log_l1
    return loss, n_active > 0.0


# %% [markdown]
# ## 4. calculating mrstft
#
# Resolutions where the mask hit zero cells (see the `n_ffts` row in the
# parameter guide) are excluded from the average rather than counted as a
# near-perfect match -- otherwise a narrow `f_range` would silently shrink
# the loss again, this time across resolutions instead of within one.

# %%
def mrstft_loss(
    y_hat: jnp.ndarray,
    y_target: jnp.ndarray,
    fs: float,
    n_ffts: Sequence[int] = DEFAULT_N_FFTS,
    overlap: float = 0.75,
    t_range: tuple[float, float] | None = None,
    f_range: tuple[float, float] | None = None,
    log_eps: float = 1e-4,
    sc_weight: float = 1.0,
    log_weight: float = 1.0,
) -> jnp.ndarray:
    """Multi-resolution STFT loss between two equal-length 1-D signals.
    See the parameter guide markdown cell above for what each argument does.
    """
    assert y_hat.shape == y_target.shape, (
        f"y_hat and y_target must have the same length, got {y_hat.shape} vs {y_target.shape}"
    )

    total, n_used = 0.0, 0.0
    for n_fft in n_ffts:
        hop = max(1, int(round(n_fft * (1.0 - overlap))))
        mag_hat = stft_magnitude(y_hat, n_fft, hop)
        mag_tgt = stft_magnitude(y_target, n_fft, hop)
        mask = build_mask(mag_hat.shape[0], n_fft, hop, fs, t_range, f_range)
        loss, has_signal = resolution_loss(mag_hat, mag_tgt, mask, log_eps, sc_weight, log_weight)
        total = total + jnp.where(has_signal, loss, 0.0)
        n_used = n_used + jnp.where(has_signal, 1.0, 0.0)

    return total / jnp.maximum(n_used, 1.0)


# %% [markdown]
# ## Examples: what changes when parameters change
#
# Same two signals throughout a 440 Hz target and a 442 Hz + slightly
# louder `y_hat`, only `mrstft_loss()` changes.

# %%
fs = 48000.0
t = jnp.arange(int(fs * 0.5)) / fs  # 0.5 s


def tone(f_hz: float, amp: float = 0.5) -> jnp.ndarray:
    return amp * jnp.sin(2.0 * jnp.pi * f_hz * t)


target = tone(440.0)
y_hat = tone(442.0, amp=0.55)  # slightly sharp and slightly louder

print("full signal, default settings:")
print(f"  loss = {float(mrstft_loss(y_hat, target, fs)):.4f}\n")

print("t_range: only look at the first 100 ms")
print(f"  loss = {float(mrstft_loss(y_hat, target, fs, t_range=(0.0, 0.1))):.4f}\n")

print("f_range: only look at 400-500 Hz (around the fundamental)")
print(f"  loss = {float(mrstft_loss(y_hat, target, fs, f_range=(400.0, 500.0))):.4f}\n")

print("f_range: only look at 20000-20100 Hz (far from any real content)")
print(f"  loss = {float(mrstft_loss(y_hat, target, fs, f_range=(20000.0, 20100.0))):.4f}  "
      "(smaller, but not ~0 -- log-magnitude comparison amplifies tiny relative\n"
      "  differences even in near-silent leakage, a known quirk of log-scale losses)\n")

print("n_ffts: fewer, smaller resolutions -> coarser frequency resolution, faster")
print(f"  loss = {float(mrstft_loss(y_hat, target, fs, n_ffts=(256, 128))):.4f}\n")

print("n_ffts: single large resolution -> can't resolve 442 vs 440 Hz as well")
print(f"  loss = {float(mrstft_loss(y_hat, target, fs, n_ffts=(64,))):.4f}\n")

print("overlap: fewer frames (less overlap) -> noisier, slightly different value")
print(f"  loss = {float(mrstft_loss(y_hat, target, fs, overlap=0.0)):.4f}\n")

print("sc_weight=0: switch off the pitch-sensitive term -> frequency gradient shrinks")
print(f"  loss = {float(mrstft_loss(y_hat, target, fs, sc_weight=0.0)):.4f}")
print(f"  d(loss)/d(freq) with sc_weight=1 (default): {float(jax.grad(lambda f: mrstft_loss(tone(f), target, fs))(442.0)): .3e}")
print(f"  d(loss)/d(freq) with sc_weight=0            : {float(jax.grad(lambda f: mrstft_loss(tone(f), target, fs, sc_weight=0.0))(442.0)): .3e}\n")

print("log_weight=0: switch off the level-sensitive term -> gain gradient shrinks")
print(f"  d(loss)/d(gain) with log_weight=1 (default): {float(jax.grad(lambda g: mrstft_loss(g * target, target, fs))(1.5)): .3e}")
print(f"  d(loss)/d(gain) with log_weight=0           : {float(jax.grad(lambda g: mrstft_loss(g * target, target, fs, log_weight=0.0))(1.5)): .3e}")
