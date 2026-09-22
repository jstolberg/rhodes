# %% [markdown]
# # Step 2 example on a single note
#
# Goal: fit the pickup parameters $p_d, p_o, m, a$
# together with a free oscillation amplitude $A_0$ and decay $\sigma_0$ per note. Only the H0-H6 band is used. Inharmonic modes
# are assumed to sit outside it, so this band contains only the pickup's
# own distortion of the fundamental. That means pickup can
# be fit without knowing something about hammer physics or inharmonic
# modes
#
#
# Assumes: folder contains a folder with the samples namded `samples/`.
#
# ## Parameters fit in this step
#
# | Parameter | Meaning | Scope |
# |---|---|---|
# | $A_0, \sigma_0$ | oscillation amplitude / decay | per note x dynamic |
# | $p_d, p_o$ | pickup distance / offset | per note |
# | $a, m$ | pickup plateau width / slope | global |
# | $\kappa$ | single multiplicative factor | global |
#
# ## Blocks
# 1. Data: load the recording, extract every harmonic's envelope, pick the usable window -> e.g. mode_measurement
# 2. Model: minimal forward model for the fundamental (without hammer physics)
# 3. Loss: compare two envelopes
# 4. Fit: in three stages
#

# %%
# !pip install equinox optax

# %%
# %%capture
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d
import optax
import importlib
importlib.invalidate_caches()
from model import z_trapz, make_surface, psi_table, psi_lookup

jax.config.update("jax_enable_x64", True)

NOTE, DYN = "A3", "mf"   # the one recording
H_MAX = 4                # how many harmonics get fitted

# %% [markdown]
# ## Block 1 Data: load, extract, window
#
# The data gets prepared for using in the model. This will be done by `mode_measurement.py` in the future
#
# - `harmonic_envelope()`: coherent demodulation, extracts one harmonic's
#   complex envelope (n corresponds to the number of harmonic e.g. n=0 H0)
# - `select_window()`: start after the attack transient (Gabrielli et
#   al.: 300 ms), ends where the signal drops into the noise floor, S4+S5 from `fitting.py`'s table
# - `noise_start_t`: estimated once per recording from H0 (the
#   slowest-decaying harmonic), used both as the noise reference and to
#   crop the signal before demodulating (skips convolving a long FIR
#   filter against several seconds of pure trailing noise)

# %%
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def f0_of(name):
    midi = (int(name[-1]) + 2) * 12 + _NOTE_NAMES.index(name[:-1])
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def load_wav(path):
    fs, x = wavfile.read(path)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return float(fs), x / (np.abs(x).max() + 1e-12)


def design_lowpass(bandwidth_hz, fs, stopband_db=60.0):
    n_taps, beta = signal.kaiserord(stopband_db, bandwidth_hz / (fs / 2))
    n_taps |= 1
    taps = signal.firwin(n_taps, bandwidth_hz, fs=fs, window=("kaiser", beta))
    return jnp.asarray(taps, dtype=jnp.complex128)


def harmonic_envelope(x, t, f0, n, taps):
    """complex envelope of harmonic n (n=0 -> H0). Factor 2 for amplitude
    compensation"""
    freq = (n + 1) * f0
    mixed = 2.0 * x * jnp.exp(-1j * 2 * jnp.pi * freq * t)
    env = jnp.convolve(mixed, taps, mode="same")
    edge = taps.shape[0] // 2
    return env[edge:-edge], t[edge:-edge]


def h0_safe_noise_start(mag0, t_env0, fit_dur=1.0, target_db=50.0):
    """fit H0 early decay, extrapolate to where it would be target_db
    below its start"""
    onset_idx = int(np.argmax(20*np.log10(mag0/mag0.max()+1e-12) > -6.0))
    dt = t_env0[1] - t_env0[0]
    i0, i1 = onset_idx, min(onset_idx + int(fit_dur / dt), mag0.size)

    slope, _ = np.polyfit(t_env0[i0:i1], np.log(mag0[i0:i1] + 1e-12), 1)
    lam = -slope
    if lam <= 0:
        return t_env0[-1] * 0.8

    noise_start_t = t_env0[i0] + target_db / (8.686 * lam)
    return noise_start_t if noise_start_t < t_env0[-1] else t_env0[-1] * 0.8


def noise_floor_of(mag, t_env, noise_start_t):
    return max(np.median(mag[t_env >= noise_start_t]), 1e-12)


def select_window(mag, t_env, noise_start_t, thresh_db=-6.0, guard=0.3,
                   snr_db=10.0, bridge_s=0.05):
    """start = end of attack transient, end = where SNR drops below
    threshold, end of S4+S5."""
    mag_db = 20 * np.log10(mag / mag.max() + 1e-12)
    onset_idx = int(np.argmax(mag_db > thresh_db))
    noise = noise_floor_of(mag, t_env, noise_start_t)

    dt = t_env[1] - t_env[0]
    bridge = max(3, int(bridge_s / dt) | 1)
    usable = uniform_filter1d((mag > noise * 10**(snr_db/20)).astype(float), size=bridge) > 0.4

    start_idx = int(np.argmax(t_env >= t_env[onset_idx] + guard))
    after = np.flatnonzero(~usable[start_idx:])
    end_idx = start_idx + (int(after[0]) if after.size else usable.size - start_idx)

    keep = np.zeros_like(mag, dtype=bool)
    keep[start_idx:end_idx] = True
    return keep

# %%
fs, x_full = load_wav(f"/samples/{NOTE}-{DYN}.wav")
t_full = jnp.arange(x_full.size) / fs
f0 = f0_of(NOTE)
print(f"{NOTE}-{DYN}: fs={fs:.0f} Hz, duration={x_full.size/fs:.2f} s, f0={f0:.2f} Hz")

bandwidth = float(np.clip(0.4 * f0, 5.0, 60.0))
taps = design_lowpass(bandwidth, fs)

# H0 on the full signal first -- needed to find noise_start_t before cropping
env0_full, t_env0_full = harmonic_envelope(jnp.asarray(x_full), t_full, f0, 0, taps)
noise_start_t = h0_safe_noise_start(np.asarray(jnp.abs(env0_full)), np.asarray(t_env0_full))

margin_s = float(taps.shape[0]) / fs   # filter edge-trim margin
crop_n = min(x_full.size, int((noise_start_t + margin_s) * fs))
x_real, t = np.asarray(x_full[:crop_n]), t_full[:crop_n]
print(f"noise_start_t = {noise_start_t:.2f} s, cropped to {crop_n/x_full.size*100:.0f}% of the recording")

env_real_list, keep_list = [], []
fig, axes = plt.subplots(H_MAX, 1, figsize=(9, 2.2*H_MAX), sharex=True)
for n, ax in enumerate(axes):
    env_r, t_env = harmonic_envelope(jnp.asarray(x_real), t, f0, n, taps)
    mag_r = np.asarray(jnp.abs(env_r))
    keep_n = select_window(mag_r, np.asarray(t_env), noise_start_t)
    noise_n = noise_floor_of(mag_r, np.asarray(t_env), noise_start_t)

    env_real_list.append(env_r)
    keep_list.append(jnp.asarray(keep_n))
    print(f"H{n}: window = [{np.asarray(t_env)[keep_n][0]:.2f}, {np.asarray(t_env)[keep_n][-1]:.2f}] s, "
          f"{keep_n.sum()} points, noise floor = {20*np.log10(noise_n):.1f} dB")

    mag_db = 20*np.log10(mag_r + 1e-12)
    ax.plot(np.asarray(t_env), mag_db, lw=.4, color="0.6")
    ax.plot(np.asarray(t_env)[keep_n], mag_db[keep_n], color="C0")
    ax.axhline(20*np.log10(noise_n), color="r", ls=":", lw=1)
    ax.set_ylabel(f"H{n}\n[dB]")
axes[-1].set_xlabel("t (s)")
fig.suptitle(f"{NOTE}-{DYN}: windowed envelopes (blue) + noise floor (red)")
fig.tight_layout()
plt.show()

sample = dict(t=t, f0=f0, H_max=H_MAX, taps=taps, env_real=env_real_list, keep=keep_list)

# %% [markdown]
# ## Block 2 Model: `alpha_fundamental()`
#
# `model.alpha()` depends on the hammer contact, so we need a more minimal model for this step, a single, freely decaying sinusoid from $t=0$. It still produces harmonics in the pickup output, assumed from nonlinearity produced by `psi_lookup`.
#
# $\kappa$ is an unknon voltage gain, that is separate from $A_0$ which should have a physical, plausible range (maybe comparable to $a$?). from H0 alone, only the product $\kappa \cdot A_0$ is determined, $\kappa$ stays unconstrained and positive, since it has no physical range.

# %%
def alpha_fundamental(t, p):
    """p needs: A0, sigma0, p_o, p_d, f0 (f0 fixed, not fit)."""
    x = p['A0'] * jnp.exp(-p['sigma0']*t) * jnp.sin(2*jnp.pi*p['f0']*t) + p['p_o']
    return jnp.array([x, 0.0, p['p_d']])


def build_table_fundamental(a, m, A0_max, po_max, p_d, N=128, r_max=3e-3, pad=1.2, **table_kw):
    pts, w = make_surface(partial(z_trapz, a=a, m=m), N=N, r_max=r_max)
    return psi_table({'p_d': p_d}, pts, w, disp_max=pad*A0_max, po_max=po_max, **table_kw)


@jax.jit
def epsilon_fundamental(t, p, table):
    """Same structure as model.epsilon(), through alpha_fundamental
    instead of the full hammer-coupled alpha()."""
    psi_of_t = lambda tt: psi_lookup(alpha_fundamental(tt, p), table)
    dpsi = lambda tt: jax.jvp(psi_of_t, (tt,), (jnp.ones_like(tt),))[1]
    return -jax.vmap(dpsi)(t)


# physical plausible A0 range
# guessed ~0.5mm so the oscillation actually engages the nonlinearity
A0_MIN, A0_MAX = 2e-5, 5e-4          # 20 to 500 micrometers
PO_MAX = 3.5e-3                       # a bit inside the table's po_max=4e-3
PD_MIN, PD_MAX = 1.5e-3, 3.5e-3       # p_d interpolation range
DISP_MAX_STATIC = 1.2 * A0_MAX        # fixed table promise, used by every stage
TABLE_FAST = dict(N=64, n=1024)       # cheap resolution for tables rebuilt every step

positive = jax.nn.softplus   # unconstrained-positive: sigma0, kappa

def bounded(raw, lo, hi):
    """sigmoid-bounded parameter, keeps the optimizer inside [lo, hi]
    (a physical or table-imposed range), so it can not run off and
    hit a silent extrapolation edge. Used for A0, p_o, p_d."""
    return lo + (hi - lo) * jax.nn.sigmoid(raw)

def unbound_init(x, lo, hi):
    """Inverse of bounded() -- turn a physical guess into the raw
    (pre-sigmoid) starting value."""
    frac = (x - lo) / (hi - lo)
    return jnp.log(frac) - jnp.log(1 - frac)

def unpack_note_params(raw):
    """raw (unconstrained) params -> physical A0/sigma0/p_o/p_d, all
    in one place (used by both stage 1 and stage 3, which fit all five)."""
    return dict(
        A0=bounded(raw['A0_raw'], A0_MIN, A0_MAX),
        sigma0=positive(raw['sigma0_raw']),
        p_o=bounded(raw['p_o_raw'], -PO_MAX, PO_MAX),
        p_d=bounded(raw['p_d_raw'], PD_MIN, PD_MAX),
    )

# %%
a_guess, m_guess = 0.5e-3, 0.4     # guessed pickup geometry, not fitted yet
sigma0_guess = 2.0
A0_phys_guess = float(jnp.sqrt(A0_MIN * A0_MAX))   # geometric avergae of the range

table_demo = build_table_fundamental(a_guess, m_guess, A0_max=DISP_MAX_STATIC/1.2, po_max=4e-3, p_d=2.5e-3)
p_probe = dict(A0=A0_phys_guess, sigma0=sigma0_guess, p_o=0.0, p_d=2.5e-3, f0=f0)
env_probe0, _ = harmonic_envelope(epsilon_fundamental(sample['t'], p_probe, table_demo), sample['t'], f0, 0, sample['taps'])

# one-off scale estimate for kappa: not a fit, just a starting point
# (demodulation+lowpass is linear, so scaling the envelope afterwards is the same to scaling the raw signal beforehand)
kappa_guess = float(jnp.abs(sample['env_real'][0]).max()) / max(float(jnp.abs(env_probe0).max()), 1e-12)
print(f"A0_phys_guess = {A0_phys_guess:.2e} m, kappa_guess = {kappa_guess:.4g}")

env_synth0 = env_probe0 * kappa_guess
plt.figure(figsize=(9, 3))
plt.plot(np.asarray(t_env), 20*np.log10(np.abs(sample['env_real'][0])+1e-12), label="real")
plt.plot(np.asarray(t_env), 20*np.log10(np.abs(env_synth0)+1e-12), label="synthetic (guess, kappa-scaled)")
plt.xlabel("t (s)"); plt.ylabel("H0 [dB]"); plt.legend()
plt.title(f"{NOTE}-{DYN}: H0 envelope, real vs. synthetic (unfit parameters)")
plt.show()

# %% [markdown]
# ## Block 3 Loss: `envelope_distance()`
#
# Plain MSE on $\log|\text{env}|$. with `valid`/`min_points` it is possible to exclued harmonics, because of beating between two
# close modes or too short a window for example

# %%
def envelope_distance(env_synth, env_real, keep_mask, valid=1.0, min_points=200):
    n_valid = jnp.sum(keep_mask)
    weight = valid * (n_valid >= min_points)
    residual = jnp.where(keep_mask, jnp.log(jnp.abs(env_synth)+1e-8) - jnp.log(jnp.abs(env_real)+1e-8), 0.0)
    return jnp.sum(residual**2) / jnp.maximum(n_valid, 1), weight

# %%
loss0, weight0 = envelope_distance(env_synth0, sample['env_real'][0], sample['keep'][0])
print(f"H0 loss (unfit guess) = {float(loss0):.4f}, weight = {float(weight0)}")

# %% [markdown]
# ## Block 4 Fit in stages
#
# **Stage 1:** $A_0, \sigma_0, \phi_0, p_o, p_d, \kappa$ from H0 alone,
# $a,m$ fixed at the initial guess, table built once, not per step.
#
# **Stage 2:** $a, m$ from H1..H_max-1 (the harmonics that only exist
# because of the pickup nonlinearity), stage 1's result held fixed.
# Table now rebuilds every step (in a less computationally complex way, with `TABLE_FAST`).
#
# **Stage 3:** a fine-tune of all 8 parameters together, low LR,
# starting from stage 1+2's result. necessary for a single note, $a,m$ also shape H0's amplitude a little, so stage 1's fit
# carries some bias from the still-rough $a,m$ guess it used
#
# **On `jax.jit` and `functools.partial`:** `run_optax()` compiles the
# full loss+grad+update step. Anything feeding `psi_table` static
# arguments (`disp_max`, `H_max` inside a python `range()`) must stay a
# genuine python constant, never a value that's part of the traced
# pytree, that is why `H_max` is bound in via `partial()` before
# `run_optax()` is called, and not passed as an extra argument

# %%
def _weighted_loss(eps_hat, sample, ns):
    total, weight_sum = 0.0, 0.0
    for n in ns:
        env_hat, _ = harmonic_envelope(eps_hat, sample['t'], sample['f0'], n, sample['taps'])
        loss_n, w_n = envelope_distance(env_hat, sample['env_real'][n], sample['keep'][n])
        total += w_n * loss_n
        weight_sum += w_n
    return total / jnp.maximum(weight_sum, 1e-8)


def stage1_loss(params_raw, sample, table):
    p_sample = dict(**unpack_note_params(params_raw), f0=sample['f0'])
    eps_hat = positive(params_raw['kappa_raw']) * epsilon_fundamental(sample['t'], p_sample, table)
    return _weighted_loss(eps_hat, sample, [0])


def stage2_loss(ab_raw, fixed, sample, H_max):
    a, m = positive(ab_raw['a_raw']), positive(ab_raw['m_raw'])
    table = build_table_fundamental(a, m, A0_max=DISP_MAX_STATIC/1.2, po_max=4e-3, p_d=fixed['p_d'], **TABLE_FAST)
    p_sample = dict(**fixed, f0=sample['f0'])   # extra 'kappa' key is harmless, alpha_fundamental ignores it
    eps_hat = fixed['kappa'] * epsilon_fundamental(sample['t'], p_sample, table)
    return _weighted_loss(eps_hat, sample, range(1, H_max))


def joint_loss(params_raw, sample, H_max):
    a, m = positive(params_raw['a_raw']), positive(params_raw['m_raw'])
    note_p = unpack_note_params(params_raw)
    table = build_table_fundamental(a, m, A0_max=DISP_MAX_STATIC/1.2, po_max=4e-3, p_d=note_p['p_d'],
                                     n_pd=16, pd_range=(PD_MIN, PD_MAX), **TABLE_FAST)
    p_sample = dict(**note_p, f0=sample['f0'])
    eps_hat = positive(params_raw['kappa_raw']) * epsilon_fundamental(sample['t'], p_sample, table)
    return _weighted_loss(eps_hat, sample, range(H_max))


def run_optax(loss_fn, params0, *extra_args, n_steps=200, lr=1e-2, label=""):
    opt = optax.adam(lr)
    opt_state = opt.init(params0)

    @jax.jit
    def step(params, opt_state, *extra):
        loss, grads = jax.value_and_grad(loss_fn)(params, *extra)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    params, history = params0, []
    for i in range(n_steps):
        params, opt_state, loss = step(params, opt_state, *extra_args)
        history.append(float(loss))
        if i % 20 == 0:
            print(f"[{label}] step {i:4d}  loss {loss:.4f}")
    return params, history

# %%
# stage 1 table built once, with n_pd/pd_range since p_d is fit here
table_fixed = build_table_fundamental(a_guess, m_guess, A0_max=DISP_MAX_STATIC/1.2, po_max=4e-3, p_d=2.5e-3,
                                       n_pd=16, pd_range=(PD_MIN, PD_MAX))

stage1_params0 = dict(
    A0_raw=unbound_init(A0_phys_guess, A0_MIN, A0_MAX),
    sigma0_raw=jnp.log(jnp.expm1(sigma0_guess)),
    p_o_raw=unbound_init(1.2e-3, -PO_MAX, PO_MAX),
    p_d_raw=unbound_init(2.5e-3, PD_MIN, PD_MAX),
    kappa_raw=jnp.log(jnp.expm1(kappa_guess)),
)
stage1_params, stage1_history = run_optax(stage1_loss, stage1_params0, sample, table_fixed,
                                           n_steps=200, lr=1e-2, label="stage1")

fixed_from_stage1 = dict(**unpack_note_params(stage1_params), kappa=positive(stage1_params['kappa_raw']))
print("stage 1 result:", {k: float(v) for k, v in fixed_from_stage1.items()})

# %%
# stage 2 a, m from H1..H_max-1
stage2_loss_bound = partial(stage2_loss, H_max=H_MAX)
stage2_params0 = dict(a_raw=jnp.log(jnp.expm1(a_guess)), m_raw=jnp.log(jnp.expm1(m_guess)))
stage2_params, stage2_history = run_optax(stage2_loss_bound, stage2_params0, fixed_from_stage1, sample,
                                           n_steps=200, lr=1e-2, label="stage2")
print("stage 2 result: a =", float(positive(stage2_params['a_raw'])),
      " m =", float(positive(stage2_params['m_raw'])))

# %%
# stage 3 fine-tune, all 8 parameters, low LR
joint_loss_bound = partial(joint_loss, H_max=H_MAX)
joint_params0 = dict(**stage1_params, **stage2_params)
joint_params, joint_history = run_optax(joint_loss_bound, joint_params0, sample,
                                         n_steps=400, lr=1e-3, label="stage3")

plt.figure(figsize=(6, 3))
bounds = np.cumsum([0, len(stage1_history), len(stage2_history), len(joint_history)])
for (a, b), h, lbl in zip(zip(bounds, bounds[1:]), [stage1_history, stage2_history, joint_history],
                           ["stage 1 (H0)", "stage 2 (a,m)", "stage 3 (joint)"]):
    plt.plot(range(a, b), h, label=lbl)
plt.xlabel("step (concatenated across stages)"); plt.ylabel("loss"); plt.legend()
plt.title(f"{NOTE}-{DYN}: staged fit")
plt.show()

final = dict(**unpack_note_params(joint_params), kappa=positive(joint_params['kappa_raw']),
             a=positive(joint_params['a_raw']), m=positive(joint_params['m_raw']))
print("final:", {k: float(v) for k, v in final.items()})

# %% [markdown]
#
