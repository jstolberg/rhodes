# %% [markdown]
# # Staged parameter estimation for a physical model of the Rhodes
#
# The pipeline of the paper, one step after the other.  Each step is a script that also runs
# on its own from the terminal and writes its result to `results/`; this file runs it (or
# loads its cached result), shows what came out, and renders the model as it stands after
# the step next to the recording.
#
# | Step | Estimates | Script | Result |
# | --- | --- | --- | --- |
# | Model | tine, hammer, pickup | `model.py` | |
# | 1. Fundamental frequency and decay | $f_0$, $\lambda_0$ | `step1_fundamental.py` | `results/step1_fundamentals.npz` |
# | 2. Pickup and free oscillation | $p_d$, $p_o$, $\kappa$, $A_0$, $\sigma_0$ | `step2_pickup.py` | `results/step2_pickup.npz` |
# | 3. Hammer parameters | $\tau_0$, $\beta$, $c_0$ | `step3_hammer.py` | `results/step3_hammer.npz` |
# | 4. Inharmonic modes | $f_n$, $\sigma_n$, $c_n$ | `step4_modes.py` | `results/step4_modes.npz` |
# | Rendering after each step | | `synth.py` | `render/` |
#
# `RERUN` below decides per step whether it is run again or its result is loaded.  A step
# reads the results of the steps before it from `results/`, so after rerunning a step, rerun
# the ones that follow.  Step 2 takes a few minutes, the others under a minute.

# %%
import os
import runpy
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sg
from IPython.display import Audio, display

import synth
from step2_pickup import DYNS, TABLE, psi_lookup, PO_MAX, PD_RANGE

RERUN  = {1: False, 2: False, 3: False, 4: False}
LISTEN = "render/listen"          # wav files of listen()


def run_step(script, *args):
    # Run a step script as `python script args` would; its prints and plots show here.
    argv, sys.argv = sys.argv, [script, *args]
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        sys.argv = argv


def listen(step, notes=("A#3",), dyns=("f",), t_spec=1.0):
    # Recording and model after `step`: spectrograms of the first t_spec s, and both as wav
    # files in LISTEN at the same gain (the model is level-matched to the recording by
    # kappa), each with a player.
    os.makedirs(LISTEN, exist_ok=True)
    for note in notes:
        for dyn in dyns:
            rec, mod = synth.recording(note, dyn), synth.render(step, note, dyn)
            gain = 0.9 / max(np.abs(rec).max(), np.abs(mod).max())
            fig, ax = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
            for a, x, name in zip(ax, (rec, mod), ("recording", f"model after step {step}")):
                f, t, S = sg.spectrogram(x[:int(t_spec * synth.FS)], synth.FS, nperseg=2048, noverlap=1792)
                db = 10 * np.log10(S + 1e-20)
                a.pcolormesh(t, f, db, vmin=db.max() - 90, vmax=db.max(), shading="auto", cmap="magma")
                a.set_title(f"{note}-{dyn}: {name}"); a.set_xlabel("t (s)")
            ax[0].set_ylim(0, 6000); ax[0].set_ylabel("f (Hz)")
            plt.tight_layout(); plt.show()
            for x, name in ((rec, "recording"), (mod, "model")):
                path = f"{LISTEN}/{note}-{dyn}_step{step}_{name}.wav"
                wavfile.write(path, synth.FS, np.int16(np.round(gain * x * 32767)))
                print(path)
                display(Audio(filename=path))

# %% [markdown]
# ## Data
#
# 73 keys, E0 to E6 in the sample library's names (one octave below the usual names, E1 to E7
# in the paper), four dynamics each, 8 s at 48 kHz.  About 7.5 s after the onset the key is
# released and the damper stops the tine; the steps leave that part out.


# %% [markdown]
# ## Model
#
# The tine is a sum of decaying modes, the hammer a half-sine force pulse, the pickup a fixed
# surface whose flux $\Psi(x, z)$ at the tine tip is tabulated once (`model.py`, the table in
# `step2_pickup.py`).  The pickup voltage is $\varepsilon = -\mathrm d\Psi/\mathrm dt$.  Below:
# $\Psi$ over the tip's horizontal position for a few distances $p_d$, and the pulse spectrum
# $|G(f_0\tau)|$ that sets how much of each mode the hammer excites.

# %%
import jax.numpy as jnp

fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
xs = np.linspace(-6e-3, 6e-3 + PO_MAX, 800)
for pd in np.linspace(*PD_RANGE, 4):
    a = jnp.stack([jnp.asarray(xs), jnp.zeros_like(xs), jnp.full_like(xs, pd)], axis=-1)
    ax[0].plot(xs * 1e3, np.asarray(psi_lookup(a, TABLE)), label=f"$p_d$ = {pd * 1e3:.1f} mm")
ax[0].set_xlabel("tip position x (mm)"); ax[0].set_ylabel(r"$\Psi$ (a.u.)"); ax[0].set_yscale("log"); ax[0].legend(fontsize=8)
ftau = np.linspace(0.0, 4, 400) + 1e-4                  # G(x) = sin(pi x) / (pi x (1 - x^2))
G = np.sinc(ftau) / (1 - ftau**2)
ax[1].plot(ftau, 20 * np.log10(np.abs(G) + 1e-12)); ax[1].set_ylim(-60, 3)
ax[1].set_xlabel(r"$f_0\,\tau$"); ax[1].set_ylabel("dB"); ax[1].set_title(r"pulse spectrum $|G(f_0\tau)|$")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Step 1: Fundamental frequency and decay
#
# The tallest peak near the nominal frequency in a long FFT, refined by the phase slope of the
# mixed-down fundamental; $\lambda_0$ from a straight line through its level in dB, reported
# only where the level fell at least 8 dB.  After this step the model is a plain decaying sine.

# %%
if RERUN[1]:
    run_step("step1_fundamental.py")
tab, notes, dyns = synth.step1_results.load()
from step1_fundamental import f0_of
f_note = np.median(tab["f"], axis=1)
cents = 1200 * np.log2(f_note / np.array([f0_of(n) for n in notes]))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].plot(cents, ".-"); ax[0].set_ylabel("tuning (cents)"); ax[0].axhline(0, color="k", lw=0.5)
ax[0].set_xticks(range(0, len(notes), 6), notes[::6], rotation=60)
for j, d in enumerate(dyns):
    m = tab["ok"][:, j]
    ax[1].loglog(f_note[m], 6.91 / tab["lam"][m, j], "o", ms=3, label=d)
ax[1].set_xlabel("$f_0$ (Hz)"); ax[1].set_ylabel("t60 (s)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
print(f"decay measured in {tab['ok'].sum()} of {tab.size} recordings")

# %%
listen(1)

# %% [markdown]
# ## Step 2: Pickup and free oscillation
#
# One free mode $x(t) = p_o + A_0 e^{-\sigma_0 t}\sin 2\pi f_0 t$ through the pickup, fitted
# on the fundamental and its first four harmonics in 12 Hann frames (a multi-start for the
# geometry, then Adam in three stages).  $p_o$, $p_d$, $\sigma_0$ per key, $A_0$ per key and
# dynamic, one gain $\kappa$.  After this step the model has the pickup's harmonics but no attack.

# %%
if RERUN[2]:
    run_step("step2_pickup.py", "real", "1")
fit2 = synth.load(2)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
ax[0].plot(fit2["p_d"] * 1e3, ".-", label="$p_d$"); ax[0].plot(fit2["p_o"] * 1e3, ".-", label="$p_o$")
ax[0].set_ylabel("mm"); ax[0].legend()
for j, d in enumerate(DYNS):
    ax[1].semilogy(fit2["A0"][:, j] * 1e3, ".-", label=d)
ax[1].set_ylabel("$A_0$ (mm)"); ax[1].legend(fontsize=8)
ax[2].semilogy(fit2["sigma"], ".-"); ax[2].set_ylabel(r"$\sigma_0$ (1/s)")
for a in ax:
    a.set_xticks(range(0, len(fit2["notes"]), 6), fit2["notes"][::6], rotation=60)
plt.tight_layout(); plt.show()
print(f"kappa = {float(fit2['kappa']):.4e},  flagged (detuned) keys: {list(fit2['notes'][fit2['flag']])}")

# %%
listen(2)

# %% [markdown]
# ## Step 3: Hammer parameters
#
# $\tau_0$ (log-linear over the keys) and $\beta$ from how $A_0$ changes across the
# dynamics; the velocities from the bass, where $A_0 \propto v$; $c_0$ per key from the mean
# residual.  After this step the model has the hammer contact and the level across dynamics
# comes from the hammer model.

# %%
if RERUN[3]:
    run_step("step3_hammer.py")
fit3 = synth.load(3)
print(f"tau_0 {fit3['tau0'][0] * 1e3:.2f} ms ({fit3['notes'][0]}) -> {fit3['tau0'][-1] * 1e3:.3f} ms ({fit3['notes'][-1]}), "
      f"beta = {float(fit3['beta']):.3f}, velocities p, mp, mf, f = {np.round(fit3['vels'], 3)}")
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
for j, d in enumerate(DYNS):
    ax[0].semilogy(fit3["tau0"] * fit3["vels"][j] ** -fit3["beta"] * 1e3, label=rf"$\tau$({d})")
ax[0].set_ylabel("contact time (ms)"); ax[0].legend(fontsize=8)
ax[1].semilogy(fit3["c0"], ".-"); ax[1].set_ylabel("$c_0$")
for a in ax:
    a.set_xticks(range(0, len(fit3["notes"]), 6), fit3["notes"][::6], rotation=60)
plt.tight_layout(); plt.show()

# %%
listen(3)

# %% [markdown]
# ## Step 4: Inharmonic modes
#
# **A.** Modes near Gabrielli's ratios, kept only where three of the four dynamics agree on
# the frequency; their decay from p and mp, as a ratio to the fundamental's; the median over
# those keys per mode.  **B.** $c_n$ from the mode's level in the first 0.3 s, on the same
# keys, as $c_n / c_0$; the median applied to all keys.  After this step the model is complete.

# %%
if RERUN[4]:
    run_step("step4_modes.py")
fit4 = synth.load(4)
# one value per mode, the same on every key; nan: the mode was not found on any key
with np.errstate(all="ignore"), __import__("warnings").catch_warnings(action="ignore"):
    sig_ratio = np.nanmedian(fit4["sig"] / fit2["sigma"][:, None], axis=0)
    c_ratio = np.nanmedian(np.where(fit4["ok"], fit4["c"][:, 0] / fit3["c0"][:, None], np.nan), axis=0)
    print(f"{'mode':>5} {'f_n/f_0':>8} {'sigma_n/sigma_0':>16} {'c_n/c_0 (dB)':>13} {'keys in the model':>18}")
    for n, m in enumerate(fit4["modes"]):
        print(f"{m:>5} {fit4['ratios'][n]:8.1f} {sig_ratio[n]:16.1f} {20 * np.log10(c_ratio[n]):13.1f} {fit4['ok'][:, n].sum():18d}")

# %% [markdown]
# ### Across the keyboard
# The finished model on a bass, a middle and a treble key, soft and loud: where it holds
# up, where it does not, and whether the dynamics come out right.

# %%
listen(4, notes=("E1", "A#3", "E5"), dyns=("p", "f"))
# %%
