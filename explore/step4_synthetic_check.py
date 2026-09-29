# %% [markdown]
# # Synthetic check of step 4: does the measurement find a known A_n / A_0?
#
#     PYTHONPATH=. python explore/step4_synthetic_check.py      (from the repo root)
#     -> cache/step4_synthetic_check.npz, plots/step4_synthetic_check.png
#
# Not part of the chain.  Parts A and B of `step4_modes.py` run on **synthetic recordings with
# a known truth**:
#
# - rendered by the full model (hammer, fundamental, modes, pickup; as `synth.py`) for every
#   key and dynamic, 8 s like the recordings, with $f_n$ and $\sigma_n$ of step 4 B;
# - a **known $A_n / A_0$ per key**: step 4's $a_n$ times a random factor within
#   $\pm$`SPREAD_DB` (about the scatter between the real keys), the same for every dynamic,
#   turned into $c_n$ by step 4's `c_of`.  Where `c_of` caps $c_n$ at a pulse null, the truth
#   is what the mode then gets;
# - white noise as strong as the noise floor of the matching recording (median of its spectrum
#   between $5 f_0$ and $45 f_0$ in the first 0.3 s after $t_0$).
#
# Part A runs unchanged, only the sample folder swapped.  Part B's `ratio_a` runs unchanged,
# with the recordings swapped for the synthetic ones and step 2's $A_0$ for the true $A_0$ of
# the synthetic fundamental (step 2 has its own check), so this checks the mode measurement
# alone.  Reported: how often part A finds the modes, the error of $A_n / A_0$ below and
# above `F0_BASS` = 150 Hz, and whether part A's clear keys are the ones with the stronger modes
# (the selection behind the median).  The synthetic wavs are kept in `cache/syn4/` and
# rendered again only when the truth changes.

# %%
import os
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal.windows import blackmanharris

import step4_modes as s4                                   # runs step 4 once (about 1 minute)
from model import calc_tau, epsilon, hammer2free
from step2_pickup import DYNS, T_START, TABLE, load_wave, onset
from synth import DUR, FS, LEAD

SYN_DIR   = "cache/syn4"
SPREAD_DB = 15.0        # true A_n / A_0 per key: a_n times up to +- this
F0_BASS   = 150.0       # Hz: below, 7.1 f_0 lies within HARM_HZ of 7 f_0
DB = 20 / np.log(10)

notes, names, ok, vels = s4.notes, s4.names, s4.ok, s4.vels
f0, sig0, c0, fit2, fit3 = s4.f0, s4.sig0, s4.c0, s4.fit2, s4.fit3
M, N = ok.shape


def noise_std(x, fs, f_0):
    """Std of white noise with the same spectral floor as x: median |X|^2 between 5 f_0 and
    45 f_0 (at most 20 kHz) in the first 0.3 s after t_0.  For white noise of std s, |X|^2 is
    exponential with mean s^2 sum(w^2), so its median is ln 2 times that."""
    a0 = onset(x) + int(T_START * fs)
    seg = x[a0:a0 + int(0.3 * fs)]
    w = blackmanharris(len(seg))
    P = np.abs(np.fft.rfft(seg * w)) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / fs)
    band = (f > 5 * f_0) & (f < min(45 * f_0, 20e3))
    return np.sqrt(np.median(P[band]) / (np.log(2) * np.sum(w ** 2)))


def A0_syn(i, j):
    """Free amplitude of the synthetic fundamental at t_0 = T_START after contact."""
    v = float(vels[j])
    tau = float(calc_tau(v, dict(tau_0=fit3["tau0"][i], beta=s4.beta)))
    return abs(float(hammer2free(f0[i], c0[i], v, tau)[0])) * np.exp(-sig0[i] * (T_START - tau))


def load_syn(note, dyn):
    fs, x = wavfile.read(f"{SYN_DIR}/{note}-{dyn}.wav")
    return np.asarray(x, dtype=float), float(fs)


# %%
# ---------- 1. the truth and the synthetic recordings ----------
rng = np.random.default_rng(0)
a_true = np.where(ok, s4.a[None, :] * 10 ** (rng.uniform(-SPREAD_DB, SPREAD_DB, ok.shape) / 20), np.nan)
a_step4 = s4.a
c_syn, short = np.zeros((M, len(DYNS), N)), np.ones((M, len(DYNS), N))
for i in range(M):
    s4.a = a_true[i]                                       # c_of reads a_n from step 4's a
    for j in range(len(DYNS)):
        c_syn[i, j], short[i, j] = s4.c_of(i, j)
s4.a = a_step4
truth = np.where(ok[:, None, :], a_true[:, None, :] * short, np.nan)          # A_n / A_0 in the wavs

os.makedirs(SYN_DIR, exist_ok=True)
stamp = f"{SYN_DIR}/truth.npz"
fresh = not (os.path.exists(stamp) and np.allclose(np.load(stamp)["c"], c_syn))
t = jnp.arange(int(DUR * FS)) / FS
for i, note in enumerate(notes):
    k = ok[i]
    for j, dyn in enumerate(DYNS):
        path = f"{SYN_DIR}/{note}-{dyn}.wav"
        if os.path.exists(path) and not fresh:
            continue
        p = dict(c=jnp.r_[c0[i], c_syn[i, j][k]], lam=jnp.r_[sig0[i], s4.sig_n[i][k]],
                 f_modes=jnp.r_[f0[i], s4.f_n[i][k]], tau_0=float(fit3["tau0"][i]), beta=s4.beta,
                 p_d=float(fit2["p_d"][i]), p_o=float(fit2["p_o"][i]))
        x = s4.kappa * np.asarray(epsilon(t, p, TABLE, vel=float(vels[j])))
        x = np.concatenate([np.zeros(int(LEAD * FS)), x])[:int(DUR * FS)]
        s = noise_std(*load_wave(note, dyn), float(f0[i]))
        wavfile.write(path, int(FS), (x + s * rng.standard_normal(len(x))).astype(np.float32))
    print(f"{note:>4}", end=" ", flush=True)
np.savez(stamp, c=c_syn)
print(f"\nsynthetic recordings in {SYN_DIR}/")

# %%
# ---------- 2. part A on the synthetic recordings ----------
src = open("step4_modes.py", encoding="utf-8").read().split("# ## B.")[0]
assert 'f"Samples/{note}-{dyn}.wav"' in src
syn = {}
exec(src.replace('f"Samples/{note}-{dyn}.wav"', f'f"{SYN_DIR}/{{note}}-{{dyn}}.wav"'), syn)
rows = syn["rows"]
clear = np.zeros(ok.shape, bool)
for x in rows:
    clear[notes.index(x["note"]), names.index(x["mode"])] = True
print("\npart A on synthetic data:")
for n, m in enumerate(names):
    if not ok[:, n].any():
        continue
    R = [x for x in rows if x["mode"] == m]
    rf = np.median([x["ratio_f"] for x in R]) if R else np.nan
    rs = np.median([x["sig_n"] / sig0[notes.index(x["note"])] for x in R]) if R else np.nan
    print(f"  {m}: clear on {len(R):2d} keys (real: {sum(x['mode'] == m for x in s4.rows)}, in the model: "
          f"{int(ok[:, n].sum())});  f_n/f_0 found {rf:.3f}, true {s4.mu[n]};  "
          f"sigma_n/sigma_0 found {rs:.1f}, true {s4.r[n]:.1f}")

# %%
# ---------- 3. part B's A_n / A_0 on the clear synthetic modes ----------
# ratio_a unchanged: it reads the recording with step 4's load_wave and A_0 from step 4's fit2,
# so both are swapped for the synthetic ones while it runs
fit2_syn = {k: fit2[k] for k in fit2.files}
fit2_syn["A0"] = np.array([[A0_syn(i, j) for j in range(len(DYNS))] for i in range(M)])
s4.load_wave, s4.fit2 = load_syn, fit2_syn
a_meas = np.full(truth.shape, np.nan)
for x in rows:
    i, n = notes.index(x["note"]), names.index(x["mode"])
    a_meas[i, :, n] = [s4.ratio_a(x, j) for j in range(len(DYNS))]
s4.load_wave, s4.fit2 = load_wave, fit2

with np.errstate(divide="ignore", invalid="ignore"):
    err = DB * np.log(a_meas / truth)                                          # (keys, dyn, N), dB
bass = f0 < F0_BASS
iqr = lambda v: np.nanpercentile(v, 75) - np.nanpercentile(v, 25)
print(f"\nA_n/A_0 measured - true (dB), clear synthetic modes x dynamics:")
print(f"  all: median {np.nanmedian(err):+.1f}, IQR {iqr(err):.1f}, |error| median "
      f"{np.nanmedian(np.abs(err)):.1f} ({np.isfinite(err).sum()} cells)")
for label, sel in ((f"below {F0_BASS:.0f} Hz", bass), (f"{F0_BASS:.0f} Hz and up", ~bass)):
    e = err[sel]
    print(f"  {label}: " + (f"median {np.nanmedian(e):+.1f}, |error| median {np.nanmedian(np.abs(e)):.1f} "
                            f"({np.isfinite(e).sum()} cells)" if np.isfinite(e).any() else "no clear synthetic mode"))

# per mode as in part B (log mean over the dynamics, median over the clear keys); the true
# median on the clear keys against the one on all keys shows the selection of the clear keys
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    key_meas = np.exp(np.nanmean(np.log(a_meas), axis=1))
    key_true = np.exp(np.nanmean(np.log(truth), axis=1))
print("\nper mode, median (dB):  measured on the clear keys | true on the clear keys | true on all keys in the model")
for n, m in enumerate(names):
    k = clear[:, n] & np.isfinite(key_meas[:, n])
    if not k.any():
        continue
    slope = [np.polyfit(np.log(vels[u]), np.log(a_meas[i, u, n] / truth[i, u, n]), 1)[0]
             for i in np.flatnonzero(k) for u in [np.isfinite(a_meas[i, :, n])] if u.sum() >= 3]
    print(f"  {m}: {DB * np.log(np.median(key_meas[k, n])):+6.1f} | {DB * np.log(np.median(key_true[k, n])):+6.1f} | "
          f"{DB * np.log(np.nanmedian(key_true[ok[:, n], n])):+6.1f}   (error p -> f "
          f"{DB * np.median(slope) * np.log(vels[-1] / vels[0]):+.1f} dB)")
np.savez("cache/step4_synthetic_check.npz", a_true=a_true, truth=truth, a_meas=a_meas, err=err, bass=bass,
         clear=clear, key_meas=key_meas, key_true=key_true, spread_db=SPREAD_DB)
print("-> cache/step4_synthetic_check.npz")

# %%
# ---------- 4. figure ----------
os.makedirs("plots", exist_ok=True)
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
ax = axs[0]
for n, m in enumerate(names):
    u = np.isfinite(err[:, :, n])
    if not u.any():
        continue
    b = np.broadcast_to(bass[:, None], u.shape)
    for sel, face in ((u & ~b, f"C{n}"), (u & b, "none")):
        ax.plot(DB * np.log(truth[:, :, n][sel]), DB * np.log(a_meas[:, :, n][sel]), "o", ms=3,
                color=f"C{n}", mfc=face, label=m if face != "none" else None)
lim = ax.get_xlim()
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("true $A_n/A_0$ (dB)"); ax.set_ylabel("measured $A_n/A_0$ (dB)"); ax.legend(fontsize=8)
ax.set_title(f"measured against true (open: below {F0_BASS:.0f} Hz)")
ax = axs[1]
kk = np.arange(M)
for n, m in enumerate(names):
    if np.isfinite(err[:, :, n]).any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ax.semilogy(kk, np.nanmedian(np.abs(err[:, :, n]), axis=1) + 0.1, "o", ms=3, color=f"C{n}", label=m)
ax.set_xticks(kk[::6]); ax.set_xticklabels(notes[::6], rotation=90, fontsize=8)
ax.set_ylabel("|error| (dB, median over dyn)"); ax.legend(fontsize=8)
ax.set_title("error of the measured A_n/A_0 per key")
plt.tight_layout()
plt.savefig("plots/step4_synthetic_check.png", dpi=100)
plt.show()
