# %% [markdown]
# # Inharmonic modes on every key (step 6)
#
# Step 5 (`step_5.py`, imported unchanged: it runs on import) measures each mode's decay
# only on the keys where the mode is clear, as a multiple of the fundamental's decay.
# Step 6 puts the modes on **all** keys, using only what step 5 gives:
#
# - **Frequency**: $f_n = \mu_n f_0$, Gabrielli's mean ratio (`MODES` of step 5) times the
#   key's $f_0$ (step 2).  Modes above `F_MAX` = 10 kHz are left out (Gabrielli found none).
# - **Decay**: $\sigma_n = r_n \sigma_0$, $r_n$ the median of step 5's ratios over the clear
#   keys.  Modes without a clear key have no $r_n$ and are left out.
# - **Excitation** $c_n$, the only thing set here, treated like the decays: measured only on
#   the keys where step 5 found the mode clearly, as a ratio $c_n / c_0$, and the median over
#   those keys applied to all keys ($c_n = (c_n/c_0) \cdot c_0$).
#   The measurement: step 5's spectrum of the first `T_FFT` s and its peak search near
#   $\mu_n f_0$ give the mode's level in the recording.  The model's spectrum over the same
#   span, read at $f_n$, is proportional to $c_n$ (a small mode passes the pickup linearly),
#   so one render with a trial $c_n$ and one scaling match the two.  This is done per
#   dynamic; a key's ratio is the mean of $\log(c_n / c_0)$ over the dynamics.
#   Keys below `F0_MIN` = 150 Hz are not measured: there $7.1 f_0$ lies within `HARM_HZ` of
#   $7 f_0$, which the peak search excludes (`step4_synthetic_check.py`).
#
# Everything else is fixed from steps 2 and 3.  Output `step6_fit.npz` for `step6_render.py`
# (`c`: $c_n$ per key, repeated per dynamic; `c_dyn`: the measured value of each dynamic, 0
# where not measured; `measured`: keys at or above `F0_MIN`).

# %%
import warnings

import jax.numpy as jnp
import numpy as np

import step_5 as s5
from model import calc_tau, epsilon, hammer2free
from step2_lib import DYNS, T_START, TABLE, VELS

F_MAX   = 10e3     # Hz, no modes above (Gabrielli 2020)
FS      = 48000.0  # model sample rate
A_TRIAL = 1e-3     # trial amplitude A_n / A_0 (small: the pickup is linear for it)
F0_MIN  = 150.0    # Hz, below: c_n not measurable (see above)

fit2 = np.load("step2_fit.npz", allow_pickle=True)
fit3 = np.load("step3_fit.npz", allow_pickle=True)
assert (fit2["notes"] == fit3["notes"]).all()              # fit3 is indexed with fit2's keys
kappa, beta = float(fit2["kappa"]), float(fit3["beta"])
names = list(s5.MODES)
mu, sd = np.array([s5.MODES[m] for m in names]).T
r = np.array([np.median([x["ratio"] for x in s5.rows if x["mode"] == m])
              if any(x["mode"] == m for x in s5.rows) else np.nan for m in names])
notes = fit2["notes"]
f_n   = mu[None, :] * fit2["f0"][:, None]                  # (keys, N)
ok    = np.isfinite(r)[None, :] & (f_n < F_MAX)             # modes in the model
sig_n = r[None, :] * fit2["sigma"][:, None]
print("modes in the model, keys:", dict(zip(names, ok.sum(0))))


# %%
def model_db(i, c, vel):
    """Step 5's spectrum (dB) of the modes' part of the model over the first T_FFT s after
    contact, read at the kept f_n (-inf for the others).  The modes' part is the model with
    the kept modes (excitation c) minus the model without them: on low keys 7.1 f_0 lies
    within the window's main lobe of 7 f_0, whose leakage would otherwise be read."""
    k = ok[i]
    t = jnp.asarray(T_START + np.arange(int(s5.T_FFT * FS)) / FS)

    def eps(c_n):
        p = dict(c=jnp.r_[fit3["c0"][i], c_n], lam=jnp.r_[fit2["sigma"][i], sig_n[i][k]],
                 f_modes=jnp.r_[fit2["f0"][i], f_n[i][k]], tau_0=float(fit3["tau0"][i]),
                 beta=beta, p_d=float(fit2["p_d"][i]), p_o=float(fit2["p_o"][i]))
        return kappa * np.asarray(epsilon(t, p, TABLE, vel=vel))

    f, db = s5.spectrum(eps(c[k]) - eps(0.0 * c[k]), FS)
    return np.where(k, np.interp(f_n[i], f, db), -np.inf)


def amps(i, vel):
    """Free amplitude A_0 of the fundamental and |A_n| per unit c_n that the hammer hands over."""
    tau = calc_tau(vel, dict(tau_0=fit3["tau0"][i], beta=beta))
    A0 = abs(float(hammer2free(fit2["f0"][i], fit3["c0"][i], vel, tau)[0]))
    g = np.abs(np.asarray(hammer2free(jnp.asarray(f_n[i]), 1.0, vel, tau)[0]))
    return A0, np.maximum(g, 1e-30)


# %%
measured = fit2["f0"] >= F0_MIN                             # keys whose c_n is measured
c = np.zeros((len(notes), len(DYNS), len(names)))
for i, note in enumerate(notes):
    if not ok[i].any() or not measured[i]:
        continue
    i5 = s5.NOTES.index(note)
    for j, dyn in enumerate(DYNS):
        # level in the recording: step 5's peak near mu_n f_0 (-inf where there is none)
        f, db = s5.spectrum(*s5.load(note, dyn))
        fp = [s5.find_peak(f, db, s5.FUND[i5, j]["f"], m, s) for m, s in zip(mu, sd)]
        db_rec = np.array([np.interp(x, f, db) if np.isfinite(x) else -np.inf for x in fp])
        # trial c_n: free amplitude A_TRIAL * A_0, then scaled to the recording's level
        A0, g = amps(i, VELS[j])
        c_try = np.where(ok[i], A_TRIAL * A0 / g, 0.0)
        with np.errstate(invalid="ignore"):
            c[i, j] = np.where(ok[i], c_try * 10 ** ((db_rec - model_db(i, c_try, VELS[j])) / 20), 0.0)
    print(f"{note:>4}", end=" ", flush=True)

# %%
c = c_dyn = np.nan_to_num(c)                                              # (keys, dyn, N)
used = ok[:, None, :] & (c > 0)
print("\nkeys x dynamics with a c_n per mode:", dict(zip(names, used.sum((0, 1)))))

# Keys where step 5 found the mode clearly
clear = np.zeros(ok.shape, bool)
for x in s5.rows:
    clear[list(notes).index(x["note"]), names.index(x["mode"])] = True
clear &= ok

# c_n / c_0 per key: mean of the log over the dynamics; spread over the dynamics in dB
c0 = fit3["c0"]
log_rel = np.where(used, np.log(np.where(used, c, 1.0)) - np.log(c0)[:, None, None], np.nan)
with warnings.catch_warnings():                                            # unmeasured keys: all nan
    warnings.simplefilter("ignore", RuntimeWarning)
    key_rel = np.nanmean(log_rel, axis=1)                                  # (keys, N)
    spread = 20 / np.log(10) * np.nanstd(log_rel, axis=1)                 # dB, (keys, N)

# One c_n / c_0 per mode: the median over the clear keys, as the decays in step 5
rel = np.full(len(names), np.nan)
for n, m in enumerate(names):
    k = clear[:, n] & np.isfinite(key_rel[:, n])                          # clear and measured
    if k.any():
        rel[n] = np.median(key_rel[k, n])
        print(f"{m}: c_n/c_0 = {20 / np.log(10) * rel[n]:+.1f} dB over {k.sum()} clear keys "
              f"({clear[:, n].sum()} clear), spread over the dynamics {np.median(spread[k, n]):.1f} dB")
    else:
        print(f"{m}: no clear key with a measurement")
c = np.where(ok & np.isfinite(rel)[None, :], np.exp(rel)[None, :] * c0[:, None], 0.0)   # (keys, N)
c = np.repeat(c[:, None, :], len(DYNS), axis=1)                            # same for every dynamic
np.savez("step6_fit.npz", notes=notes, dyns=np.array(DYNS), ratios=mu, modes=names,
         f_modes=f_n, ok=ok, used=used, measured=measured, c=c, c_dyn=c_dyn, sig=np.where(ok, sig_n, np.nan))
print("-> step6_fit.npz")
