# %% [markdown]
# # Fitting hammer parameters (step 3 of fitting.py)
#
# **Idea.**  Step 2 measured, for every key $k$ and dynamic, the amplitude $A_0(k, v)$
# of the free oscillation of the fundamental.  How $A_0$ changes from p to f depends
# only on the hammer: a faster strike gives a bigger push ($\propto v$) and a shorter
# contact $\tau(v) = \tau_0\, v^{-\beta}$.  `hammer2free` of `model.py` gives the
# amplitude the hammer leaves behind, so
# $$\log A_0(k, v) = \log c_0(k) + \log|\text{hammer2free}(f_0(k), 1, v, \tau(v))|.$$
# (The $v$ of the push is inside `hammer2free`.)  The pickup plays no role.
#
# **Removing $c_0$.**  $c_0(k)$ (how strongly key $k$ is excited) is the same at every
# velocity, so in the log it is one constant per key.  Subtracting the per-key mean of
# the residual removes it; the loss then only sees the *differences* between the
# dynamics.  The mean itself is $\log c_0(k)$ (step 4 of `fitting.py`).
#
# **What the data can tell.**  The information on $\tau_0$ and $\beta$ is in $f_0 \tau$.
# Bass keys have $f_0 \tau \ll 1$: the pulse acts like an instant kick,
# $A_0 \propto v$, whatever $\tau$ is.  Only the treble is sensitive.  So $\log\tau_0$ is
# a straight line over the keys (2 numbers) and the treble fixes the bass.  The start
# must lie inside the main lobe of the pulse spectrum, $f_0\,\tau(v_\min) < 2$, or the
# loss has several valleys.
#
# **Fitted.**  $\tau_0$ at the lowest and highest key, $\beta$ (global); $c_0$ per key.
# **Assumed.**  The velocities `VELS` = 0.25, 0.5, 0.75, 1 of p, mp, mf, f.
#
# The file first checks the method on synthetic data with known parameters, then fits
# the real $A_0$ of step 2 and writes `step3_fit.npz`.

# %%
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from model import hammer2free, calc_tau
from step2_lib import T_START, VELS, fit

STEPS, LR = 3000, 0.02                                    # Adam


# %%
# ---------- keys ----------
data  = np.load("fundamentals.npz", allow_pickle=True)
notes = data["notes"]
M     = len(notes)
pos   = jnp.arange(M) / (M - 1)                            # key position: 0 at E0, 1 at the top key


# %%
# ---------- model ----------
# theta = dict(log_tau0 = (log tau_0 at the lowest key, log tau_0 at the highest key),
#              log_beta).  Everything in log space: Adam sees gradients of size ~1 and
# the values stay positive.

def tau0_of_key(theta):
    """(M,) contact time at v = 1: a straight line in log over the keyboard."""
    lo, hi = theta["log_tau0"]
    return jnp.exp(lo + (hi - lo) * pos)


def log_amp(theta, f0, sig=None, t_meas=0.0):
    """(M, 4) log|A_0| of every key and velocity for c_0 = 1.

    With sig (M,) the amplitude the hammer hands over at t = tau is carried on to the
    time t_meas where step 2 measured it: A(t_meas) = A_hand e^{-sig (t_meas - tau)}.
    """
    beta = jnp.exp(theta["log_beta"])
    tau0 = tau0_of_key(theta)
    sig  = jnp.zeros(M) if sig is None else sig

    def one(f, t0, s, v):                                  # one key, one velocity
        tau = calc_tau(v, dict(tau_0=t0, beta=beta))
        return jnp.log(jnp.abs(hammer2free(f, 1.0, v, tau)[0])) - s * (t_meas - tau)

    over_vel = jax.vmap(one, (None, None, None, 0))       # the 4 velocities of a key
    return jax.vmap(over_vel, (0, 0, 0, None))(f0, tau0, sig, VELS)


def residual(theta, logA0, ok, f0, sig=None, t_meas=0.0):
    """(M, 4) log residual data - model, and its per-key mean (= log c_0 estimate, (M,)).
    Cells where step 2 failed (ok False) are left out of the mean."""
    r     = logA0 - log_amp(theta, f0, sig, t_meas)
    log_c = jnp.where(ok, r, 0.0).sum(1) / ok.sum(1)
    return r, log_c


def loss(theta, logA0, ok, f0, sig=None, t_meas=0.0):
    """Mean square of the residual with the per-key mean removed, so c_0 drops out."""
    r, log_c = residual(theta, logA0, ok, f0, sig, t_meas)
    d = jnp.where(ok, r - log_c[:, None], 0.0)
    return jnp.sum(d**2) / ok.sum()


def report(theta):
    t0 = tau0_of_key(theta)
    return (f"tau_0: {float(t0[0])*1e3:.3f} ms ({notes[0]}) -> "
            f"{float(t0[-1])*1e3:.3f} ms ({notes[-1]}),  "
            f"beta = {float(jnp.exp(theta['log_beta'])):.4f}")


# Start: tau_0 long in the bass (the short side is flat there), and in the treble inside
# the main lobe for the softest velocity, f_0 tau_0 v_min^-beta < 2.  With the top key at
# 2.64 kHz that caps tau_0 there at ~0.57 ms (beta = 0.2); a truth beyond the first null
# cannot be reached from below.
theta0 = dict(log_tau0=jnp.log(jnp.array([8e-3, 0.4e-3])), log_beta=jnp.log(0.3))


# %%
# ---------- synthetic round trip (see "Evaluation measures" in fitting.py) ----------
# A_0 from known parameters and a random c_0 per key, 3 % noise, 5 % of the cells
# missing; then fit and compare.  f_0: step 1, equal temperament where step 1 has none.

f0_syn = np.nanmedian(data["table"]["f"], axis=1)          # pitch does not depend on velocity
f0_et  = 41.2034 * 2.0 ** (np.arange(M) / 12)              # E0 ... in equal temperament
f0_syn = jnp.asarray(np.where(np.isfinite(f0_syn), f0_syn, f0_et))

key        = jax.random.PRNGKey(0)
theta_true = dict(log_tau0=jnp.log(jnp.array([3e-3, 0.5e-3])), log_beta=jnp.log(0.2))
c0_true    = jnp.exp(jax.random.normal(key, (M,)))

k1, k2 = jax.random.split(key)
logA0_s = jnp.log(c0_true)[:, None] + log_amp(theta_true, f0_syn)
logA0_s = logA0_s + 0.03 * jax.random.normal(k1, logA0_s.shape)    # 3 % noise
ok_s    = jax.random.uniform(k2, logA0_s.shape) > 0.05             # 5 % missing

print("top-key f0*tau(v_min) at start:",
      float(f0_syn[-1] * calc_tau(VELS[0], dict(tau_0=jnp.exp(theta0["log_tau0"][1]),
                                                beta=jnp.exp(theta0["log_beta"])))))

theta_s, hist_s, _ = fit(theta0, lambda th: loss(th, logA0_s, ok_s, f0_syn), STEPS, lr=LR, label="synthetic: ")
print(f"  noise floor ~ {0.03**2 * 3/4:.1e}")
print("true: ", report(theta_true))
print("start:", report(theta0))
print("fit:  ", report(theta_s))
c0_s = jnp.exp(residual(theta_s, logA0_s, ok_s, f0_syn)[1])
print(f"c_0 max rel error: {float(jnp.max(jnp.abs(c0_s / c0_true - 1))):.3%}")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].semilogy(hist_s); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
ax[1].plot(tau0_of_key(theta_true) * 1e3, label="true")
ax[1].plot(tau0_of_key(theta0) * 1e3, ":", label="start")
ax[1].plot(tau0_of_key(theta_s) * 1e3, "--", label="fit")
ax[1].set_xlabel("key"); ax[1].set_ylabel("tau_0 (ms)"); ax[1].legend()
ax[2].plot(c0_true, label="true"); ax[2].plot(c0_s, "--", label="fit")
ax[2].set_xlabel("key"); ax[2].set_ylabel("c_0"); ax[2].legend()
plt.tight_layout(); plt.show()


# %%
# ---------- real data ----------
# A_0 (M, 4) of step 2, in the order p, mp, mf, f, measured at t_0 = onset + T_START,
# with sigma_0 per key.  The model carries the hand-over amplitude to t_0 with that
# decay, so c_0 is the true excitation coefficient.
fit2  = np.load("step2_fit.npz", allow_pickle=True)
assert (fit2["notes"] == notes).all()
logA0 = jnp.log(jnp.asarray(fit2["A0"]))
ok    = jnp.asarray(fit2["ok"])
sig0  = jnp.asarray(fit2["sigma"])
f0    = jnp.asarray(fit2["f0"])                            # the f_0 step 2 used

theta, hist, _ = fit(theta0, lambda th: loss(th, logA0, ok, f0, sig0, T_START), STEPS, lr=LR, label="real: ")
print("fit: ", report(theta))

r, log_c0 = residual(theta, logA0, ok, f0, sig0, T_START)
c0 = jnp.exp(log_c0)
r  = jnp.where(ok, r - log_c0[:, None], 0.0)              # what the hammer model does not explain
print("rms residual per dynamic (log):", np.round(np.sqrt(np.asarray((r**2).sum(0) / ok.sum(0))), 3))
print("bass check, mean A_0 ratios p/mp/mf : f over the lowest 12 keys (assumed .25 .5 .75):",
      np.round(np.asarray(jnp.exp(logA0[:12, :3] - logA0[:12, 3:]).mean(0)), 3))

np.savez("step3_fit.npz", notes=notes, f0=np.asarray(f0), tau0=np.asarray(tau0_of_key(theta)),
         beta=float(jnp.exp(theta["log_beta"])), c0=np.asarray(c0),
         log_tau0=np.asarray(theta["log_tau0"]))

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].semilogy(hist); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
ax[1].semilogy(tau0_of_key(theta) * 1e3, label="fit")
ax[1].semilogy(tau0_of_key(theta0) * 1e3, ":", label="start")
ax[1].set_ylabel("tau_0 (ms)"); ax[1].legend()
ax[2].semilogy(c0); ax[2].set_ylabel("c_0")
for a in ax[1:]:
    a.set_xticks(range(0, M, 6)); a.set_xticklabels(notes[::6], rotation=60)
plt.suptitle(f"step 3 on real data: {report(theta)}"); plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/step3_params.png", dpi=90); plt.show()

plt.figure(figsize=(15, 4))
for j, d in enumerate(["p", "mp", "mf", "f"]):
    plt.plot(np.asarray(r[:, j]), ".-", label=d)
plt.axhline(0, color="k", lw=0.5); plt.ylabel("log residual (c_0 removed)"); plt.legend()
plt.xticks(range(0, M, 6), notes[::6], rotation=60); plt.tight_layout()
plt.savefig("plots/step3_residuals.png", dpi=90); plt.show()
