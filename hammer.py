# %% [markdown]
# # Fitting hammer parameters (step 3 of fitting.py)
#
# Step 2 hands over, per key $k$ and dynamic, the amplitude $A_0$ of the free
# fundamental at $t_0$ = onset + `T_START`, and its decay $\sigma_0$.  The hammer
# is a half-sine force pulse of impulse $v$ and duration $\tau$; the amplitude it
# leaves (`hammer2free`) is $v$ times the pulse spectrum $g$ at $f_0$.  Carried to $t_0$:
# $$\log A_0(k, v) = \log c_0(k) + \log v + \log|g(f_0 \tau)| - \sigma_0 (t_0 - \tau),
#   \qquad \tau = \tau_0(k)\, v^{-\beta}.$$
#
# * $c_0(k)$ -- per-key scale (tine, pickup, recording level).  Subtracting the
#   per-key mean of the log residual cancels it, so only the amplitude ratios
#   between dynamics are fitted; the mean is then $\log c_0$ for free.
# * $v$ -- one velocity per dynamic, relative to f ($v_f = 1$).  In the bass
#   $f_0\tau$ is smallest, $g \approx 1$ and $A_0 \propto v$: the bass measures the
#   velocities, as the median ratio $A_0 / A_0(f)$ over its keys.  They are held
#   in the fit: fitted jointly they trade with $\beta$ (the fit escapes to
#   $\beta \to 1$ with compressed velocities).
# * $\tau_0$ -- log-linear over the key index, $\beta$ global.  Only the treble
#   ($g < 1$) carries information on them; the line lets it anchor the bass.  The
#   start must keep the treble inside the main lobe of the pulse spectrum,
#   $f_0 \tau(v_\min) < 2$, or the loss becomes multimodal.
#
# Keys flagged in step 2 (detuned samples) are left out of the loss; their $c_0$
# is interpolated from the neighbours.
#
# | Input | Content |
# | --- | --- |
# | `step2_fit.npz` | `A0` (73 x 4), `sigma`, `f0`, `flag` |
#
# | Output (`step3_fit.npz`) | Content |
# | --- | --- |
# | `tau0`, `log_tau0` | contact time at $v_f$ per key (s); the line's end points |
# | `beta` | velocity exponent of the contact time |
# | `vels` | velocities p, mp, mf, f relative to f (from the bass) |
# | `c0` | per-key scale (flagged keys interpolated) |
#
# | Name | Value | Meaning |
# | --- | --- | --- |
# | `T_START` | 10 ms | measurement time of $A_0$ after onset (as in step 2) |
# | `BASS` | E0-D#1 | keys whose median amplitude ratio gives the velocities |
# | `STEPS`, `LR` | 3000, 0.02 | Adam |

# %%
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

from model import hammer2free, calc_tau

T_START    = 10e-3
BASS       = slice(0, 12)
STEPS, LR  = 3000, 0.02
DYNS       = ["p", "mp", "mf", "f"]

# %%
# ---------- data ----------
fit2  = np.load("step2_fit.npz", allow_pickle=True)
notes = fit2["notes"]
M     = len(notes)
f0    = jnp.asarray(fit2["f0"])
sig   = jnp.asarray(fit2["sigma"])
logA0 = jnp.log(jnp.asarray(fit2["A0"]))
flag  = np.asarray(fit2["flag"])
use   = jnp.asarray(~flag)                                   # keys in the loss
keys  = jnp.arange(M) / (M - 1)                              # 0 at E0, 1 at top key


def bass_vels(logA0):
    """(4,) velocities p, mp, mf, f: median A_0 / A_0(f) over the unflagged bass keys."""
    b = np.arange(M)[BASS][~flag[BASS]]
    return jnp.asarray(np.append(np.median(np.exp(np.asarray(logA0)[b, :3] - np.asarray(logA0)[b, 3:]), axis=0), 1.0))


# %%
# ---------- model ----------
# theta: log_tau0 = (log tau_0 at E0, at the top key); log_beta.  v (4,) held.
# Everything in log space so Adam sees O(1) gradients and positivity is free.

def tau0_of_key(theta):
    lo, hi = theta["log_tau0"]
    return jnp.exp(lo + (hi - lo) * keys)


def log_amp(theta, v):
    """(M, 4) log A_0 for c_0 = 1."""
    beta = jnp.exp(theta["log_beta"])

    def one(f, t0, s, vk):
        tau = calc_tau(vk, dict(tau_0=t0, beta=beta))
        return jnp.log(jnp.abs(hammer2free(f, 1.0, vk, tau)[0])) - s * (T_START - tau)

    return jax.vmap(jax.vmap(one, (None, None, None, 0)), (0, 0, 0, None))(f0, tau0_of_key(theta), sig, v)


def log_c0(theta, logA0, v):
    """(M,) per-key mean log residual = log c_0."""
    return (logA0 - log_amp(theta, v)).mean(1)


def residual(theta, logA0, v):
    """(M, 4) log residual with c_0 removed."""
    return logA0 - log_amp(theta, v) - log_c0(theta, logA0, v)[:, None]


def loss(theta, logA0, v):
    return jnp.sum(jnp.where(use[:, None], residual(theta, logA0, v) ** 2, 0.0)) / (4 * use.sum())


def fit(theta, logA0, v):
    opt   = optax.adam(LR)
    state = opt.init(theta)

    @jax.jit
    def step(theta, state):
        l, g = jax.value_and_grad(loss)(theta, logA0, v)
        upd, state = opt.update(g, state, theta)
        return optax.apply_updates(theta, upd), state, l

    hist = []
    for _ in range(STEPS):
        theta, state, l = step(theta, state)
        hist.append(float(l))
    return theta, np.array(hist)


# Start: tau_0 long in the bass (the short side is a plateau there), inside the main lobe
# in the treble.
theta0 = dict(log_tau0=jnp.log(jnp.array([8e-3, 0.4e-3])), log_beta=jnp.log(0.3))


def c0_filled(theta, logA0, v):
    """c_0 per key, flagged keys log-interpolated from the others."""
    lc = np.asarray(log_c0(theta, logA0, v))
    k  = np.arange(M)
    return np.exp(np.interp(k, k[~flag], lc[~flag]))


def report(theta, v):
    t0 = tau0_of_key(theta)
    return (f"tau_0: {float(t0[0]) * 1e3:.3f} ms ({notes[0]}) -> {float(t0[-1]) * 1e3:.3f} ms ({notes[-1]}),  "
            f"beta = {float(jnp.exp(theta['log_beta'])):.4f},  v = {np.round(np.asarray(v), 3)}")


# %%
# ---------- synthetic round trip ----------
# A_0 from known parameters and random c_0, 3 % log-normal noise; fit and compare.
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
theta_true = dict(log_tau0=jnp.log(jnp.array([3e-3, 0.5e-3])), log_beta=jnp.log(0.2))
v_true   = jnp.array([0.3, 0.55, 0.8, 1.0])
c0_true  = jnp.exp(jax.random.normal(k1, (M,)))
logA0_s  = jnp.log(c0_true)[:, None] + log_amp(theta_true, v_true) + 0.03 * jax.random.normal(k2, (M, 4))

v_s = bass_vels(logA0_s)
print("top-key f0*tau(v_min) at start:",
      float(f0[-1] * calc_tau(v_s[0], dict(tau_0=jnp.exp(theta0["log_tau0"][1]), beta=jnp.exp(theta0["log_beta"])))))
theta_s, hist_s = fit(theta0, logA0_s, v_s)
print(f"synthetic: loss {hist_s[0]:.3e} -> {hist_s[-1]:.3e}  (noise floor ~ {0.03 ** 2 * 3 / 4:.1e})")
print("  true: ", report(theta_true, v_true))
print("  fit:  ", report(theta_s, v_s))
c0_s = jnp.exp(log_c0(theta_s, logA0_s, v_s))
print(f"  c_0 max rel error (unflagged): {float(jnp.max(jnp.abs(c0_s / c0_true - 1)[use])):.2%}")

# %%
# ---------- real data ----------
v = bass_vels(logA0)
theta, hist = fit(theta0, logA0, v)
c0 = c0_filled(theta, logA0, v)
r  = np.asarray(residual(theta, logA0, v))
print(f"real: loss {hist[0]:.3e} -> {hist[-1]:.3e}")
print("  fit:  ", report(theta, v))
print("  rms log residual per dynamic:", np.round(np.sqrt((r[~flag] ** 2).mean(0)), 3))
print("  excluded (flagged in step 2):", list(notes[flag]))

np.savez("step3_fit.npz", notes=notes, f0=np.asarray(f0), tau0=np.asarray(tau0_of_key(theta)),
         log_tau0=np.asarray(theta["log_tau0"]), beta=float(jnp.exp(theta["log_beta"])),
         vels=np.asarray(v), c0=c0, flag=flag)

# %%
os.makedirs("plots", exist_ok=True)
tick = dict(ticks=range(0, M, 6), labels=notes[::6], rotation=60)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].semilogy(hist); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
for j, d in enumerate(DYNS):
    tau = calc_tau(v[j], dict(tau_0=tau0_of_key(theta), beta=jnp.exp(theta["log_beta"])))
    ax[1].semilogy(np.asarray(tau) * 1e3, label=f"$\\tau$({d})")
ax[1].set_ylabel("contact time (ms)"); ax[1].legend(fontsize=8)
ax[2].semilogy(c0, ".-")
ax[2].semilogy(np.flatnonzero(flag), c0[flag], "o", mfc="none", color="C3", label="flagged (interpolated)")
ax[2].set_ylabel("$c_0$"); ax[2].legend(fontsize=8)
for a in ax[1:]:
    a.set_xticks(**tick)
plt.suptitle(f"step 3: {report(theta, v)}", fontsize=10); plt.tight_layout()
plt.savefig("plots/step3_params.png", dpi=90); plt.show()

plt.figure(figsize=(15, 4))
for j, d in enumerate(DYNS):
    plt.plot(np.where(flag, np.nan, r[:, j]), ".-", label=d)
plt.axhline(0, color="k", lw=0.5); plt.ylabel("log residual ($c_0$ removed)"); plt.legend()
plt.xticks(**tick); plt.title("flagged keys omitted"); plt.tight_layout()
plt.savefig("plots/step3_residuals.png", dpi=90); plt.show()
