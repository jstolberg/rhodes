# %% [markdown]
# # Fitting hammer parameters (step 3 of fitting.py)
#
# Fixed: surface, $p_o$, $p_d$, fundamental $f_0$ and its decay.  Free:
# $\tau_0$ (smooth over the keyboard) and $\beta$ (global).  Arbitrary $c_0$.
#
# Step 2 hands over, per key $k$ and dynamic, the free-oscillation displacement
# amplitude $A_0(k, v)$.  That is exactly what `hammer2free` returns, so no
# forward pass through the pickup is needed:
# $$\log A_0(k, v) = \log c_0(k) + \log v + \log|g(k, v; \tau_0, \beta)|,
#   \qquad \tau(v) = \tau_0(k)\, v^{-\beta}.$$
# $c_0(k)$ is an additive per-key constant in the log, so subtracting the
# per-key mean of the residual cancels it (the log-domain version of taking
# amplitude ratios between velocities); that mean is then $\log c_0(k)$ for free.
#
# The information on $\tau_0, \beta$ sits in $f_0 \cdot \tau$: bass keys are
# in the impulsive limit ($A_0 \propto v$, blind to $\tau$), the treble is not.
# Hence $\log\tau_0$ is a straight line over the key index so the treble anchors
# the bass, and the initial guess must stay inside the main lobe of the pulse
# spectrum, $f_0\,\tau(v_\min) < 2$, or the loss becomes multimodal.

# %%
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

from model import hammer2free, calc_tau

# %%
# ---------- data ----------
# Measured fundamentals: structured (73 keys, 4 dynamics), fields f, lam, amp0,
# fell_db, resid_db, ok.  Only f is used here; A_0 comes from step 2.
data  = np.load("fundamentals.npz", allow_pickle=True)
notes = data["notes"]
M     = len(notes)

f0 = np.nanmedian(data["table"]["f"], axis=1)            # pitch is velocity-free
et = 41.2034 * 2.0 ** (np.arange(M) / 12)                 # E0 ... equal temperament
f0 = jnp.asarray(np.where(np.isfinite(f0), f0, et))       # fallback for unfitted keys

VELS = jnp.array([0.25, 0.5, 0.75, 1.0])                  # assumed for p, mp, mf, f
keys = jnp.arange(M) / (M - 1)                            # 0 at E0, 1 at top key


# %%
# ---------- model ----------
# theta: log_tau0 = (log tau_0 at bottom key, log tau_0 at top key); log_beta.
# Everything in log space so Adam sees O(1) gradients and positivity is free.

def tau0_of_key(theta):
    """(M,) contact time at v=1, log-linear over the keyboard."""
    lo, hi = theta['log_tau0']
    return jnp.exp(lo + (hi - lo) * keys)


def log_amp(theta):
    """(M, 4) log|A_0| for c_0 = 1: the velocity-dependent part of the model."""
    beta = jnp.exp(theta['log_beta'])
    tau0 = tau0_of_key(theta)

    def one(f, t0, v):
        tau = calc_tau(v, dict(tau_0=t0, beta=beta))
        return jnp.log(jnp.abs(hammer2free(f, 1.0, v, tau)[0]))

    return jax.vmap(jax.vmap(one, (None, None, 0)), (0, 0, None))(f0, tau0, VELS)


def log_c0(theta, logA0, ok):
    """(M,) per-key mean residual = log c_0 estimate (masked)."""
    r = jnp.where(ok, logA0 - log_amp(theta), 0.0)
    return r.sum(1) / ok.sum(1)


def loss(theta, logA0, ok):
    """MSE of the residual with the per-key mean removed -- c_0 cancelled."""
    r  = logA0 - log_amp(theta)
    d  = jnp.where(ok, r - log_c0(theta, logA0, ok)[:, None], 0.0)
    return jnp.sum(d**2) / ok.sum()


def fit(theta, logA0, ok, steps=3000, lr=0.02):
    opt   = optax.adam(lr)
    state = opt.init(theta)

    @jax.jit
    def step(theta, state):
        l, g = jax.value_and_grad(loss)(theta, logA0, ok)
        upd, state = opt.update(g, state, theta)
        return optax.apply_updates(theta, upd), state, l

    hist = []
    for _ in range(steps):
        theta, state, l = step(theta, state)
        hist.append(float(l))
    return theta, np.array(hist)


def report(theta):
    t0 = tau0_of_key(theta)
    return (f"tau_0: {float(t0[0])*1e3:.3f} ms ({notes[0]}) -> "
            f"{float(t0[-1])*1e3:.3f} ms ({notes[-1]}),  "
            f"beta = {float(jnp.exp(theta['log_beta'])):.4f}")


# %%
# ---------- synthetic round trip (see 'Evaluation measures' in fitting.py) ----------
# Generate A_0 from known parameters, perturb, fit, compare.

def synth(theta, c0, key, noise=0.03, p_missing=0.05):
    """(logA0, ok) from the model with per-key c0 and log-normal noise."""
    k1, k2 = jax.random.split(key)
    logA0 = jnp.log(c0)[:, None] + log_amp(theta)
    logA0 = logA0 + noise * jax.random.normal(k1, logA0.shape)
    ok    = jax.random.uniform(k2, logA0.shape) > p_missing
    return logA0, ok


key = jax.random.PRNGKey(0)
theta_true = dict(log_tau0=jnp.log(jnp.array([3e-3, 0.5e-3])),   # bass -> treble
                  log_beta=jnp.log(0.2))
c0_true    = jnp.exp(jax.random.normal(key, (M,)))                # arbitrary per key
logA0, ok  = synth(theta_true, c0_true, key)

# Start the bass long (the short side is a plateau there), the treble inside the
# main lobe of the pulse spectrum for the softest velocity: f0*tau_0*v_min^-beta
# < 2.  With the top key at 2.64 kHz that caps tau_0 there at ~0.57 ms (beta=0.2);
# a truth beyond the null cannot be reached from below, see the markdown above.
theta0 = dict(log_tau0=jnp.log(jnp.array([8e-3, 0.4e-3])),
              log_beta=jnp.log(0.3))
print("top-key f0*tau(v_min) at init:",
      float(f0[-1] * calc_tau(VELS[0], dict(tau_0=jnp.exp(theta0['log_tau0'][1]),
                                            beta=jnp.exp(theta0['log_beta'])))))

theta_fit, hist = fit(theta0, logA0, ok)
print(f"loss: {hist[0]:.3e} -> {hist[-1]:.3e}  (noise floor ~ {0.03**2 * 3/4:.1e})")
print("true:", report(theta_true))
print("init:", report(theta0))
print("fit: ", report(theta_fit))

c0_fit = jnp.exp(log_c0(theta_fit, logA0, ok))
print(f"c_0 max rel error: {float(jnp.max(jnp.abs(c0_fit / c0_true - 1))):.3%}")

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].semilogy(hist); ax[0].set_xlabel('step'); ax[0].set_ylabel('loss')
ax[1].plot(tau0_of_key(theta_true) * 1e3, label='true')
ax[1].plot(tau0_of_key(theta0) * 1e3, ':', label='init')
ax[1].plot(tau0_of_key(theta_fit) * 1e3, '--', label='fit')
ax[1].set_xlabel('key'); ax[1].set_ylabel('tau_0 (ms)'); ax[1].legend()
ax[2].plot(c0_true, label='true'); ax[2].plot(c0_fit, '--', label='fit')
ax[2].set_xlabel('key'); ax[2].set_ylabel('c_0'); ax[2].legend()
plt.tight_layout(); plt.show()

# %%
# ---------- real data ----------
# Once step 2 exists: logA0 = jnp.log(A0_measured) with shape (M, 4) in the
# order p, mp, mf, f and ok its validity mask, then
#   theta_fit, hist = fit(theta0, logA0, ok);  c0 = jnp.exp(log_c0(theta_fit, logA0, ok))
# Sanity check on the assumed velocities: for bass keys A_0 is blind to tau, so
# exp(logA0[:12, 0] - logA0[:12, 3]) should sit near VELS[0] = 0.25.
