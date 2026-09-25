# %% [markdown]
# # Step 3 with free velocities (check of the assumed VELS)
#
# `hammer.py` fits $\tau_0$ (log-linear over the keys) and $\beta$ with the velocities
# of the four dynamics fixed at `VELS` = 0.25, 0.5, 0.75, 1 -- an assumption, they were
# never measured.  This script tests that assumption in two ways:
#
# 1. Bass estimate: for $f_0 \tau \ll 1$ the pulse acts as a pure impulse, so
#    $A_0 \propto v$ regardless of $\tau_0, \beta$.  The ratios $A_0(\text{dyn}) / A_0(f)$
#    over the lowest keys are then the velocity ratios directly.
# 2. Joint fit: step 3 again, with $v_p, v_{mp}, v_{mf}$ free as well ($v_f = 1$ fixes
#    the scale, $c_0$ absorbs the rest).
#
# Both are compared with the assumed VELS by the rms residual per dynamic.  Output:
# `step3_fit_vels.npz`; `step3_fit.npz` is not touched.
#
# | Name | Value | Meaning |
# | --- | --- | --- |
# | `N_BASS` | 12 | lowest keys used for the bass estimate |
# | `VELS_ASSUMED` | .25 .5 .75 1 | the fixed velocities of `hammer.py` |
# | `STEPS`, `LR` | 3000, 0.02 | Adam, as in `hammer.py` |

# %%
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

from model import calc_tau, hammer2free

N_BASS       = 12
VELS_ASSUMED = np.array([0.25, 0.5, 0.75, 1.0])
STEPS, LR    = 3000, 0.02
T_START      = 10e-3                     # A_0 of step 2 is measured at onset + T_START
DYNS         = ["p", "mp", "mf", "f"]

# %%
# ---------- data of step 2 ----------
fit2  = np.load("step2_fit.npz", allow_pickle=True)
notes = fit2["notes"]
M     = len(notes)
f0    = jnp.asarray(fit2["f0"])
logA0 = jnp.log(jnp.asarray(fit2["A0"]))          # (M, 4) free amplitude per key and dynamic
ok    = jnp.asarray(fit2["ok"])                   # (M, 4) step 2 succeeded
sig   = jnp.asarray(fit2["sigma"])                # (M,) decay of the fundamental
keys  = jnp.arange(M) / (M - 1)                   # 0 at the lowest, 1 at the top key

# ---------- 1) bass estimate ----------
# A_0(dyn) / A_0(f) per key, averaged over the lowest N_BASS keys (only where both are ok)
A0   = np.asarray(fit2["A0"])
okn  = np.asarray(fit2["ok"])
rat  = np.where(okn[:N_BASS] & okn[:N_BASS, 3:], A0[:N_BASS] / A0[:N_BASS, 3:], np.nan)
vels_bass = np.nanmean(rat, 0)
print("bass estimate of the velocities (f = 1):", np.round(vels_bass, 3))
print("  spread over the keys (std):          ", np.round(np.nanstd(rat, 0), 3))


# %%
# ---------- model: as log_amp in hammer.py, with the velocities as an argument ----------

def tau0_of_key(theta):
    lo, hi = theta["log_tau0"]
    return jnp.exp(lo + (hi - lo) * keys)


def vels_of(theta):
    """Velocities of p, mp, mf, f.  Free ones come from theta["log_v"] (p, mp, mf), f = 1."""
    if "log_v" in theta:
        return jnp.concatenate([jnp.exp(theta["log_v"]), jnp.ones(1)])
    return jnp.asarray(theta["vels"])


def log_amp(theta):
    """(M, 4) log|A_0| for c_0 = 1, carried from the hand-over to t = T_START."""
    beta, tau0, vels = jnp.exp(theta["log_beta"]), tau0_of_key(theta), vels_of(theta)

    def one(f, t0, s, v):
        tau = calc_tau(v, dict(tau_0=t0, beta=beta))
        return jnp.log(jnp.abs(hammer2free(f, 1.0, v, tau)[0])) - s * (T_START - tau)

    return jax.vmap(jax.vmap(one, (None, None, None, 0)), (0, 0, 0, None))(f0, tau0, sig, vels)


def residual(theta):
    """(M, 4) log residual with the per-key mean removed (c_0 cancelled), 0 where not ok."""
    r = jnp.where(ok, logA0 - log_amp(theta), 0.0)
    r = r - (r.sum(1) / ok.sum(1))[:, None]
    return jnp.where(ok, r, 0.0)


def loss(free, fixed):
    return jnp.sum(residual({**fixed, **free}) ** 2) / ok.sum()


def run(free, fixed):
    """Adam on the free parameters, the fixed ones stay."""
    opt   = optax.adam(LR)
    state = opt.init(free)

    @jax.jit
    def step(free, state):
        l, g = jax.value_and_grad(loss)(free, fixed)
        upd, state = opt.update(g, state, free)
        return optax.apply_updates(free, upd), state, l

    for _ in range(STEPS):
        free, state, l = step(free, state)
    return {**fixed, **free}, float(l)


# %%
# ---------- fits ----------
# Start as in hammer.py: tau_0 long in the bass, inside the main lobe in the treble.
start = dict(log_tau0=jnp.log(jnp.array([8e-3, 0.4e-3])), log_beta=jnp.log(0.3))

results = {}
# a) assumed VELS (reproduces hammer.py)
results["assumed"] = run(start, dict(vels=VELS_ASSUMED))
# b) VELS fixed at the bass estimate
results["bass"] = run(start, dict(vels=vels_bass))
# c) velocities free, started at the assumed values
results["free"] = run({**start, "log_v": jnp.log(jnp.asarray(VELS_ASSUMED[:3]))}, {})


def rms_per_dyn(theta):
    r = residual(theta)
    return np.sqrt(np.asarray((r ** 2).sum(0) / ok.sum(0)))


print(f"\n{'':>8} {'loss':>8}  {'velocities p mp mf f':>26}  {'beta':>6}  "
      f"{'tau_0 bottom/top (ms)':>22}  rms per dynamic {DYNS}")
for name, (th, l) in results.items():
    t0 = tau0_of_key(th)
    print(f"{name:>8} {l:8.4f}  {np.array2string(np.asarray(vels_of(th)), precision=3):>26}  "
          f"{float(jnp.exp(th['log_beta'])):6.3f}  "
          f"{float(t0[0]) * 1e3:9.3f} / {float(t0[-1]) * 1e3:6.3f}     "
          f"{np.round(rms_per_dyn(th), 3)}")

# %%
# Residual per key and dynamic for the three variants: a systematic offset of one
# dynamic (a line away from 0) means its velocity or the hammer model is off.
fig, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True, sharey=True)
for a, (name, (th, l)) in zip(ax, results.items()):
    r = np.asarray(residual(th))
    for j, d in enumerate(DYNS):
        a.plot(np.where(np.asarray(ok[:, j]), r[:, j], np.nan), ".-", label=d)
    a.axhline(0, color="k", lw=0.5)
    a.set_ylabel("log residual")
    a.set_title(f"{name}: velocities {np.round(np.asarray(vels_of(th)), 3)}, loss {l:.4f}")
ax[0].legend()
ax[-1].set_xticks(range(0, M, 6)); ax[-1].set_xticklabels(notes[::6], rotation=60)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/step3_vels.png", dpi=90); plt.show()

# %%
# Save the fit with free velocities; c_0 is the per-key mean residual, as in hammer.py.
th = results["free"][0]
c0 = jnp.exp(jnp.where(ok, logA0 - log_amp(th), 0.0).sum(1) / ok.sum(1))
np.savez("step3_fit_vels.npz", notes=notes, f0=np.asarray(f0), vels=np.asarray(vels_of(th)),
         vels_bass=vels_bass, tau0=np.asarray(tau0_of_key(th)), beta=float(jnp.exp(th["log_beta"])),
         c0=np.asarray(c0), log_tau0=np.asarray(th["log_tau0"]))
print("-> step3_fit_vels.npz")
