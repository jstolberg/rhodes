# %% [markdown]
# # Fitting pickup and fundamental (step 2 of fitting.py) -- harmonic projection
#
# The model in this step is a single free mode, $x(t) = p_o + A_0 e^{-\sigma_0 t}
# \sin 2\pi f_0 t$, seen through the pickup.  The pickup is a static
# nonlinearity, so the output has energy only at $n f_0$.  Rather than an STFT
# over all bins, both target and synthetic signals are projected onto the
# fundamental and its first four harmonics in a sequence of Hann-windowed frames,
# $$H[n, k] = \int w(t - t_k)\, \epsilon(t)\, e^{-2\pi i\, n f_0 t}\, \mathrm{d}t,
#   \qquad n = 1 \ldots 5,$$
# and the loss is the mean square of the log-magnitude difference, with a
# per-frame noise floor $\eta$ and a mask,
# $$L = \frac{1}{\sum m}\sum_{n,k} m[n,k]\big(\log(|H_t| + \eta) - \log(\kappa
#   |H_s| + \eta)\big)^2 .$$
# $\kappa$ is one global gain (the recording level and the pickup constant are
# both arbitrary); it is not a free parameter but the closed-form level match
# $\log\kappa = \overline{\log|H_t| - \log|H_s|}$ over the kept cells.  The
# coil's RLC filter is ignored for now.
#
# The pickup surface is fixed from photos (fitting.py: $r \approx 4$ mm,
# $a \approx 1.3$ mm, $m \approx 1$), so $\Psi(x, z)$ is tabulated **once**;
# the per-key $(p_o, p_d)$ enter only through the lookup.  $A_0$ is the
# amplitude at $t_0$ = onset + `T_START`, after the hammer has left; step 3
# needs the amplitude at handover, which differs by $e^{\sigma_0 (t_0 - \tau)}$.
# The decay of step 1 sets the frame span per key and the starting $\sigma_0$.
# Definitions and hyperparameters: `step2_lib.py`.
#
# ## Three stages
#
# $p_o$, $p_d$, $\sigma_0$ are per key, $A_0$ per key and dynamic.  The
# starting geometry comes from `step2_init.py` (multi-start, cached).
#
# * **A** -- $p_o$, $p_d$, $\sigma_0$, $A_0(f)$ from the f samples alone.
# * **B** -- $A_0$ of p, mp, mf per cell with everything else fixed (one
#   parameter per cell, started from the best of a log-spaced scan).
# * **C** -- all parameters, all cells, fine-tuned together.
#
# | Name | Value | Meaning |
# | --- | --- | --- |
# | `TAG` | real / syn | which target features and init |
# | `STEPS_A`, `STEPS_B`, `STEPS_C` | 300, 100, 300 | Adam steps of the three stages |
# | `A0_SCAN` | 32 values, 0.01-5.7 mm | log-spaced $A_0$ scan per cell that starts stage B |

# %%
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from step2_lib import *
from step2_init import init_path, winner

TAG = "syn"
STEPS_A, STEPS_B, STEPS_C = 300, 100, 300
A0_SCAN = np.geomspace(0.01e-3, 0.95 * DISP_MAX, 32)     # stage B start: per-cell scan of A_0


def stage_A(theta0, keys, Ht, eta, mask):
    """p_o, p_d, sigma_0, A_0(f) from the f cells."""
    sl = np.s_[:, J_FIT:J_FIT + 1]
    logHt, eta_f, mask_f = jnp.log(Ht + eta)[sl], eta[sl], mask[sl]
    theta, hist, _ = fit(theta0, lambda th: loss(th, keys, logHt, eta_f, mask_f)[0], STEPS_A, label="A: ")
    return theta, loss(theta, keys, logHt, eta_f, mask_f)[1], hist


def stage_B(theta_A, log_kappa, keys, Ht, eta, mask):
    """A_0 of the other dynamics, one parameter per cell, geometry / sigma / kappa fixed.
    The loss of a cell is rugged in A_0 (harmonics have nulls as a function of amplitude),
    so each cell starts from the best of a log-spaced scan A0_SCAN, then Adam."""
    others = [j for j in range(N_DYN) if j != J_FIT]
    logHt, eta_o, mask_o = jnp.log(Ht + eta)[:, others], eta[:, others], mask[:, others]
    fixed = {k: v for k, v in theta_A.items() if k != "A0_raw"}
    Mk = len(keys["f0s"])

    def cell_loss(A0_raw):                                    # (Mk, n_others) per-cell losses
        pred = jnp.logaddexp(log_kappa + predict(dict(fixed, A0_raw=A0_raw), keys), jnp.log(eta_o))
        return masked_mean((logHt - pred) ** 2, mask_o, axis=(2, 3))

    scan = jax.jit(jax.vmap(lambda a0: cell_loss(jnp.full((Mk, len(others)), raw_a0(a0)))))(A0_SCAN)
    A0 = A0_SCAN[np.asarray(scan).argmin(0)]                  # (Mk, n_others)
    loss_B = lambda q: loss(dict(fixed, **q), keys, logHt, eta_o, mask_o, log_kappa)[0]
    q, hist, _ = fit(dict(A0_raw=raw_a0(A0)), loss_B, STEPS_B, label="B: ")
    A0_raw = jnp.concatenate([q["A0_raw"][:, :J_FIT], theta_A["A0_raw"], q["A0_raw"][:, J_FIT:]], axis=1)
    return dict(fixed, A0_raw=A0_raw), hist


def stage_C(theta, keys, Ht, eta, mask):
    """All parameters on all cells."""
    logHt = jnp.log(Ht + eta)
    theta, hist, _ = fit(theta, lambda th: loss(th, keys, logHt, eta, mask)[0], STEPS_C, label="C: ")
    return theta, loss(theta, keys, logHt, eta, mask)[1], hist


def fit_all(theta0, keys, Ht, eta, mask):
    t0 = time.time()
    theta_A, lk_A, hA = stage_A(theta0, keys, Ht, eta, mask)
    theta_B, hB       = stage_B(theta_A, lk_A, keys, Ht, eta, mask)
    theta_C, lk_C, hC = stage_C(theta_B, keys, Ht, eta, mask)
    print(f"total {time.time() - t0:.0f} s")
    return theta_C, lk_C, (hA, hB, hC)


# %%
keys = load_keys()
Ht, eta, mask = features(TAG, keys)
report_mask(mask)
with open(init_path(TAG), "rb") as f:
    cache = pickle.load(f)
assert (cache["notes"] == keys["notes"]).all(), "init cache is for other keys"
theta0 = dict(winner(cache), log_sig=jnp.log(jnp.asarray(keys["sig1"])))

true = theta_ref(keys) if TAG == "syn" else None
if true is not None:
    print(f"loss at truth: {float(loss(true, keys, jnp.log(Ht + eta), eta, mask)[0]):.4f}")

theta, lk, hists = fit_all(theta0, keys, Ht, eta, mask)
print("fit:"); g, A0 = report(theta, lk, true)
residuals(theta, lk, keys, Ht, eta, mask)

# %%
plot_params(hists, theta, keys, true)
plot_fit(theta, lk, keys, Ht, eta, mask, len(keys["f0s"]) // 2, J_FIT)

# %%
# Hand-over to step 3: A_0 (Mk, N_DYN) and sigma_0 (Mk,) with the geometry.
if TAG == "real":
    np.savez("step2_fit.npz", notes=keys["notes"], f0=keys["f0s"], A0=np.asarray(A0),
             sigma=np.asarray(g["sig"]), p_d=np.asarray(g["p_d"]), p_o=np.asarray(g["p_o"]),
             kappa=float(jnp.exp(lk)), tfit=keys["tfit"], ok=mask[:, :, 0].sum(-1) >= 3)
