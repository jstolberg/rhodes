# %% [markdown]
# # Fitting pickup and fundamental (step 2 of fitting.py)
#
# **Model.**  One freely decaying oscillation of the tine tip,
# $x(t) = p_o + A_0 e^{-\sigma_0 t} \sin 2\pi f_0 t$, seen through the pickup.
# The pickup is nonlinear, so its voltage contains $f_0$ and the harmonics
# $2 f_0, 3 f_0, \ldots$; how strong they are depends on where the tine sits in
# front of the pickup ($p_o$, $p_d$).
#
# **Comparison.**  Recording and model are cut into frames; in each frame the
# magnitude at $f_0 \ldots 5 f_0$ is measured,
# $$H[n, k] = \int w(t - t_k)\, \epsilon(t)\, e^{-2\pi i\, n f_0 t}\, \mathrm{d}t .$$
# The loss is the mean squared difference of the log magnitudes over the cells
# above the noise floor $\eta$ (mask $m$),
# $$L = \frac{1}{\sum m}\sum_{n,k} m[n,k]\big(\log(|H_t| + \eta) - \log(\kappa
#   |H_s| + \eta)\big)^2 .$$
# $\kappa$ (overall level) is not fitted but the mean log difference.  All
# definitions are in `step2_lib.py`.
#
# **Fitted.**  $p_o$, $p_d$, $\sigma_0$ per key, $A_0$ per key and dynamic.
# $A_0$ is the amplitude at $t_0$ = onset + 10 ms (hammer gone); step 3 carries it
# back to the moment the hammer lets go.
#
# **Three stages**, starting from the best start of `step2_init.py`:
#
# * **A** -- $p_o$, $p_d$, $\sigma_0$ and $A_0$ of the f recordings.
# * **B** -- $A_0$ of p, mp, mf, everything else held.  Each recording gets one
#   number; its loss has several valleys in $A_0$, so each starts from the best of
#   a scan over `A0_SCAN`.
# * **C** -- all parameters on all recordings together.
#
# | Name | Value | Meaning |
# | --- | --- | --- |
# | `TAG` | real / syn | recordings or synthetic test data (start values must exist for it) |
# | `STEPS_A`, `STEPS_B`, `STEPS_C` | 300, 100, 300 | Adam steps of the three stages |
# | `A0_SCAN` | 32 values, 0.01-5.7 mm | $A_0$ values tried per recording before stage B |

# %%
import pickle
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from step2_lib import (DISP_MAX, DYNS, J_FIT, N_DYN, N_HARM, features, fit, frame_times,
                       load_keys, loss, masked_mean, predict, raw_a0, report_mask, theta_ref, unpack)
from step2_init import init_path, winner

TAG = "syn"
STEPS_A, STEPS_B, STEPS_C = 300, 100, 300
A0_SCAN = np.geomspace(0.01e-3, 0.95 * DISP_MAX, 32)


# %%
# ---------- the three stages ----------

def stage_A(theta0, keys, Ht, eta, mask):
    """p_o, p_d, sigma_0, A_0(f), fitted on the f recordings only."""
    sl = np.s_[:, J_FIT:J_FIT + 1]                              # only the FIT_DYN column
    logHt, eta_f, mask_f = jnp.log(Ht + eta)[sl], eta[sl], mask[sl]
    theta, hist, _ = fit(theta0, lambda th: loss(th, keys, logHt, eta_f, mask_f)[0], STEPS_A, label="A: ")
    return theta, loss(theta, keys, logHt, eta_f, mask_f)[1], hist


def stage_B(theta_A, log_kappa, keys, Ht, eta, mask):
    """A_0 of the other dynamics; geometry, sigma_0 and kappa held from stage A."""
    others = [j for j in range(N_DYN) if j != J_FIT]            # p, mp, mf
    logHt, eta_o, mask_o = jnp.log(Ht + eta)[:, others], eta[:, others], mask[:, others]
    fixed = {k: v for k, v in theta_A.items() if k != "A0_raw"}
    Mk = len(keys["f0s"])

    def cell_loss(A0_raw):                                      # loss of every recording (Mk, 3)
        pred = jnp.logaddexp(log_kappa + predict(dict(fixed, A0_raw=A0_raw), keys), jnp.log(eta_o))
        return masked_mean((logHt - pred) ** 2, mask_o, axis=(2, 3))

    # scan: try every A0_SCAN value for all recordings at once, keep the best per recording
    scan = jax.jit(jax.vmap(lambda a0: cell_loss(jnp.full((Mk, len(others)), raw_a0(a0)))))(A0_SCAN)
    A0 = A0_SCAN[np.asarray(scan).argmin(0)]                    # (Mk, 3)

    # then Adam from there
    loss_B = lambda q: loss(dict(fixed, **q), keys, logHt, eta_o, mask_o, log_kappa)[0]
    q, hist, _ = fit(dict(A0_raw=raw_a0(A0)), loss_B, STEPS_B, label="B: ")

    # put the f column of stage A back in its place
    A0_raw = jnp.concatenate([q["A0_raw"][:, :J_FIT], theta_A["A0_raw"], q["A0_raw"][:, J_FIT:]], axis=1)
    return dict(fixed, A0_raw=A0_raw), hist


def stage_C(theta, keys, Ht, eta, mask):
    """All parameters on all recordings."""
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
# ---------- reporting ----------

def report(theta, log_kappa, true=None):
    """Print the fitted parameters (and, for synthetic data, the error to the truth)."""
    g, A0 = unpack(theta)
    mm = lambda v: np.array2string(np.asarray(v) * 1e3, precision=2, max_line_width=200)
    print(f"kappa={float(jnp.exp(log_kappa)):.3e}")
    print(f"  p_d (mm): {mm(g['p_d'])}\n  p_o (mm): {mm(g['p_o'])}")
    print(f"  sigma (1/s): {np.array2string(np.asarray(g['sig']), precision=3, max_line_width=200)}")
    print(f"  A_0 (mm), rows keys, cols {DYNS}:\n{mm(A0)}")
    print("  A_0 ratios to f:\n", np.round(np.asarray(A0[:, :J_FIT] / A0[:, J_FIT:J_FIT + 1]), 3))
    if true is not None:
        gt, A0t = unpack(true)
        print("  " + "  ".join(f"{k}: max rel err {float(jnp.max(jnp.abs(g[k] / gt[k] - 1))):.2%}"
                               for k in ("p_d", "p_o", "sig")))
        print(f"  A_0: max rel err {float(jnp.max(jnp.abs(A0 / A0t - 1))):.2%}")
    return g, A0


def residuals(theta, log_kappa, keys, Ht, eta, mask):
    """Print the rms log residual per harmonic, key and dynamic."""
    d = jnp.where(mask, jnp.log(Ht + eta) - jnp.logaddexp(log_kappa + predict(theta, keys), jnp.log(eta)), 0.0)
    rms = lambda ax: np.asarray(jnp.sqrt(masked_mean(d**2, mask, ax)))
    print("rms log residual per harmonic:", np.round(rms((0, 1, 3)), 3))
    print("per key:", np.round(rms((1, 2, 3)), 3), " per dynamic:", np.round(rms((0, 2, 3)), 3))


def plot_fit(theta, log_kappa, keys, Ht, eta, mask, i, j):
    """Measured (solid) vs fitted (dashed) harmonic magnitudes over time of one recording;
    grey dots: below the noise, not in the loss."""
    pred = np.asarray(jnp.logaddexp(log_kappa + predict(theta, keys), jnp.log(eta)))[i, j]
    T, starts = frame_times(keys["f0s"][i], keys["tfit"][i])
    plt.figure(figsize=(8, 4))
    for n in range(N_HARM):
        db = 20 * np.log10(Ht[i, j, n] + eta[i, j, n])
        plt.plot(starts, db, color=f"C{n}", label=f"H{n+1}")
        plt.plot(starts, 20 * pred[n] / np.log(10), "--", color=f"C{n}")
        plt.plot(starts[~mask[i, j, n]], db[~mask[i, j, n]], ".", color="0.6")
    plt.title(f"{keys['notes'][i]}-{DYNS[j]}: target (solid), fit (dashed), masked (grey)")
    plt.xlabel("t after t_0 (s)"); plt.ylabel("dB"); plt.legend(ncol=5)
    plt.tight_layout(); plt.show()


def plot_params(hists, theta, keys, true=None):
    """Loss history of the three stages, A_0 and sigma_0 over the keys (o: truth)."""
    g, A0 = unpack(theta)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].semilogy(np.concatenate(hists)); ax[0].set_xlabel("step (A | B | C)"); ax[0].set_ylabel("loss")
    for j in range(N_DYN):
        ax[1].plot(np.asarray(A0[:, j]) * 1e3, "x-", color=f"C{j}", label=DYNS[j])
    ax[2].plot(np.asarray(g["sig"]), "x-", color="k")
    if true is not None:
        gt, A0t = unpack(true)
        for j in range(N_DYN):
            ax[1].plot(np.asarray(A0t[:, j]) * 1e3, "o", color=f"C{j}", mfc="none")
        ax[2].plot(np.asarray(gt["sig"]), "o", color="k", mfc="none")
    ax[1].set_ylabel("A_0 (mm)" + ("  (o = true)" if true is not None else "")); ax[1].legend()
    ax[2].set_ylabel("sigma_0 (1/s)")
    for a_ in ax[1:]:
        a_.set_xticks(range(len(keys["notes"]))); a_.set_xticklabels(keys["notes"], rotation=60)
    plt.tight_layout(); plt.show()


# %%
# ---------- run ----------
keys = load_keys()
Ht, eta, mask = features(TAG, keys)
report_mask(mask)

with open(init_path(TAG), "rb") as f:
    cache = pickle.load(f)
assert (cache["notes"] == keys["notes"]).all(), "start values are for other keys"
theta0 = dict(winner(cache), log_sig=jnp.log(jnp.asarray(keys["sig1"])))   # sigma_0 starts at step 1

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
# ok: at least 3 frames of the fundamental above the noise.
if TAG == "real":
    np.savez("step2_fit.npz", notes=keys["notes"], f0=keys["f0s"], A0=np.asarray(A0),
             sigma=np.asarray(g["sig"]), p_d=np.asarray(g["p_d"]), p_o=np.asarray(g["p_o"]),
             kappa=float(jnp.exp(lk)), tfit=keys["tfit"], ok=mask[:, :, 0].sum(-1) >= 3)
