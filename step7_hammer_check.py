# %% [markdown]
# # How do the modes grow with velocity? (hammer check after step 6)
#
# **Question.**  Step 6 fits the mode excitation $c_n$ separately for p, mp, mf, f.
# Multiplied with what the hammer model hands over, that gives the free amplitude
# $A_n(\text{dyn})$ that reproduces the recording -- a measurement that does not
# depend on the hammer model.  How well do simple hammer models predict how $A_n$
# grows from p to f, and does a small neural network (a learned correction on top of
# the physics) add anything?
#
# **Target.**  Per line (key, mode) and dynamic, the level relative to f,
# $$y = 20 \log_{10} A_n(\text{dyn}) - 20 \log_{10} A_n(f) \quad\text{[dB]},$$
# at the time $t_0$ the frames of step 6 start.  Relative to f, because $c_n$ of a
# key is unknown to any hammer model; it cancels in the difference (as $c_0$ in
# step 3).  Only dynamics with frames above the noise are used.
#
# **Variants** (all predict $y$ from $f_n$, $\tau_0$, $\beta$ and the velocities):
#
# | | Hammer model | Velocities | Free parameters |
# | --- | --- | --- | --- |
# | A | $\sin^2$ pulse (`hammer2free`), as now | assumed 0.25 ... 1 | 0 |
# | B | $\sin^2$ pulse | from the bass (step 2 $A_0$ ratios) | 0 |
# | C | pulse envelope without its nulls | from the bass | 0 |
# | D | $A_n \propto v^k$ | from the bass | 1 ($k$) |
# | E | D + small MLP $r(\log f_n\tau, \log v)$ | from the bass | ~300 |
#
# **Scatter check.**  At the end the targets are filtered by how many step-6 frames
# stood above the noise: if the scatter were measurement noise, it would shrink.
#
# **Evaluation.**  Leave one key out: every key is predicted by a model fitted to the
# other keys (only D and E fit anything).  Reported: rms and median absolute error in
# dB.  A variant with more parameters is only worth it if it predicts *unseen* keys
# better.

# %%
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from model import calc_tau, hammer2free
from step2_lib import DYNS, T_START, VELS

N_BASS    = 12                  # lowest keys for the velocity estimate
HIDDEN    = 16                  # MLP width (two hidden layers)
NN_STEPS  = 2000                # Adam steps per fit
NN_LR     = 1e-2
NN_L2     = 1.0                 # weight decay; weaker (1e-3 ... 1e-1) overfits: rms 19 ... 13 dB
JF        = DYNS.index("f")
DB        = 20 / np.log(10)     # natural log -> dB

# %%
# ---------- data ----------
fit2 = np.load("step2_fit.npz", allow_pickle=True)
fit3 = np.load("step3_fit.npz", allow_pickle=True)
fit6 = np.load("step6_fit.npz", allow_pickle=True)
notes2 = list(fit2["notes"])
beta   = float(fit3["beta"])

# Velocities from the bass: there f_0 tau << 1, the pulse is a pure kick and A_0 ~ v, so
# the A_0 ratios of step 2 are the velocity ratios.
A0, ok0 = fit2["A0"], fit2["ok"]
rat = np.where(ok0[:N_BASS] & ok0[:N_BASS, JF:JF + 1], A0[:N_BASS] / A0[:N_BASS, JF:JF + 1], np.nan)
VELS_BASS = np.nanmean(rat, 0)
print("velocities p, mp, mf, f  assumed:", VELS, "  from the bass:", np.round(VELS_BASS, 3))


def tau_of(tau0, v):
    return float(calc_tau(v, dict(tau_0=tau0, beta=beta)))


def g_exact(f, v, tau):
    """|free amplitude| per unit c the sin^2 pulse hands over (with its nulls)."""
    return abs(float(hammer2free(f, 1.0, v, tau)[0]))


def g_envelope(f, v, tau):
    """The same without the factor |sin(w tau / 2)|: the upper envelope, no nulls.
    (Valid for f tau > 1, which holds for every line here.)"""
    F0, Om, w = 2 * v / tau, 2 * np.pi / tau, 2 * np.pi * f
    return F0 * Om**2 / (w**2 * abs(w**2 - Om**2))


# One row per line (key, mode): what is needed to measure and to predict y.
rows = []
for m, note in enumerate(fit6["notes"]):
    i = notes2.index(note)
    for n in range(len(fit6["ratios"])):
        used = fit6["used"][m, :, n]                   # dynamics with frames above the noise
        if not fit6["ok"][m, n] or not used[JF] or used.sum() < 2:
            continue
        f_n, sig_n, tau0 = fit6["f_modes"][m, n], fit6["sig"][m, n], fit3["tau0"][i]
        # measured level at t_0: the fitted c_n through the hammer model step 6 used
        # (assumed velocities), decayed from the hand-over at tau to t_0
        lvl = np.full(len(DYNS), np.nan)
        for j, v in enumerate(VELS):
            if used[j]:
                tau = tau_of(tau0, v)
                lvl[j] = DB * (np.log(g_exact(f_n, v, tau) * fit6["c"][m, j, n]) - sig_n * (T_START - tau))
        rows.append(dict(note=str(note), key=m, n=n, ratio=float(fit6["ratios"][n]), f=f_n, sig=sig_n,
                         tau0=tau0, used=used, y=lvl - lvl[JF]))
print(f"{len(rows)} lines, {sum(int(r['used'].sum()) - 1 for r in rows)} targets (dynamics other than f)")


def predict_physics(r, vels, g):
    """y of one line for velocities vels and hammer function g (dB, per dynamic)."""
    lvl = np.array([DB * (np.log(g(r["f"], v, tau_of(r["tau0"], v))) - r["sig"] * (T_START - tau_of(r["tau0"], v)))
                    for v in vels])
    return lvl - lvl[JF]


# targets as flat arrays: line index, dynamic, measured y
T = [(k, j, r["y"][j]) for k, r in enumerate(rows) for j in range(len(DYNS)) if j != JF and r["used"][j]]
t_line, t_dyn, y = (np.array(a) for a in zip(*T))
t_key = np.array([rows[k]["key"] for k in t_line])

# %%
# ---------- variants ----------
pred_A = np.array([predict_physics(rows[k], VELS, g_exact)[j] for k, j in zip(t_line, t_dyn)])
pred_B = np.array([predict_physics(rows[k], VELS_BASS, g_exact)[j] for k, j in zip(t_line, t_dyn)])
pred_C = np.array([predict_physics(rows[k], VELS_BASS, g_envelope)[j] for k, j in zip(t_line, t_dyn)])
x_v    = DB * np.log(VELS_BASS[t_dyn] / VELS_BASS[JF])        # level ratio of the velocities (dB)


def fit_D(train):
    """k of A_n ~ v^k, least squares on the train targets."""
    return float(np.sum(y[train] * x_v[train]) / np.sum(x_v[train] ** 2))


# E: impulse (k = 1) plus a learned correction r(log f tau, log v) of the level, applied
# as r(dyn) - r(f) so c_n still cancels.
def features(k, j):
    r = rows[k]
    v = VELS_BASS[j]
    return np.array([np.log(r["f"] * tau_of(r["tau0"], v)), np.log(v)])


X_dyn = np.stack([features(k, j) for k, j in zip(t_line, t_dyn)])
X_f   = np.stack([features(k, JF) for k in t_line])
mu, sd = X_dyn.mean(0), X_dyn.std(0) + 1e-9                    # input standardisation


def mlp_init(key):
    sizes = [2, HIDDEN, HIDDEN, 1]
    keys = jax.random.split(key, len(sizes) - 1)
    return [dict(W=jax.random.normal(k, (a, b)) / np.sqrt(a), b=jnp.zeros(b))
            for k, a, b in zip(keys, sizes[:-1], sizes[1:])]


def mlp(params, x):
    h = (x - mu) / sd
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    return (h @ params[-1]["W"] + params[-1]["b"])[..., 0]


def fit_E(train):
    """Adam on the MLP: y ~ x_v + r(dyn) - r(f), with weight decay."""
    Xd, Xf, base, yt = X_dyn[train], X_f[train], x_v[train], y[train]

    def loss(p):
        pred = base + mlp(p, Xd) - mlp(p, Xf)
        l2 = sum(jnp.sum(layer["W"] ** 2) for layer in p)
        return jnp.mean((pred - yt) ** 2) + NN_L2 * l2

    params, opt = mlp_init(jax.random.PRNGKey(0)), optax.adam(NN_LR)
    state = opt.init(params)

    @jax.jit
    def step(p, s):
        g = jax.grad(loss)(p)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s

    for _ in range(NN_STEPS):
        params, state = step(params, state)
    return params


# %%
# ---------- leave one key out ----------
pred_D, pred_E = np.full_like(y, np.nan), np.full_like(y, np.nan)
k_per_fold = []
for key in np.unique(t_key):
    test, train = t_key == key, t_key != key
    k = fit_D(train)
    k_per_fold.append(k)
    pred_D[test] = k * x_v[test]
    p = fit_E(train)
    pred_E[test] = np.asarray(x_v[test] + mlp(p, X_dyn[test]) - mlp(p, X_f[test]))
print(f"D: k = {np.mean(k_per_fold):.2f} (range over folds {min(k_per_fold):.2f}-{max(k_per_fold):.2f}),"
      f" i.e. A_n ~ v^k; the sin^2 pulse beyond its main lobe gives 1 + 3 beta = {1 + 3 * beta:.2f}")

names = ["A  sin^2, assumed velocities", "B  sin^2, bass velocities", "C  envelope, bass velocities",
         "D  v^k (1 parameter)", "E  v + MLP correction"]
preds = [pred_A, pred_B, pred_C, pred_D, pred_E]
print(f"\n{'variant':32s} {'rms (dB)':>9} {'median |err|':>13} {'bias':>7}   (unseen keys)")
print(f"{'predict 0 dB (no growth)':32s} {np.sqrt(np.mean(y**2)):9.1f} {np.median(np.abs(y)):13.1f} {np.mean(-y):7.1f}")
for name, p in zip(names, preds):
    e = p - y
    print(f"{name:32s} {np.sqrt(np.mean(e**2)):9.1f} {np.median(np.abs(e)):13.1f} {np.mean(e):7.1f}")

# %%
# ---------- where does the scatter come from? ----------
# A target is only as good as the two c_n it compares.  How well c_n of a dynamic is
# determined depends on how many of the 8 frames of step 6 stood above the noise there.
# If the scatter is mostly measurement uncertainty, it has to shrink when only targets
# with many frames (at the dynamic and at f) are kept.
kept_t = np.array([min(fit6["kept"][rows[k]["key"], j, rows[k]["n"]],
                       fit6["kept"][rows[k]["key"], JF, rows[k]["n"]]) for k, j in zip(t_line, t_dyn)])
resid_D = y - np.mean(k_per_fold) * x_v                   # scatter around the power law D
print(f"\n{'frames >=':>9} {'targets':>8} {'lines':>6}  {'median y  p / mp / mf (dB)':>28} {'k':>5}"
      f" {'scatter: rms':>13} {'IQR':>6}")
for kmin in (1, 2, 4, 6, 8):
    s = kept_t >= kmin
    if s.sum() < 5:
        continue
    med = " / ".join(f"{np.median(y[s & (t_dyn == j)]):+5.1f}" if (s & (t_dyn == j)).any() else "   - "
                     for j in range(JF))
    k_s = float(np.sum(y[s] * x_v[s]) / np.sum(x_v[s] ** 2))
    r = y[s] - k_s * x_v[s]
    print(f"{kmin:>9} {s.sum():>8} {len(np.unique(t_line[s])):>6}  {med:>28} {k_s:5.2f}"
          f" {np.sqrt(np.mean(r**2)):10.1f} dB {np.subtract(*np.percentile(r, [75, 25])):5.1f}")

# %%
# ---------- plots ----------
# Left: measured level relative to f over the (bass) velocity; grey lines = one line
# each, black = median per dynamic, coloured = median prediction of each variant.
# Right: error of B (sin^2 pulse) and D (v^k) over f tau at p: the nulls of the pulse
# spectrum at f tau = 2, 3, 4, ... show up as scatter in B.
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
vb = VELS_BASS
for k, r in enumerate(rows):
    js = np.flatnonzero(r["used"])
    ax[0].plot(vb[js], r["y"][js], "-", color="0.8", lw=0.8)
med = [np.median(y[t_dyn == j]) for j in range(JF)]
ax[0].plot(vb[:JF], med, "ko-", lw=2, label="measured (median)")
for c, (name, p) in enumerate(zip(names, preds)):
    ax[0].plot(vb[:JF], [np.median(p[t_dyn == j]) for j in range(JF)], "o--", color=f"C{c}", label=name)
ax[0].set_xlabel("velocity (bass estimate)"); ax[0].set_ylabel("level relative to f (dB)")
ax[0].legend(fontsize=8); ax[0].set_ylim(-45, 30)

ftau_p = np.array([rows[k]["f"] * tau_of(rows[k]["tau0"], VELS[0]) for k in t_line])
sel = t_dyn == 0
ax[1].plot(ftau_p[sel], (pred_B - y)[sel], "o", color="C1", label="B  sin^2 pulse")
ax[1].plot(ftau_p[sel], (pred_D - y)[sel], "o", color="C3", mfc="none", label="D  v^k")
for nul in range(2, 18):
    ax[1].axvline(nul, color="0.9", lw=0.8)
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xscale("log"); ax[1].set_xlabel("f_n tau at p (grey: nulls of the sin^2 pulse)")
ax[1].set_ylabel("prediction - measured at p (dB)"); ax[1].legend(fontsize=8)
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/step7_hammer_check.png", dpi=90); plt.show()
