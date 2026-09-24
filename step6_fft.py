# %% [markdown]
# # Fitting the inharmonic modes (step 6 of fitting.py) -- line projection
#
# Fixed: everything of steps 2 and 3 ($f_0$, $\sigma_0$, $p_o$, $p_d$, $\kappa$,
# $\tau_0$, $\beta$, $c_0$) and the mode frequencies $f_n$ of step 5.  Free: the
# excitation coefficient $c_n$ and decay $\sigma_n$ of every mode, per key.
#
# After contact the tine tip moves as the sum of all modes with the amplitude and
# phase the hammer hands over (`hammer2free`, as in `alpha` of `model.py`),
# $$x(t) = p_o + \sum_n A_n e^{-\sigma_n (t-\tau)} \sin\big(2\pi f_n (t-\tau) + \varphi_n\big),$$
# seen through the pickup, $\varepsilon = -\kappa\, d\Psi/dt$.  The intermodulation
# products that spoil a free extraction are produced by the model itself, so they
# need not be told apart.
#
# As in step 2, target and model are projected onto a set of lines in Hann frames,
# here the mode lines $f_n$ over the first `T_MODES` after $t_0$ (areas S2 and S7 of
# fitting.py); the loss is the masked mean squared log-magnitude difference.  $\kappa$
# is **fixed** from step 2 (a closed-form level would lose the modes' level relative
# to the fundamental).  $c_n$ does not depend on velocity, so all four dynamics are fit
# together and the hammer alone sets their differences: a residual that differs
# systematically between dynamics points at the hammer model.
#
# The hammer excites a stiff mode with far less displacement per unit $c$ than the
# fundamental, so $c_n$ spans many decades.  The start is therefore a scan over the
# free *amplitude* $A_n / A_0$ at $v = 1$, converted to $c_n$ (`c_for_amp`).
#
# | Name | Value | Meaning |
# | --- | --- | --- |
# | `TAG` | real / syn | recordings, or waves rendered from `TRUE_AMP`, `TRUE_SIG` |
# | `N_PERIODS_M` | 64 | frame length in periods of $f_0$: separates 7.1 $f_0$ from 7 $f_0$ by 6 bins |
# | `K_M`, `T_MODES` | 8, 0.8 s | frames, spread over this span after $t_0$ = onset + `T_START` |
# | `NOISE_BINS` | 3 | noise floor: projection this many bins beside each line (smaller side) |
# | `AMP_SCAN` | 20 values, 1e-4 - 0.3 | start: $A_n / A_0$ tried per key and line |
# | `SIG_INIT` | 2 | start: $\sigma_n = 2\,\sigma_0$ |
# | `STEPS`, `LR_M` | 300, 0.05 | Adam |
# | `TRUE_AMP`, `TRUE_SIG` | see code | syn: $A_n / A_0$ and $\sigma_n / \sigma_0$ of the truth |

# %%
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from model import calc_tau, hammer2free, psi_lookup
from step2_lib import (DYNS, FS_SYN, NOISE_MARGIN, NOISE_SYN, PRE_SYN, T_START, TABLE, VELS,
                       fit, hann, load_wave, onset)

TAG        = "real"
N_PERIODS_M, K_M, T_MODES = 64, 8, 0.8
NOISE_BINS = 6
AMP_SCAN   = np.geomspace(1e-4, 0.3, 20)
SIG_INIT   = 2.0
STEPS, LR_M = 300, 0.05
TRUE_AMP   = np.array([0.05, 0.01, 0.005])     # one entry per RATIOS of step 5
TRUE_SIG   = np.array([3.0, 50.0, 100.0]) # [3.0, 6.0, 10.0]


# %%
# ---------- fixed parameters of steps 2 and 3, lines of step 5 ----------
fit2   = np.load("step2_fit.npz", allow_pickle=True)
fit3   = np.load("step3_fit.npz", allow_pickle=True)
lines5 = np.load("step5_lines.npz", allow_pickle=True)
notes  = lines5["notes"]
kappa, beta = float(fit2["kappa"]), float(fit3["beta"])
ok     = lines5["ok"]                                                   # (Mk, N)
f_lin  = np.where(ok, lines5["f_modes"], lines5["ratios"] * lines5["f0"][:, None])
Mk, N  = f_lin.shape

keys = []
for i, note in enumerate(notes):
    k = list(fit2["notes"]).index(note)
    keys.append(dict(note=note, f0=fit2["f0"][k], sig0=fit2["sigma"][k], c0=fit3["c0"][k],
                     tau0=fit3["tau0"][k], p_o=fit2["p_o"][k], p_d=fit2["p_d"][k], lines=f_lin[i]))

tau_max = max(float(calc_tau(VELS[0], dict(tau_0=k["tau0"], beta=beta))) for k in keys)
assert tau_max < T_START, "contact still running at t_0: frames would see the hammer"
print(f"{Mk} keys, {int(ok.sum())} of {ok.size} lines from step 5")


# %%
# ---------- model ----------

def modes(key, c_n, sig_n):
    """c, sigma, f of all modes of a key: the fixed fundamental first."""
    c   = jnp.concatenate([jnp.array([key["c0"]]), c_n])
    sig = jnp.concatenate([jnp.array([key["sig0"]]), sig_n])
    f   = jnp.concatenate([jnp.array([key["f0"]]), jnp.asarray(key["lines"])])
    return c, sig, f


def eps_modes(t, c, sig, f, vel, tau, p_o, p_d):
    """-dPsi/dt of the free tip motion after contact (t > tau), all modes."""
    A, phi = hammer2free(f, c, vel, tau)

    def psi(tt):
        x = p_o + jnp.sum(A * jnp.exp(-sig * (tt - tau)) * jnp.sin(2 * jnp.pi * f * (tt - tau) + phi))
        return psi_lookup(jnp.array([x, 0.0, p_d]), TABLE)
    d = jax.vmap(lambda tt: jax.jvp(psi, (tt,), (1.0,))[1])(t.ravel())
    return -d.reshape(t.shape)


def c_for_amp(key, rel):
    """c_n that gives free amplitudes rel * (free amplitude of the fundamental), at v = 1."""
    tau = calc_tau(1.0, dict(tau_0=key["tau0"], beta=beta))
    A0  = jnp.abs(hammer2free(key["f0"], key["c0"], 1.0, tau)[0])
    g   = jnp.abs(hammer2free(jnp.asarray(key["lines"]), 1.0, 1.0, tau)[0])
    return rel * A0 / jnp.maximum(g, 1e-30)


# ---------- framing and projection (shared by target and model) ----------

def frames(f0):
    """Frame length T and the K_M frame starts (s after t_0)."""
    T = N_PERIODS_M / f0
    return T, (T_MODES - T) / (K_M - 1) * np.arange(K_M)


def project(seg, lines, fs):
    """|H| (N, K) of the frames seg (K, L): Hann window, projected onto the lines (N,)."""
    L = seg.shape[-1]
    E = jnp.exp(-2j * jnp.pi * jnp.outer(lines, jnp.arange(L) / fs))
    return jnp.abs(jnp.einsum("nl,kl->nk", E, jnp.asarray(hann(L)) * seg)) / fs


def target(x, fs, key):
    """|H_t| and noise floor (N, K) of one recording."""
    T, starts = frames(key["f0"])
    L  = int(round(T * fs))
    i0 = onset(x) + int(round(T_START * fs))
    seg = np.stack([x[i0 + int(round(s * fs)):][:L] for s in starts])
    df  = NOISE_BINS / T
    eta = jnp.minimum(project(seg, key["lines"] - df, fs), project(seg, key["lines"] + df, fs))
    return np.asarray(project(seg, key["lines"], fs)), np.asarray(eta)


def model_H(key, fs, c_n, sig_n):
    """|H_s| (n_dyn, N, K) of one key at the four velocities."""
    T, starts = frames(key["f0"])
    t = jnp.asarray(T_START + starts[:, None] + np.arange(int(round(T * fs)))[None, :] / fs)

    @jax.checkpoint                                           # recompute in backward: less memory
    def one(vel, c, sig, f):
        tau = calc_tau(vel, dict(tau_0=key["tau0"], beta=beta))
        return project(kappa * eps_modes(t, c, sig, f, vel, tau, key["p_o"], key["p_d"]), key["lines"], fs)

    return jax.vmap(one, (0, None, None, None))(jnp.asarray(VELS), *modes(key, c_n, sig_n))


# %%
# ---------- targets ----------

def synth_wave(key, vel, c_n, sig_n, rng):
    """A 'recording' rendered by the model: PRE_SYN s silence, white noise as in step 2."""
    t = jnp.arange(int((T_START + T_MODES + 0.1) * FS_SYN)) / FS_SYN
    tau = calc_tau(vel, dict(tau_0=key["tau0"], beta=beta))
    x = kappa * np.asarray(eps_modes(t, *modes(key, c_n, sig_n), vel, tau, key["p_o"], key["p_d"]))
    x = np.concatenate([np.zeros(int(PRE_SYN * FS_SYN)), x])
    return x + NOISE_SYN * rng.standard_normal(len(x))


def features(tag, truth=None):
    """Per key (|H_t|, eta, mask, fs), arrays (n_dyn, N, K)."""
    rng, feats = np.random.default_rng(1), []
    for i, key in enumerate(keys):
        H, E = [], []
        for j, dyn in enumerate(DYNS):
            if tag == "real":
                x, fs = load_wave(key["note"], dyn)
            else:
                c_n = ok[i] * jnp.exp(truth["log_c"][i])
                x, fs = synth_wave(key, VELS[j], c_n, jnp.exp(truth["log_sig"][i]), rng), FS_SYN
            h, e = target(x, fs, key)
            H.append(h); E.append(e)
        H, E = np.stack(H), np.stack(E)
        feats.append((H, E, (H > NOISE_MARGIN * E) & ok[i][None, :, None], fs))
    return feats


# %%
# ---------- loss, start, fit ----------
# theta: log_c, log_sig (Mk, N).  Lines step 5 rejected keep c_n = 0 and are masked.

def cell_errors(theta, feats):
    """Per key: squared log residuals (n_dyn, N, K) on the kept cells, and the mask."""
    out = []
    for i, key in enumerate(keys):
        Ht, eta, mask, fs = feats[i]
        Hs = model_H(key, fs, ok[i] * jnp.exp(theta["log_c"][i]), jnp.exp(theta["log_sig"][i]))
        pred = jnp.logaddexp(jnp.log(Hs + 1e-30), jnp.log(eta))   # tiny offset: finite gradient
        out.append((jnp.where(mask, (jnp.log(Ht + eta) - pred) ** 2, 0.0), mask))
    return out


def loss(theta, feats):
    e = cell_errors(theta, feats)
    return sum(d.sum() for d, _ in e) / max(sum(int(m.sum()) for _, m in e), 1)


def line_loss(theta, feats):
    """(Mk, N) mean squared residual per key and line, over dynamics and frames."""
    return jnp.stack([d.sum((0, 2)) / np.maximum(m.sum((0, 2)), 1) for d, m in cell_errors(theta, feats)])


def start(feats):
    """Per key and line the best A_n / A_0 of AMP_SCAN; sigma_n = SIG_INIT * sigma_0."""
    log_sig = jnp.stack([jnp.full(N, jnp.log(SIG_INIT * k["sig0"])) for k in keys])
    ll   = jax.jit(lambda lc: line_loss(dict(log_c=lc, log_sig=log_sig), feats))
    cand = np.stack([np.stack([np.log(np.asarray(c_for_amp(k, a))) for k in keys]) for a in AMP_SCAN])
    L    = np.stack([np.asarray(ll(jnp.asarray(lc))) for lc in cand])        # (S, Mk, N)
    best = cand[L.argmin(0), np.arange(Mk)[:, None], np.arange(N)[None, :]]
    return dict(log_c=jnp.asarray(best), log_sig=log_sig)


def report(theta, truth=None):
    for i, key in enumerate(keys):
        amp = np.where(ok[i], np.exp(theta["log_c"][i]) / np.asarray(c_for_amp(key, 1.0)), np.nan)
        sig = np.where(ok[i], np.exp(theta["log_sig"][i]), np.nan)
        print(f"  {key['note']:>4}: A_n/A_0 {np.round(amp, 4)}   sigma_n (1/s) {np.round(sig, 2)}")
    if truth is not None:
        err = lambda k: np.abs(np.exp(theta[k] - truth[k]) - 1)[ok]
        print(f"  max rel err: c_n {err('log_c').max():.2%}, sigma_n {err('log_sig').max():.2%}")


# %%
truth = None
if TAG == "syn":
    truth = dict(log_c=jnp.stack([jnp.log(c_for_amp(k, TRUE_AMP[:N])) for k in keys]),
                 log_sig=jnp.stack([jnp.log(TRUE_SIG[:N] * k["sig0"]) for k in keys]))
feats = features(TAG, truth)
print("kept frames per line (all keys, dynamics):", sum(m.sum((0, 2)) for _, _, m, _ in feats))
if truth is not None:
    print(f"loss at truth: {float(loss(truth, feats)):.4f}")

theta0 = start(feats)
print("start:"); report(theta0, truth)
theta, hist, _ = fit(theta0, lambda th: loss(th, feats), STEPS, lr=LR_M, label="modes: ")
print("fit:"); report(theta, truth)

# Hammer check: a residual that differs between dynamics is not explained by c_n.
e = cell_errors(theta, feats)
rms = np.sqrt(sum(np.asarray(d).sum((1, 2)) for d, _ in e) / np.maximum(sum(m.sum((1, 2)) for _, m in e), 1))
print("rms log residual per dynamic", DYNS, ":", np.round(rms, 3))

# %%
# Measured (solid) vs fitted (dashed) line magnitudes over time, one key, dynamic f;
# grey dots: below the noise margin, not in the loss.
i, j = Mk // 2, len(DYNS) - 1
Ht, eta, mask, fs = feats[i]
Hs = np.asarray(model_H(keys[i], fs, ok[i] * jnp.exp(theta["log_c"][i]), jnp.exp(theta["log_sig"][i])))
_, starts = frames(keys[i]["f0"])
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].semilogy(hist); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
for n in range(N):
    db = 20 * np.log10(Ht[j, n] + eta[j, n])
    ax[1].plot(starts, db, color=f"C{n}", label=f"{f_lin[i, n] / keys[i]['f0']:.2f} f_0")
    ax[1].plot(starts, 20 * np.log10(Hs[j, n] + eta[j, n]), "--", color=f"C{n}")
    ax[1].plot(starts[~mask[j, n]], db[~mask[j, n]], ".", color="0.6")
ax[1].set_xlabel("t after t_0 (s)"); ax[1].set_ylabel("dB"); ax[1].legend()
ax[1].set_title(f"{keys[i]['note']}-{DYNS[j]} ({TAG})")
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/step6_fit_{TAG}.png", dpi=90); plt.show()

# %%
# Hand-over to step6_render.py.
if TAG == "real":
    np.savez("step6_fit.npz", notes=notes, ratios=lines5["ratios"], f_modes=f_lin, ok=ok,
             c=np.asarray(ok * jnp.exp(theta["log_c"])), sig=np.asarray(jnp.exp(theta["log_sig"])))
    print("-> step6_fit.npz")

# %%
