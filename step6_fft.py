# %% [markdown]
# # Fitting the inharmonic modes (step 6 of fitting.py) -- line projection
#
# **Which lines.**  The robust lines of step 5 (`step5_lines.npz`: accepted in at least
# 3 of the 4 dynamics, frequency = median over them).  Every key with at least one
# robust line is fit, if its frames fit into `T_MODES`.  Output: `step6_fit.npz`.
#
# **Per key.**  The keys share no free parameter, so each key is fit on its own: small
# jit graphs and progress per key.
#
# **$c_n$ per dynamic.**  In the model $c_n$ does not depend on the velocity; the hammer
# (the assumed `VELS` and the pulse shape) alone would set how the modes grow from p to
# f.  That does not match the recordings, so here $c_n$ is free per dynamic and the
# growth is measured instead: $c_n(\text{dyn}) / c_n(f)$ is 0 dB where the hammer model
# is right (hammer check).  $\sigma_n$ stays shared: in the linear modal model the decay
# is a property of the tine, not of the strike (`SIG_PER_DYN` frees it to check this).
#
# Fixed: everything of steps 2 and 3 ($f_0$, $\sigma_0$, $p_o$, $p_d$, $\kappa$,
# $\tau_0$, $\beta$, $c_0$) and the mode frequencies $f_n$ of step 5.  Free: the
# excitation coefficient $c_n$ (per dynamic) and decay $\sigma_n$ of every mode, per key.
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
# to the fundamental).
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
# | `NOISE_BINS` | 6 | noise floor: projection this many bins beside each line (smaller side) |
# | `AMP_SCAN` | 20 values, 1e-4 - 0.3 | start: $A_n / A_0$ tried per key and line |
# | `SIG_INIT` | 2 | start: $\sigma_n = 2\,\sigma_0$ |
# | `STEPS`, `LR_M` | 300, 0.05 | Adam |
# | `TRUE_AMP`, `TRUE_SIG` | see code | syn: $A_n / A_0$ and $\sigma_n / \sigma_0$ of the truth |
# | `SIG_PER_DYN` | False | one $\sigma_n$ per dynamic instead of shared |
# | `SHOW_KEY` | A#3 | key of the detail plot |

# %%
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from model import calc_tau, hammer2free, psi_lookup
from step2_lib import (DYNS, FS_SYN, NOISE_MARGIN, NOISE_SYN, PRE_SYN, T_START, TABLE, VELS,
                       fit, hann, load_wave, onset)

TAG        = os.environ.get("STEP6_TAG", "real")     # "syn": set STEP6_TAG=syn
N_PERIODS_M, K_M, T_MODES = 64, 8, 0.8
NOISE_BINS = 6
AMP_SCAN   = np.geomspace(1e-4, 0.3, 20)
SIG_INIT   = 2.0
STEPS, LR_M = 300, 0.05
TRUE_AMP   = np.array([0.05, 0.01, 0.005])     # one entry per RATIOS of step 5
TRUE_SIG   = np.array([10.0, 13.0, 16.0])      # median sigma_n / sigma_0 of the real fit
SIG_PER_DYN  = False    # True: one sigma_n per dynamic as well (check that decays do not depend on v)
SHOW_KEY     = "A#3"    # key of the detail plot (falls back to the middle key)


# %%
# ---------- fixed parameters of steps 2 and 3, lines of step 5 ----------
fit2   = np.load("step2_fit.npz", allow_pickle=True)
fit3   = np.load("step3_fit.npz", allow_pickle=True)
lines5 = np.load("step5_lines.npz", allow_pickle=True)
kappa, beta = float(fit2["kappa"]), float(fit3["beta"])
robust = lines5["robust"]                               # (73 keys, N ratios): line usable

# Keys: at least one robust line, and the K_M frames of N_PERIODS_M periods fit into T_MODES
fits_frames = N_PERIODS_M / lines5["f0"] < T_MODES
sel    = np.flatnonzero(robust.any(1) & fits_frames)
skip   = np.flatnonzero(robust.any(1) & ~fits_frames)
print("robust lines per ratio:", dict(zip(lines5["ratios"], robust.sum(0))))
print("skipped (frames longer than T_MODES):", list(lines5["notes"][skip]))

notes  = lines5["notes"][sel]
ok     = robust[sel]                                                    # (Mk, N)
# line frequencies: step 5's median for robust lines; the nominal ratio for the others
# (those keep c_n = 0 and are masked, the frequency only has to be finite)
f_lin  = np.where(ok, lines5["f_lines"][sel], lines5["ratios"] * lines5["f0"][sel, None])
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
    """|H_s| (n_dyn, N, K) of one key at the four velocities.  c_n and sig_n are either
    (N,), shared by all dynamics, or (n_dyn, N), one row per dynamic."""
    T, starts = frames(key["f0"])
    t = jnp.asarray(T_START + starts[:, None] + np.arange(int(round(T * fs)))[None, :] / fs)
    c_n   = jnp.broadcast_to(c_n, (len(VELS), N))             # shared -> one copy per dynamic
    sig_n = jnp.broadcast_to(sig_n, (len(VELS), N))

    @jax.checkpoint                                           # recompute in backward: less memory
    def one(vel, c_row, sig_row):
        c, sig, f = modes(key, c_row, sig_row)
        tau = calc_tau(vel, dict(tau_0=key["tau0"], beta=beta))
        return project(kappa * eps_modes(t, c, sig, f, vel, tau, key["p_o"], key["p_d"]), key["lines"], fs)

    return jax.vmap(one)(jnp.asarray(VELS), c_n, sig_n)


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
# ---------- loss, start, fit: one key at a time ----------
# The keys share no free parameter (kappa, beta, tau_0, ... are fixed), so each key is
# fit on its own: small jit graphs, progress per key, same optimum as one joint fit.
# theta of a key: log_c (n_dyn, N), one c_n per dynamic, so the hammer's velocity
# dependence (VELS, pulse shape) is absorbed instead of imposed; log_sig (N,), or
# (n_dyn, N) with SIG_PER_DYN.  Lines step 5 rejected keep c_n = 0 and are masked.

def key_errors(th, i):
    """Squared log residuals (n_dyn, N, K) of key i on the kept cells, and the mask."""
    Ht, eta, mask, fs = feats[i]
    Hs = model_H(keys[i], fs, ok[i] * jnp.exp(th["log_c"]), jnp.exp(th["log_sig"]))
    pred = jnp.logaddexp(jnp.log(Hs + 1e-30), jnp.log(eta))   # tiny offset: finite gradient
    return jnp.where(mask, (jnp.log(Ht + eta) - pred) ** 2, 0.0), mask


def key_loss(th, i):
    """Mean squared log residual of key i over all kept cells."""
    d, m = key_errors(th, i)
    return d.sum() / max(int(m.sum()), 1)


def key_start(i):
    """Per line the best A_n / A_0 of AMP_SCAN (same for all dynamics), converted to c_n;
    sigma_n = SIG_INIT * sigma_0."""
    key     = keys[i]
    log_sig = jnp.full(N, jnp.log(SIG_INIT * key["sig0"]))

    @jax.jit
    def line_loss(lc):                                         # (N,) mean residual per line
        d, m = key_errors(dict(log_c=lc, log_sig=log_sig), i)
        return d.sum((0, 2)) / np.maximum(m.sum((0, 2)), 1)

    cand = np.stack([np.log(np.asarray(c_for_amp(key, a))) for a in AMP_SCAN])   # (S, N)
    L    = np.stack([np.asarray(line_loss(jnp.asarray(lc))) for lc in cand])     # (S, N)
    best = cand[L.argmin(0), np.arange(N)]                                       # (N,)
    n_d  = len(DYNS)
    return dict(log_c=jnp.tile(jnp.asarray(best), (n_d, 1)),                     # (n_dyn, N)
                log_sig=jnp.tile(log_sig, (n_d, 1)) if SIG_PER_DYN else log_sig)


# %%
truth = None
if TAG == "syn":
    truth = dict(log_c=jnp.stack([jnp.log(c_for_amp(k, TRUE_AMP[:N])) for k in keys]),
                 log_sig=jnp.stack([jnp.log(TRUE_SIG[:N] * k["sig0"]) for k in keys]))
feats = features(TAG, truth)
kept  = np.stack([m.sum(2) for _, _, m, _ in feats])     # (Mk, n_dyn, N) frames in the loss
print("kept frames per line (all keys, dynamics):", kept.sum((0, 1)))

fitted = []                                                # per key: (theta, loss history)
for i, key in enumerate(keys):
    th, hist, _ = fit(key_start(i), lambda th: key_loss(th, i), STEPS, lr=LR_M,
                      label=f"{key['note']:>4} ({i + 1}/{Mk}): ")
    fitted.append((th, hist))
log_c   = np.stack([np.asarray(th["log_c"]) for th, _ in fitted])     # (Mk, n_dyn, N)
log_sig = np.stack([np.asarray(th["log_sig"]) for th, _ in fitted])   # (Mk, N) or (Mk, n_dyn, N)

# %%
# ---------- report ----------
# used[i, j, n]: line n of key i is robust and dynamic j has kept frames, so c_n is determined
used  = ok[:, None, :] & (kept > 0)
c_rel = np.where(used, np.exp(log_c), np.nan)
sig   = np.where(ok if log_sig.ndim == 2 else ok[:, None, :], np.exp(log_sig), np.nan)
jf    = DYNS.index("f")

# Hammer check: c_n of each dynamic relative to f, in dB.  If the hammer model (VELS and
# pulse shape) explained the velocity dependence, all of these would be 0 dB.
c_db = 20 * np.log10(c_rel / c_rel[:, jf:jf + 1, :])
print("\nc_n(dyn) / c_n(f) in dB, median over keys (n keys):")
for n, r in enumerate(lines5["ratios"]):
    cells = [f"{d}: {np.nanmedian(c_db[:, j, n]):+6.1f} ({np.isfinite(c_db[:, j, n]).sum():2d})"
             if np.isfinite(c_db[:, j, n]).any() else f"{d}:    -        "
             for j, d in enumerate(DYNS)]
    print(f"  r = {r:5.1f}:  " + "   ".join(cells))

rms = []
for i in range(Mk):
    d, m = key_errors(fitted[i][0], i)
    rms.append(np.sqrt(np.asarray(d).sum((1, 2)) / np.maximum(m.sum((1, 2)), 1)))
print("rms log residual per dynamic", DYNS, "(median over keys):", np.round(np.median(rms, 0), 3))

print("\nper key: sigma_n (1/s), and c_n(dyn)/c_n(f) in dB for p, mp, mf")
for i, key in enumerate(keys):
    s = sig[i] if sig.ndim == 2 else sig[i, jf]
    db = "  ".join(np.array2string(c_db[i, :jf, n], precision=1) for n in range(N) if ok[i, n])
    print(f"  {key['note']:>4}: sigma_n {np.round(s, 2)}   {db}")

if truth is not None:
    err_c = np.abs(np.exp(log_c - np.asarray(truth["log_c"])[:, None, :]) - 1)[used]
    s_fit = log_sig if log_sig.ndim == 2 else log_sig[:, jf]
    err_s = np.abs(np.exp(s_fit - np.asarray(truth["log_sig"])) - 1)[ok]
    print(f"syn, max rel err: c_n {err_c.max():.2%}, sigma_n {err_s.max():.2%}")

# %%
# ---------- plots ----------
os.makedirs("plots", exist_ok=True)
x = np.arange(Mk)

# 1) Hammer check (top): c_n(dyn)/c_n(f) over the keys, one panel per ratio.
#    Decays (bottom): sigma_n / sigma_0.
fig, ax = plt.subplots(2, N, figsize=(6 * N, 7), sharex=True, squeeze=False)
sig0 = np.array([k["sig0"] for k in keys])
for n, r in enumerate(lines5["ratios"]):
    for j, d in enumerate(DYNS[:jf]):
        ax[0, n].plot(x, c_db[:, j, n], "o", color=f"C{j}", ms=4, label=d)
    ax[0, n].axhline(0, color="k", lw=0.8)
    ax[0, n].set_title(f"r = {r}"); ax[0, n].set_ylabel("c_n(dyn) / c_n(f) (dB)")
    s = sig[:, n] if sig.ndim == 2 else sig[:, jf, n]
    ax[1, n].semilogy(x, s / sig0, "o", color="k", ms=4)
    ax[1, n].set_ylabel("sigma_n / sigma_0")
    ax[1, n].set_xticks(x); ax[1, n].set_xticklabels(notes, rotation=90, fontsize=7)
ax[0, 0].legend(title="dynamic", fontsize=8)
plt.tight_layout()
plt.savefig(f"plots/step6_fit_{TAG}_params.png", dpi=90); plt.show()

# 2) One key, all four dynamics: measured (solid) vs fitted (dashed) line magnitudes;
#    grey dots: below the noise margin, not in the loss.
i = list(notes).index(SHOW_KEY) if SHOW_KEY in notes else Mk // 2
Ht, eta, mask, fs = feats[i]
th = fitted[i][0]
Hs = np.asarray(model_H(keys[i], fs, ok[i] * jnp.exp(th["log_c"]), jnp.exp(th["log_sig"])))
_, starts = frames(keys[i]["f0"])
fig, ax = plt.subplots(1, len(DYNS) + 1, figsize=(4 * (len(DYNS) + 1), 3.5))
ax[0].semilogy(fitted[i][1]); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
for j, d in enumerate(DYNS):
    a = ax[j + 1]
    for n in range(N):
        db = 20 * np.log10(Ht[j, n] + eta[j, n])
        a.plot(starts, db, color=f"C{n}", label=f"{f_lin[i, n] / keys[i]['f0']:.2f} f_0")
        a.plot(starts, 20 * np.log10(Hs[j, n] + eta[j, n]), "--", color=f"C{n}")
        a.plot(starts[~mask[j, n]], db[~mask[j, n]], ".", color="0.6")
    a.set_title(f"{keys[i]['note']}-{d} ({TAG})"); a.set_xlabel("t after t_0 (s)")
ax[1].set_ylabel("dB"); ax[-1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"plots/step6_fit_{TAG}.png", dpi=90); plt.show()

# %%
# Save for step6_render.py.  c: (Mk, n_dyn, N), 0 for unused
# lines; sig: nan for lines that were not fit.
if TAG == "real":
    np.savez("step6_fit.npz", notes=notes, dyns=np.array(DYNS), ratios=lines5["ratios"],
             f_modes=f_lin, ok=ok, used=used, kept=kept,
             c=np.where(used, np.exp(log_c), 0.0), sig=sig)
    print("-> step6_fit.npz")
