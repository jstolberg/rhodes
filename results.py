# %% [markdown]
# # Results for the paper
#
# Loads the results of the four estimation steps, computes the numbers quoted in the paper,
# draws the figures and prints the numbers as short English sentences.  Nothing is fitted.
#
# | Step | Script | Result file |
# | --- | --- | --- |
# | 1 fundamental | `fundamental_measurement.py` | `fundamentals.npz` |
# | 2 pickup and free oscillation | `step2_init.py`, `step2_fft.py` | `step2_fit.npz` |
# | 3 hammer, incl. $c_0$ | `hammer.py` | `step3_fit.npz` |
# | 4 inharmonic modes | `step_5.py` (decays), `step6_modes.py` (excitation) | `step6_fit.npz` |
#
# Also needed: `cache/features_real_1.npz` (step 2 targets), `cache/step4_synthetic_check.npz`
# (`step4_synthetic_check.py`) and the recordings in `Samples/`.
#
# Output: figures as PDF in `figures/`, all sentences and the mode table in `results_numbers.md`.
# Run time about 10 minutes (step 5 runs on import, the full model is rendered for every
# recording).

# %%
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import blackmanharris

from model import calc_tau, epsilon, hammer2free
from step2_lib import (DYNS, NOISE_MARGIN, PD_RANGE, T_START, TABLE, VELS, frame_times, hann,
                       load_wave, onset, project_synth, project_target)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.0,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight"})
W1, W2 = 3.5, 7.16                      # IEEE column widths (in)
DB = 20 / np.log(10)                    # natural log -> dB
FS = 48000.0
JF = DYNS.index("f")
os.makedirs("figures", exist_ok=True)
text = {}                               # sentences per section, written to results_numbers.md

fund = np.load("fundamentals.npz", allow_pickle=True)
fit2 = np.load("step2_fit.npz", allow_pickle=True)
fit3 = np.load("step3_fit.npz", allow_pickle=True)
fit6 = np.load("step6_fit.npz", allow_pickle=True)
notes = [str(n) for n in fit2["notes"]]
M = len(notes)
for d in (fund, fit3, fit6):                         # same keys in the same order everywhere
    assert [str(n) for n in d["notes"]] == notes
assert list(fund["dyns"]) == list(DYNS) == list(fit6["dyns"])
assert {"c_dyn", "measured"} <= set(fit6.files), "run the new step6_modes.py first"
print("velocities VELS (step2_lib.py, used by hammer.py):", dict(zip(DYNS, VELS)))

f0 = fit2["f0"]
kappa, beta = float(fit2["kappa"]), float(fit3["beta"])
c0 = fit3["c0"]


def say(section, s):
    # print a sentence and keep it for results_numbers.md
    print(s)
    text.setdefault(section, []).append(s)


# %% [markdown]
# ## Step 1: fundamental

# %%
tab = fund["table"]
ok1 = tab["ok"]
lam = np.where(ok1, tab["lam"], np.nan)                       # decay per recording (1/s)

# tuning: equal temperament, A = 440 Hz; the library names are an octave low
# (as f0_of in fundamental_measurement.py)
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
midi = np.array([(int(n[-1]) + 2) * 12 + NAMES.index(n[:-1]) for n in notes])
nominal = 440.0 * 2.0 ** ((midi - 69) / 12)
cents = 1200 * np.log2(np.median(tab["f"], axis=1) / nominal)  # per key, median over the dynamics
c5, c95 = np.percentile(cents, [5, 95])

# sigma_0 ~ f0^b, log-log fit over the valid recordings
b, a = np.polyfit(np.log(tab["f"][ok1]), np.log(lam[ok1]), 1)
sig_key = np.array([np.nanmedian(r) if np.isfinite(r).any() else np.nan for r in lam])
T60 = 3 * np.log(10) / sig_key                                # 6.91 / sigma_0
t60_low = np.nanmedian(T60[f0 < 100])
t60_high = np.nanmedian(T60[f0 > 800])
both = ok1[:, 0] & ok1[:, JF]
fp_ratio = np.median(lam[both, JF] / lam[both, 0])

S = "Step 1"
say(S, f"The measured f0 deviates from equal temperament (A = 440 Hz) by {np.median(cents):+.1f} cents "
       f"in the median (5th to 95th percentile {c5:+.1f} to {c95:+.1f} cents).")
say(S, f"The decay of the fundamental could be measured in {ok1.sum()} of {ok1.size} recordings.")
say(S, f"The decay rate grows with frequency as sigma_0 ~ f0^{b:.2f}.")
say(S, f"The time to decay by 60 dB is {t60_low:.0f} s below 100 Hz and {t60_high:.1f} s above 800 Hz (median).")
say(S, f"At f the fundamental decays {100 * (fp_ratio - 1):+.0f} % faster than at p "
       f"(median of sigma_0(f)/sigma_0(p) = {fp_ratio:.2f} over {both.sum()} keys).")
print("\nPaper so far: +1.9 cents, 245/292, b = 0.8, T60 = 40 s / 4 s, 11 %.")
print(f"Here:          {np.median(cents):+.1f} cents, {ok1.sum()}/{ok1.size}, b = {b:.2f}, "
      f"T60 = {t60_low:.0f} s / {t60_high:.1f} s, {100 * (fp_ratio - 1):+.0f} %.")

# %% [markdown]
# ## Step 2: pickup and free oscillation
#
# The fit residual is not stored in `step2_fit.npz`; it is recomputed here as in the step-2
# loss: target and model projected onto $f_0 \dots 5 f_0$ in the step-2 frames, masked
# mean squared log difference (cells above the noise).

# %%
z = np.load("cache/features_real_1.npz")
H, eta, mask = z["H"], z["eta"], z["mask"]                    # (keys, dyn, harmonics, frames)


def step2_model(i, j):
    # |H_s| of the step-2 model (free fundamental through the pickup), with kappa
    return kappa * np.asarray(project_synth(jnp.asarray(float(fit2["A0"][i, j])), jnp.asarray(float(fit2["sigma"][i])),
                                            jnp.asarray(float(f0[i])), float(fit2["tfit"][i]),
                                            float(fit2["p_o"][i]), float(fit2["p_d"][i]), TABLE))


rms2 = np.full((M, len(DYNS)), np.nan)
for i in range(M):
    for j in range(len(DYNS)):
        d = np.log(H[i, j] + eta[i, j]) - np.log(step2_model(i, j) + eta[i, j])
        if mask[i, j].any():
            rms2[i, j] = np.sqrt(np.mean(d[mask[i, j]] ** 2))
rms2_key = np.nanmedian(rms2, axis=1)

pd, po = 1e3 * fit2["p_d"], 1e3 * fit2["p_o"]
n_bound = int(np.sum(fit2["p_d"] < PD_RANGE[0] + 0.01e-3))

S = "Step 2"
say(S, f"The rms log residual of the fit is {np.median(rms2_key):.2f} in the median over the keys "
       f"({DB * np.median(rms2_key):.1f} dB).")
say(S, f"The pickup distance p_d is {np.median(pd):.2f} mm in the median ({pd.min():.2f} to {pd.max():.2f} mm), "
       f"the offset p_o {np.median(po):.2f} mm ({po.min():.2f} to {po.max():.2f} mm).")
say(S, f"p_d reaches the lower bound of {1e3 * PD_RANGE[0]:.1f} mm on {n_bound} key(s).")
say(S, f"The global gain is kappa = {kappa:.3g}.")

# %% [markdown]
# ## Step 3: hammer

# %%
p_contact = (1 + beta) / (1 - beta)
N_BASS = 12
A0, ok2 = fit2["A0"], fit2["ok"]
rat_bass = np.where(ok2[:N_BASS] & ok2[:N_BASS, JF:JF + 1], A0[:N_BASS] / A0[:N_BASS, JF:JF + 1], np.nan)
v_bass = np.nanmean(rat_bass, axis=0)

S = "Step 3"
say(S, f"The fitted exponent is beta = {beta:.3f}, equivalent to a contact exponent p = (1+beta)/(1-beta) = "
       f"{p_contact:.2f} (Falaize et al.: p = 2.5, beta = 0.43).")
say(S, f"The contact time at full velocity falls from {1e3 * fit3['tau0'][0]:.2f} ms ({notes[0]}) "
       f"to {1e3 * fit3['tau0'][-1]:.2f} ms ({notes[-1]}).")
say(S, f"On the {N_BASS} lowest keys, A_0 relative to f is {v_bass[0]:.2f}, {v_bass[1]:.2f} and {v_bass[2]:.2f} "
       f"for p, mp and mf, against the assumed velocities {VELS[0]}, {VELS[1]} and {VELS[2]}.")

# %% [markdown]
# ## Step 4: inharmonic modes
#
# `step_5.py` runs on import (about 1 minute).

# %%
import step_5 as s5                     # noqa: E402  (runs step 5 once)

names = list(s5.MODES)
mu = np.array([s5.MODES[m][0] for m in names])
ok6, c6, c_dyn, measured = fit6["ok"], fit6["c"], fit6["c_dyn"], fit6["measured"]
rows = s5.rows
clear = np.zeros(ok6.shape, bool)
for x in rows:
    clear[notes.index(x["note"]), names.index(x["mode"])] = True

# spread over the dynamics: per key std of log(c_dyn / c0) over the dynamics with c_dyn > 0
spread = np.full(ok6.shape, np.nan)
for i in range(M):
    for n in range(len(names)):
        u = c_dyn[i, :, n] > 0
        if u.sum() >= 2:
            spread[i, n] = DB * np.std(np.log(c_dyn[i, u, n] / c0[i]))

tab4 = []
for n, m in enumerate(names):
    r = np.array([x["ratio"] for x in rows if x["mode"] == m])
    k = clear[:, n] & measured
    cc = c6[ok6[:, n], 0, n] / c0[ok6[:, n]] if ok6[:, n].any() else np.array([np.nan])
    tab4.append(dict(mode=m, mu=mu[n], clear=int(clear[:, n].sum()), meas=int(k.sum()),
                     low=int((clear[:, n] & ~measured).sum()),
                     ratio=np.median(r) if r.size else np.nan,
                     cdb=DB * np.log(cc[0]) if ok6[:, n].any() else np.nan,
                     spread=np.nanmedian(spread[k, n]) if np.isfinite(spread[k, n]).any() else np.nan))

fmt = lambda v, f: "--" if not np.isfinite(v) else format(v, f)
print(f"{'mode':<5}{'mu_n':>6}{'clear':>7}{'measured':>10}{'sigma_n/sigma_0':>17}{'c_n/c_0 (dB)':>14}{'spread (dB)':>13}")
for t in tab4:
    print(f"{t['mode']:<5}{t['mu']:>6}{t['clear']:>7}{t['meas']:>10}{fmt(t['ratio'], '.1f'):>17}"
          f"{fmt(t['cdb'], '+.1f'):>14}{fmt(t['spread'], '.1f'):>13}")
latex = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
         "Mode & $\\mu_n$ & clear & measured & $\\sigma_n/\\sigma_0$ & $c_n/c_0$ (dB) & spread (dB) \\\\",
         "\\midrule"]
latex += [f"{t['mode']} & {t['mu']} & {t['clear']} & {t['meas']} & {fmt(t['ratio'], '.1f')} & "
          f"{fmt(t['cdb'], '+.1f')} & {fmt(t['spread'], '.1f')} \\\\" for t in tab4]
latex += ["\\bottomrule", "\\end{tabular}"]
print("\n" + "\n".join(latex))

S = "Step 4"
n_keys = len({x["note"] for x in rows})
say(S, f"A mode was found with confidence in {len(rows)} cases on {n_keys} keys.")
for t in tab4:
    if t["clear"]:
        say(S, f"{t['mode']} ({t['mu']} f0): clear on {t['clear']} keys, {t['low']} of them below 150 Hz and "
               f"therefore not used for c_n; decays {t['ratio']:.1f} times faster than the fundamental.")
say(S, "No mode above m4 was found with confidence, so m5 and m6 are not part of the model.")

# synthetic check (step4_synthetic_check.py), results saved there
zs = np.load("cache/step4_synthetic_check.npz")
err_c, bass = zs["err_c"], zs["bass"]
say(S, f"On synthetic recordings with known excitation, c_n is recovered within {np.nanmedian(np.abs(err_c[~bass])):.1f} dB "
       f"(median) from 150 Hz up, but misses by {np.nanmedian(np.abs(err_c[bass])):.0f} dB below 150 Hz.")
print("(synthetic check: saved result of step4_synthetic_check.py, run with the previous step-6 version; "
      "the measurement it tests is unchanged)")


def params(i, j):
    # all modes of key i at dynamic j, as step6_render.py
    k = ok6[i]
    return dict(c=jnp.r_[c0[i], c6[i, j][k]], lam=jnp.r_[fit2["sigma"][i], fit6["sig"][i][k]],
                f_modes=jnp.r_[f0[i], fit6["f_modes"][i][k]], tau_0=float(fit3["tau0"][i]), beta=beta,
                p_d=float(fit2["p_d"][i]), p_o=float(fit2["p_o"][i]))


def line_db(x, fs, f):
    # level (dB) of the line at f: Blackman-Harris window, normalised per sample
    w = blackmanharris(len(x))
    return DB * np.log(np.abs((x * w) @ np.exp(-2j * np.pi * f * np.arange(len(x)) / fs)) / w.sum() + 1e-15)


# level of each mode relative to the fundamental in the model, dynamic f, first 0.3 s after contact
t03 = jnp.asarray(T_START + np.arange(int(0.3 * FS)) / FS)
lev = np.full((M, len(names)), np.nan)
for i in range(M):
    if not ok6[i].any():
        continue
    x = kappa * np.asarray(epsilon(t03, params(i, JF), TABLE, vel=VELS[JF]))
    for n in np.flatnonzero(ok6[i]):
        lev[i, n] = line_db(x, FS, fit6["f_modes"][i, n]) - line_db(x, FS, f0[i])
for n, m in enumerate(names[:3]):
    say(S, f"In the model at f, {m} lies {-np.nanmedian(lev[:, n]):.0f} dB below the fundamental "
           f"(median over {np.isfinite(lev[:, n]).sum()} keys; Gabrielli et al.: 42-55 dB).")

# decay ratios against Gabrielli et al., Table II (laser vibrometer)
gab = [("F1", "m2", 7.2, 2.6), ("F1", "m3", 20.6, 8.3), ("F3", "m2", 7.4, 24.5), ("F3", "m3", 20.7, 3.1), ("F3", "m4", 38.7, 13.4)]
print("\ndecay ratio sigma_n/sigma_0, Gabrielli Table II against here (clear keys only):")
for key, m, ratio_f, g in gab:
    hit = [x for x in rows if x["note"] == key and x["mode"] == m]
    here = f"{hit[0]['ratio']:.1f} (at {hit[0]['ratio_f']:.2f} f0)" if hit else "not clear"
    print(f"  {key} {m} ({ratio_f} f0): Gabrielli {g:5.1f}   here {here}")

# %% [markdown]
# ## Full model against the recordings
#
# Every recording is rendered as in `step6_render.py` and projected onto lines in the
# step-2 frames (`project_target` of `step2_lib.py`): $f_0$, the harmonics $2 f_0 \dots 5 f_0$
# and the modes.  The recording's mode is projected at the peak step 5 finds, the model's at
# $f_n$; modes only on keys from 150 Hz up and only in the first frame (they die out within
# it), against the noise 6 bins beside the line.  Only cells above the noise count.

# %%
LEAD = 10e-3
DUR = float(fit2["tfit"].max()) + 0.3                      # one length for all keys: one jit compile
t_full = jnp.arange(int(DUR * FS)) / FS


def render(i, j, t):
    x = kappa * np.asarray(epsilon(t, params(i, j), TABLE, vel=VELS[j]))
    return np.concatenate([np.zeros(int(LEAD * FS)), x])


def project_lines(x, fs, f0_, tfit, freqs):
    # |H| (len(freqs), frames) in the step-2 frames, as project_target
    T, starts = frame_times(f0_, tfit)
    L = int(round(T * fs))
    w, tt = hann(L), np.arange(L) / fs
    i0 = onset(x) + int(round(T_START * fs))
    E = np.exp(-2j * np.pi * np.outer(freqs, tt))
    return np.stack([np.abs(E @ (w * x[i0 + int(round(s * fs)):][:L])) / fs for s in starts], axis=1)


dev_h = np.full((M, len(DYNS), 5), np.nan)                 # model - recording (dB): f0, 2f0 ... 5f0
dev_m = np.full((M, len(DYNS), len(names)), np.nan)
for i in range(M):
    tfit = float(fit2["tfit"][i]); i5 = s5.NOTES.index(notes[i])
    for j, dyn in enumerate(DYNS):
        xr, fs = load_wave(notes[i], dyn)
        xm = render(i, j, t_full)
        Hr, er = project_target(xr, fs, f0[i], tfit)
        Hm, _ = project_target(xm, FS, f0[i], tfit)
        for n in range(5):
            m_ = Hr[n] > NOISE_MARGIN * er[n]
            if m_.any():
                dev_h[i, j, n] = np.median(DB * np.log(Hm[n][m_] / Hr[n][m_]))
        if measured[i]:
            fr, dbr = s5.spectrum(*s5.load(notes[i], dyn))
            for n in np.flatnonzero(ok6[i]):
                fp = s5.find_peak(fr, dbr, float(s5.FUND[i5, j]["f"]), *s5.MODES[names[n]])
                if not np.isfinite(fp):
                    continue
                # first frame only: later the mode has died out and the recording's line holds
                # only leakage of the neighbouring harmonics.  Noise: 6 bins beside the line
                # (the floor between the harmonics is raised by the attack in the first frame)
                T_ = frame_times(f0[i], tfit)[0]
                hr, lo, hi = project_lines(xr, fs, f0[i], tfit, [fp, fp - 6 / T_, fp + 6 / T_])[:, 0]
                hm = project_lines(xm, FS, f0[i], tfit, [fit6["f_modes"][i, n]])[0, 0]
                if hr > NOISE_MARGIN * min(lo, hi):
                    dev_m[i, j, n] = DB * np.log(hm / hr)
    print(f"{notes[i]:>4}", end=" ", flush=True)
print()

S = "Full model"
q = lambda v: (np.nanmedian(v), np.nanpercentile(v, 75) - np.nanpercentile(v, 25))
print(f"{'':>12}" + "".join(f"{d:>16}" for d in DYNS))
for name, v in (("fundamental", dev_h[:, :, :1]), ("harmonics", dev_h[:, :, 1:]), ("modes", dev_m)):
    print(f"{name:>12}" + "".join(f"{q(v[:, j])[0]:>+8.1f} ({q(v[:, j])[1]:4.1f})" for j in range(len(DYNS))))
for name, v in (("fundamental", dev_h[:, :, :1]), ("harmonics 2f0-5f0", dev_h[:, :, 1:]), ("modes", dev_m)):
    med, iqr = q(v)
    say(S, f"Model minus recording, {name}: {med:+.1f} dB in the median, interquartile range {iqr:.1f} dB.")

# %% [markdown]
# ## Figures

# %%
# Figure A: decay rates over f0
fig, ax = plt.subplots(figsize=(W1, 2.6))
ax.loglog(np.median(tab["f"], axis=1), sig_key, "o", ms=2.5, color="0.6", label=r"$\sigma_0$")
fg = np.geomspace(f0.min(), f0.max() * 45, 50)
ax.loglog(fg[fg < 3000], np.exp(a) * fg[fg < 3000] ** b, "-", color="0.4", label=fr"$\propto f_0^{{{b:.2f}}}$")
for n, m in enumerate(names):
    hit = [x for x in rows if x["mode"] == m]
    if not hit:
        continue
    fx = np.array([x["f0"] for x in hit]); sg = np.array([x["sig_n"] for x in hit])
    low = fx < 150
    ax.loglog(fx[~low], sg[~low], "o", ms=3.5, color=f"C{n}", label=m)
    ax.loglog(fx[low], sg[low], "o", ms=3.5, mfc="none", color=f"C{n}")
    r_n = np.median([x["ratio"] for x in hit])
    ax.loglog(fg[fg < 3000], r_n * np.exp(a) * fg[fg < 3000] ** b, "--", color=f"C{n}", lw=0.8)
ax.set_xlabel("$f_0$ (Hz)"); ax.set_ylabel(r"decay rate $\sigma$ (1/s)")
ax.legend(ncol=2, loc="upper left")
plt.savefig("figures/fig_decay_rates.pdf"); plt.show()

# Figure B: example key, step 2, harmonics measured and modelled over time
EX = "A#3" if "A#3" in notes else notes[M // 2]
i, j = notes.index(EX), DYNS.index("mf")
xr, fs = load_wave(EX, "mf")
Hr, er = project_target(xr, fs, f0[i], float(fit2["tfit"][i]))
Hs = step2_model(i, j)
T, starts = frame_times(f0[i], float(fit2["tfit"][i]))
fig, ax = plt.subplots(figsize=(W1, 2.4))
for n in range(5):
    lab = "$f_0$" if n == 0 else f"${n + 1}f_0$"
    ax.plot(starts + T / 2, DB * np.log(Hr[n] + er[n]), "-", color=f"C{n}", label=lab)
    ax.plot(starts + T / 2, DB * np.log(Hs[n] + er[n]), "--", color=f"C{n}")
ax.plot(starts + T / 2, DB * np.log(er[0]), ":", color="0.5", label="noise")
ax.set_xlabel("time after $t_0$ (s)"); ax.set_ylabel("level (dB)")
ax.set_ylim(top=ax.get_ylim()[1] + 12)                     # room for the label
ax.text(0.98, 0.95, f"{EX}, mf\nsolid: measured, dashed: model", transform=ax.transAxes,
        ha="right", va="top", fontsize=7)
ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=0.8, handlelength=1.5)
plt.savefig("figures/fig_step2_example.pdf"); plt.show()

# Figure C: A_0(dyn) / A_0(f) over the keys, measurement and hammer model
pred = np.zeros((M, len(DYNS)))
for i in range(M):
    a_ = []
    for v in VELS:
        tau = float(calc_tau(v, dict(tau_0=fit3["tau0"][i], beta=beta)))
        a_.append(abs(float(hammer2free(float(f0[i]), 1.0, v, tau)[0])) * np.exp(-float(fit2["sigma"][i]) * (T_START - tau)))
    pred[i] = np.array(a_) / a_[-1]
meas = np.where(ok2 & ok2[:, JF:JF + 1], A0 / A0[:, JF:JF + 1], np.nan)
fig, ax = plt.subplots(figsize=(W1, 2.4))
for j in range(3):
    ax.semilogx(f0, DB * np.log(meas[:, j]), "o", ms=2.5, color=f"C{j}", label=DYNS[j])
    ax.semilogx(f0, DB * np.log(pred[:, j]), "-", color=f"C{j}")
    ax.axhline(DB * np.log(VELS[j]), color=f"C{j}", ls="--", lw=0.8)
ax.set_xlabel("$f_0$ (Hz)"); ax.set_ylabel(r"$A_0(\mathrm{dyn})/A_0(f)$ (dB)")
ax.legend(title="dots: measured, lines: model", ncol=3)
plt.savefig("figures/fig_hammer_amplitudes.pdf"); plt.show()

# Figure D: spectrograms, recording and model, p and f
from scipy.signal import spectrogram      # noqa: E402

i = notes.index(EX)
t15 = jnp.arange(int(1.5 * FS)) / FS
panels = []
for dyn in ("p", "f"):
    xr, fs = load_wave(EX, dyn)
    a0 = max(onset(xr) - int(LEAD * fs), 0)
    panels.append((f"{EX}-{dyn}, recording", xr[a0:a0 + int(1.5 * fs)], fs))
    panels.append((f"{EX}-{dyn}, model", render(i, DYNS.index(dyn), t15)[:int(1.5 * FS)], FS))
specs = []
for title, x, fs in panels:
    fq, tt, Sx = spectrogram(x, fs, window="hann", nperseg=2048, noverlap=1792)
    specs.append((title, fq, tt, 10 * np.log10(Sx + 1e-30)))
vmax = max(s[3].max() for s in specs)
fig, axs = plt.subplots(2, 2, figsize=(W2, 4.0), sharex=True, sharey=True)
for ax, (title, fq, tt, Sx) in zip(axs.ravel(), specs):
    im = ax.pcolormesh(tt, fq / 1e3, Sx, vmin=vmax - 90, vmax=vmax, shading="auto", cmap="magma", rasterized=True)
    ax.set_ylim(0, 10); ax.set_title(title); ax.grid(False)
for ax in axs[:, 0]:
    ax.set_ylabel("frequency (kHz)")
for ax in axs[1]:
    ax.set_xlabel("time (s)")
fig.colorbar(im, ax=axs, label="dB", shrink=0.8)
plt.savefig("figures/fig_spectrograms.pdf"); plt.show()

# %%
# All sentences and the mode table
with open("results_numbers.md", "w", encoding="utf-8") as f:
    f.write("# Numbers for the results section\n\nGenerated by `results.py`.\n")
    for section, lines in text.items():
        f.write(f"\n## {section}\n\n" + "\n".join(f"- {s}" for s in lines) + "\n")
    f.write("\n## Table: modes\n\n```latex\n" + "\n".join(latex) + "\n```\n")
print("-> results_numbers.md, figures/fig_*.pdf")
