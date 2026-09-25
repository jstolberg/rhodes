# %% [markdown]
# # Fitting the model -- current state
#
# `fitting.py` is the plan.  This file records what is actually implemented on
# `paula/dev` (as of 2026-09-24): which script does which step, what is fitted,
# measured or fixed, which assumptions are made, the results so far, and where
# the implementation departs from the plan.  Nothing here is run; it is
# documentation only.
#
# ## Data
#
# | Item | Value |
# | --- | --- |
# | Keys | 73, E0 ... E6 (the library labels its files one octave low) |
# | Dynamics | p, mp, mf, f -- four recordings per key, 292 in total |
# | Format | `Samples/{note}-{dyn}.wav`, 48 kHz, ~8 s, library fade-out from ~7.3 s |
# | Velocities | **not measured**; assumed `VELS` = 0.25, 0.5, 0.75, 1 for p, mp, mf, f |
#
# ## Pipeline
#
# ```
# Samples/*.wav
#   -> step 1    fundamental_measurement.py        -> fundamentals.npz
#   -> step 2    step2_init.py, step2_fft.py       -> step2_fit.npz
#   -> step 3+4  hammer.py                          -> step3_fit.npz
#   -> step 5    step5_lines.py                    -> step5_lines.npz
#   -> step 6    step6_fft.py                      -> step6_fit.npz
#   -> render    step{2,3,6}_render.py             -> render/*.wav
# ```
#
# Every step reads the results of the steps before it and holds them fixed.
#
# | File | Role |
# | --- | --- |
# | `model.py` | the forward model: pickup surface, $\Psi$ table, hammer, tine motion, pickup voltage |
# | `fitting.py` | the plan (parameters, steps, signal areas) |
# | `fundamental_measurement.py` | step 1 |
# | `step2_lib.py` | step 2: shared definitions and hyperparameters (also used by steps 3, 6) |
# | `step2_init.py`, `step2_fft.py` | step 2: initial values, fit |
# | `hammer.py` | steps 3 and 4 |
# | `step3_vels.py` | check of the assumed `VELS` (not part of the chain) |
# | `step5_lines.py` | step 5: all 292 recordings, robust lines per key |
# | `step6_fft.py` | step 6: all keys with robust lines, $c_n$ per dynamic |
# | `step7_hammer_check.py` | how the modes grow with velocity: hammer variants and an MLP, leave-one-key-out (not part of the chain) |
# | `step2_render.py`, `step3_render.py`, `step6_render.py` | wavs of the model after each step |
# | `plot_params.py` | overview plots of the step 2 fit |
# | `step2_single_note.py` | earlier single-note example of step 2 (also fits $a, m$); not in the chain |
# | `mode_measurement.py` | demodulation measurement of the modes of one note; not in the chain |
# | `mrstft_loss.py` | multi-resolution STFT loss; proposed in the plan, not used by any step |

# %% [markdown]
# ## Model and its assumptions (`model.py`)
#
# **Tine.**  A sum of independent, linearly decaying modes: the fundamental
# $f_0$ and the inharmonic modes $f_n$, each with excitation coefficient $c_n$ and
# decay $\sigma_n$.  The tip moves only along $x$ ($y = 0$, $z = p_d$ fixed).
# Decays are a property of the tine, independent of the strike.
#
# **Hammer.**  A prescribed force, no feedback from the tine and no damping
# during contact:
# $$F(t) = F_0 \sin^2(\pi t / \tau), \quad 0 \le t \le \tau, \qquad
#   F_0 = 2v/\tau \;\Rightarrow\; \int F\,dt = v, \qquad \tau = \tau_0\, v^{-\beta}.$$
# The pulse is a raised cosine (Hann), **not** a half-sine as `fitting.py`
# says.  Its spectrum has its main lobe up to
# $f\tau = 2$ and nulls at $f\tau = 2, 3, \ldots$; beyond the main lobe the
# free amplitude falls like $1/(f\tau)^3$, so a mode far above $1/\tau$ grows with
# velocity like $v^{1 + 3\beta}$, while a mode with $f\tau \ll 1$ (bass
# fundamental) grows like $v$.  `hammer2free` hands the displacement and velocity
# at $t = \tau$ over to the free oscillation (checked by hand: exact).
#
# **Pickup.**  A static nonlinearity: $\varepsilon = -\kappa\, d\Psi/dt$ with the
# flux $\Psi(x, p_d)$ of a surface of fixed shape.  $\Psi$ is tabulated once and
# interpolated; $p_o$ and $p_d$ enter only through the lookup.  The coil's RLC
# filter is ignored.  $\kappa$ is one global gain (recording level and pickup
# constant are both arbitrary).
#
# ## Parameters: plan vs. now
#
# | Group | Symbol | Scope (plan) | Now | Where |
# | --- | :---: | --- | --- | --- |
# | Tine | $f_0$ | per key | **measured** | step 1 |
# | - | $\sigma_0$ | per key | fitted | step 2 |
# | - | $c_0$ | per key | fitted (falls out of step 3) | step 4 |
# | - | $f_n$ | per mode | **searched** near fixed ratios, 3 modes | step 5 |
# | - | $\sigma_n$ | per mode | fitted | step 6 |
# | - | $c_n$ | per mode | fitted per mode **and dynamic** | step 6 |
# | Pickup | $m$, $a$, $r$ | global, fitted | **fixed** from photos: 1, 1.3 mm, 4 mm | `step2_lib.py` |
# | - | $p_d$, $p_o$ | per key | fitted | step 2 |
# | - | $\kappa$ | (not in plan) | closed form, global | step 2 |
# | Hammer | $\tau_0$ | log-linear over keys (2 numbers) | fitted | step 3 |
# | - | $\beta$ | global | fitted | step 3 |
# | - | $v$ | (not in plan) | **assumed** `VELS` | `step2_lib.py`, `hammer.py` |

# %% [markdown]
# ## Signal areas
#
# As in `fitting.py`, with H0 = $f_0$, H1 = $2 f_0$, ...  Note that the 7.1 mode
# lies **between H6 and H7**, not above H7 as the plan assumes, and the pickup
# mixes it down to $6.1 f_0$, $5.1 f_0$, ... inside the H0-H6 band.
#
# | | Contact | Inharmonic decay | Harmonic decay | Fundamental decay |
# | :---: | :---: | :---: | :---: | :---: |
# | $>H6$ | S1 | S2 (steps 5, 6) | - | - |
# | $H0-H6$ | S3 | S4 (step 2) | S5 (step 2) | - |
# | $\leq H0$ | S6 | S7 (not used) | S8 | S9 |
# | Approx. duration: | 1-2 ms | $<1$ s | $<4$ s | $>4$ s |
#
# ## Common method of the fits
#
# Steps 2 and 6 do not use an STFT.  Target and model are **projected onto a few
# lines** (the harmonics in step 2, the mode frequencies in step 6) in Hann
# frames whose length is a fixed number of periods of $f_0$.  The loss is the
# masked mean squared difference of the log magnitudes, with a noise floor
# $\eta$ measured beside each line:
# $$L = \frac{1}{\sum m}\sum m\,\big(\log(|H_t| + \eta) - \log(|H_s| + \eta)\big)^2,$$
# and a cell counts only if $|H_t| > 3\eta$ (`NOISE_MARGIN`, 9.5 dB).  The optimiser
# is Adam on log-parameters; start values come from a multi-start or a scan
# because the loss has several basins.

# %% [markdown]
# ## 1. $f_0$ -- measured
#
# `fundamental_measurement.py` -> `fundamentals.npz` (one row per recording)
#
# - $f_0$: tallest bin within 3 % of the nominal pitch in one large FFT of the
#   opening, refined by the phase slope after demodulation to 0 Hz.
# - Decay $\lambda$: straight-line fit to the log envelope, down to 40 dB below its
#   peak; left empty (`ok` False) if the line fell less than 8 dB (lowest keys).
# - Attack (first 50 ms) and fade-out (last 0.7 s) are cut off.
# - Nothing is averaged or filled in.  Step 2 (`load_keys`) takes the median over
#   the dynamics and fills missing decays by log interpolation over the keyboard.
#
# Plan: refers to `mode_measurement.py` on `niklas/dev`; the chain uses
# `fundamental_measurement.py` instead.

# %% [markdown]
# ## 2. Pickup and fundamental -- fitted
#
# `step2_init.py` (start values, cached) + `step2_fft.py` (fit) -> `step2_fit.npz`
#
# - Model: one free mode $x = p_o + A_0 e^{-\sigma_0 t}\sin 2\pi f_0 t$ through the
#   pickup.  No hammer, no inharmonic modes.
# - Target: projection onto $f_0$ and 4 harmonics (H0-H4), 12 frames of 64
#   periods over $\min(3\text{ s}, 4/\sigma_1)$ after $t_0$ = onset + 10 ms.  64
#   periods keep the mixing products of the 7.1 mode more than 6 bins away from a
#   harmonic.
# - Fitted: $p_o$, $p_d$, $\sigma_0$ per key; $A_0$ per key and dynamic
#   (intermediate, the free amplitude at $t_0$); $\kappa$ in closed form.
# - Fixed: surface shape ($r$, $a$, $m$).  Bounds: $p_d$ in 0.5-3.5 mm, $p_o$ in
#   0-6 mm (sigmoid).
# - Start: multi-start in $p_o$ (20) x $A_0$ (3), best third kept per round.
# - Stages: A geometry on the f recordings, B $A_0$ of p, mp, mf, C everything.
#
# Result: all 292 cells ok.  $p_d$ 0.5-3.24 mm (median 1.38; the lowest value is
# the lower bound), $p_o$ 0.64-5.46 mm (median 1.75).
#
# Plan: MR-STFT over S4/S5 with $m$, $a$ free.  Now: line projection, geometry
# fixed, only H0-H4 (the H0-H6 band is not free of inharmonic content, see above).

# %% [markdown]
# ## 3. and 4. Hammer ($\tau_0$, $\beta$) and $c_0$ -- fitted
#
# `hammer.py` -> `step3_fit.npz`
#
# - Data: $A_0(k, v)$ of step 2, carried back from $t_0$ to the hand-over at
#   $\tau$ with $\sigma_0$.
# - Model: $A_0 = c_0\, |\text{hammer2free}(f_0, 1, v, \tau(v))|$; $c_0$ cancels by
#   subtracting the per-key mean of the log residual, which is then $\log c_0$.
# - Fitted: $\tau_0$ log-linear over the keys (2 numbers), $\beta$ global; $c_0$
#   per key as a by-product (step 4).
# - Assumed: `VELS` = 0.25, 0.5, 0.75, 1.
# - Synthetic round trip in the same file.
#
# Result: $\beta$ = 0.252, $\tau_0$ = 3.89 ms (E0) -> 0.43 ms (E6).  rms log
# residual per dynamic p, mp, mf, f: 0.31, 0.15, 0.17, 0.22.
#
# ### Check of `VELS` (`step3_vels.py` -> `step3_fit_vels.npz`)
#
# In the bass ($f_0 \tau \ll 1$) $A_0 \propto v$, so the ratios
# $A_0(\text{dyn}) / A_0(f)$ over the 12 lowest keys are the velocity ratios:
# **0.46, 0.66, 0.83** (std over keys ~0.2), not 0.25, 0.5, 0.75.  Refitting step 3:
#
# | Velocities | Loss | $\beta$ |
# | --- | --- | --- |
# | assumed 0.25, 0.5, 0.75, 1 | 0.049 | 0.25 |
# | bass estimate 0.46, 0.66, 0.83, 1 | 0.042 | 0.42 |
# | free (0.75, 0.81, 0.88, 1) | 0.039 | 3.7 (unphysical) |
#
# The fundamental alone does not determine the velocities: the loss barely
# changes and the key-to-key scatter dominates the residual.
#
# Plan: as implemented, except that it calls the pulse a half-sine.

# %% [markdown]
# ## 5. Mode frequencies $f_n$ -- searched
#
# `step5_lines.py` -> `step5_lines.npz` (all 73 keys x 4 dynamics, and the robust lines per key)
#
# - Free extraction fails: the pickup's mixing products $r f_0 \pm k f_0$ look
#   like modes.  So the search is anchored at the ratios Gabrielli et al. (2020)
#   measured, $r_n$ = 7.1, 20.4, 39.7 (the sub mode 0.51 is left out: mains hum).
# - Per mode the tallest peak within $\pm$6 % / 4 % / 4 % of $r_n f_0$, harmonics
#   excluded ($\pm 0.03 f_0$), in 0.5 / 0.05 / 0.05 s of signal.
# - Accepted if a true local maximum, 12 dB above the window median, and below
#   $0.45 f_s$.
# - Robust line of a key (the ones step 6 uses): accepted in $\ge 3$ dynamics with
#   frequencies within 0.5 %; $f_n$ = median over those.  26 / 12 / 4 lines.
#
# Result over all recordings (accepted / searched): 7.1: 140 / 292, 20.4: 69 / 224,
# 39.7: 51 / 176.  Reliable where the four dynamics agree: 7.1 roughly E2-G#4,
# 20.4 D2-D#3, 39.7 A#1-B2.  In the bass the 7.1 search only finds the flank of
# $7 f_0$ at the guard edge; above A4 it finds noise.
#
# Plan: TBD, suggested the demodulation of step 1.  Now: anchored peak search.

# %% [markdown]
# ## 6. Mode excitation $c_n$ and decay $\sigma_n$ -- fitted
#
# Fixed: everything of steps 2-5.  Model: all modes through the pickup (so the
# mixing products are produced by the model and need not be told apart).
# Target: projection onto the mode lines, 8 frames of 64 periods over 0.8 s
# after $t_0$.  Start: scan of $A_n/A_0$ (20 values) per line.
#
# **`step6_fft.py`** -> `step6_fit.npz`
# - The robust lines of step 5: 31 keys (G0 skipped: 64 periods exceed 0.8 s).
# - Each key is fit on its own (no shared free parameter).
# - $c_n$ free **per dynamic**, $\sigma_n$ shared (`SIG_PER_DYN` to free it).
#   This absorbs the hammer's velocity dependence instead of imposing it, so
#   $c_n(\text{dyn})$ is no longer a model parameter in the sense of the plan.
#   (An earlier version shared $c_n$ over the dynamics, as the plan says, on five
#   keys: at A#3 its fit stayed ~12 dB above the measured 7.1 line, loss ~1.3.)
# - Result: loss typically 0.01-0.1 (A#3: 0.02); poor at A#4 (2.1), G#4 (0.53),
#   C5 (0.28).  Median $\sigma_n/\sigma_0$ = 10.5, 13.4, 15.9.
# - Hammer check, median over keys of $c_n(p)/c_n(f)$: +13.5 dB (7.1), +13.3 dB
#   (20.4).  The model predicts a high mode ~21 dB weaker at p than at f; measured
#   is ~7-8 dB, about as much as the bass fundamental (-6.8 dB).  Per key the
#   ratio scatters by $\pm$30 dB, so only medians are meaningful.
# - Synthetic test ($\sigma_n/\sigma_0$ = 10, 13, 16): recovered to ~0 dB almost
#   everywhere; worst A1 (39.7 only, few frames, 1.8 dB) and F#4-A#4 (~1 dB).
#
# Plan: TBD, suggested MR-STFT over S2 and S7.  Now: line projection over S2
# only (S7 would hold the left-out 0.51 mode).

# %% [markdown]
# ## Hammer model: what the data say (`step7_hammer_check.py`)
#
# Target: the free amplitude of each mode line per dynamic relative to f (dB), from
# step 6 ($c_n$ per dynamic times what the hammer hands over; independent of the
# hammer model, $c_n$ of the key cancels).  39 lines, 117 targets.  Every variant is
# judged on keys it was not fitted on (leave one key out).
#
# | Variant | rms (dB) | median abs. error (dB) | bias (dB) |
# | --- | --- | --- | --- |
# | no growth at all (0 dB) | 11.9 | 8.3 | +1.3 |
# | A: $\sin^2$ pulse, assumed velocities (current model) | 19.1 | 12.0 | -9.2 |
# | B: $\sin^2$ pulse, velocities from the bass | 17.3 | 10.2 | -4.7 |
# | C: pulse envelope without nulls, bass velocities | 12.8 | 8.1 | -5.9 |
# | D: $A_n \propto v^k$, bass velocities (1 parameter, $k \approx 0.6$) | 11.8 | 6.7 | -1.1 |
# | E: D with $k = 1$ + MLP correction $r(\log f_n\tau, \log v)$ | 11.7 | 7.2 | -0.5 |
#
# Findings:
#
# 1. **The assumed velocities are too far apart.**  In the bass $A_0 \propto v$, so
#    the step 2 amplitudes give p, mp, mf = 0.46, 0.66, 0.83 of f, not 0.25, 0.5, 0.75
#    (step 3 check).  This alone explains about half of the error of the current model
#    (A -> B).
# 2. **The modes barely grow from p to f.**  Median level relative to f: p -8 dB,
#    mp +2 dB, mf +1 dB -- mp and mf are even slightly louder than f.  The
#    $\sin^2$ pulse predicts growth like $v^{1+3\beta} \approx v^{1.76}$ beyond its main
#    lobe; the best power law is $k \approx 0.6$, less than a pure kick ($k = 1$).
# 3. **The nulls of the $\sin^2$ pulse make things worse.**  All lines lie at
#    $f_n\tau$ = 2-17, where the pulse spectrum has nulls at every integer; between p
#    and f, $\tau$ changes by 1.4x, so a line often crosses one.  Dropping the nulls
#    (B -> C) lowers the rms from 17.3 to 12.8 dB.
# 4. **The data themselves scatter a lot.**  Even the best variant misses by ~12 dB
#    rms, about as much as predicting no growth at all.  Per line the measured level
#    ratios scatter by $\pm$10-20 dB.  This is *not* because lines sit near the noise:
#    keeping only targets with all 8 frames above the noise (at the dynamic and at f)
#    leaves the scatter about the same (rms 11.6 -> 9.8 dB, IQR ~15 dB), and the
#    medians stay put (p about -8 to -10 dB, mp about +1 to +3 dB, mf about 0 dB).  More
#    likely: each recording is a single strike, and how strongly the high modes are
#    excited varies from strike to strike (strike point, felt), or $c_n$ and
#    $\sigma_n$ trade off in the fit.  Repeated strikes would be needed to tell.
# 5. **A neural network adds nothing here.**  With weak weight decay the MLP
#    overfits (rms 19 dB on unseen keys); with strong weight decay it becomes a smooth
#    trend and is as good as the one-parameter power law D.  The limit is the
#    scatter of the data, not the flexibility of the model.
#
# Consequence: for rendering keep $c_n$ per dynamic (step 6).  For a hammer model
# that interpolates between dynamics, the bass velocities and a smooth power law
# $A_n \propto v^{k}$ are enough; a learned correction is not supported by the data.

# %% [markdown]
# ## Rendering
#
# | Script | Content | Output |
# | --- | --- | --- |
# | `step2_render.py` | free fundamental through the pickup, dynamic f | `render/{note}-f_model.wav` |
# | `step3_render.py` | + hammer contact | `render/{note}-{dyn}_hammer.wav` |
# | `step6_render.py` | + modes of `step6_fit.npz`, $c_n$ of the dynamic (unknown ones from the nearest dynamic) | `render/{note}-{dyn}_modes.wav` |
#
# Check at A#3: level of the 7.1 mode relative to $f_0$, recording vs. render,
# within 1.5 dB at all four dynamics.

# %% [markdown]
# ## Evaluation measures (synthetic round trips)
#
# | Step | Synthetic test | Status |
# | --- | --- | --- |
# | 1 | - (measurement) | - |
# | 2 | `TAG = "syn"` in `step2_init.py`, `step2_fft.py` | available |
# | 3 | round trip in `hammer.py` | runs with every call |
# | 5 | - (search) | - |
# | 6 | `STEP6_TAG=syn` for `step6_fft.py` | passed, see above |

# %% [markdown]
# ## Assumptions, collected
#
# 1. The dynamics correspond to $v$ = 0.25, 0.5, 0.75, 1 (contradicted in the bass: 0.46, 0.66, 0.83).
# 2. Hammer: prescribed $\sin^2$ force, impulse $= v$, $\tau = \tau_0 v^{-\beta}$, $\tau_0$ log-linear over the keys, $\beta$ global, no damping or feedback during contact.
# 3. Tine: independent linear modes; decays do not depend on the strike.
# 4. Pickup: static nonlinearity, surface shape fixed from photos ($r$ = 4 mm, $a$ = 1.3 mm, $m$ = 1), no RLC filter, one global $\kappa$.
# 5. The tip moves along $x$ only.
# 6. The hammer has left the tine 10 ms after the onset (asserted in step 6).
# 7. H0-H4 of the first seconds carry only the pickup's distortion of the fundamental (step 2).
# 8. Mode frequencies stay within a few percent of the Gabrielli ratios over the whole keyboard (step 5).
# 9. The sub mode at 0.51 $f_0$ is ignored.
# 10. The four recordings of a key are the same tine, pickup and hammer at different velocities.

# %% [markdown]
# ## Open points
#
# - **Hammer model vs. data.** See "Hammer model: what the data say": use the bass
#   velocities and a smooth growth law for the modes; understand why mp/mf modes are as
#   loud as at f, and reduce the scatter of the per-dynamic mode levels.
# - **Keys without modes.** 42 of 73 keys have no robust mode.  Options: relax
#   `LINE_MIN_DYN` to 2, or carry $f_n/f_0$, $\sigma_n/\sigma_0$ and $A_n/A_0$ over
#   from neighbouring keys.
# - **Poor keys in step 6.** A#4, G#4, C5 (detail plot with `SHOW_KEY`).
# - **Decays per dynamic.** Run `step6_fft.py` with `SIG_PER_DYN = True` to test assumption 3.
# - **Documentation.** `fitting.py` still says half-sine;
#   steps 5 and 6 are still TBD there; the H0-H6 claim does not hold for the 7.1 mode.
