"""Render the fitted model as it stands after a step: 8 s at 48 kHz, LEAD s of silence
before the onset, through the fitted pickup (p_o, p_d, kappa), no RLC.

* step 1 -- a plain decaying sine at f_0, lambda_0 (no pickup, no hammer)
* step 2 -- the free fundamental x(t) = p_o + A_0 e^{-sigma_0 t} sin 2 pi f_0 t (no hammer)
* step 3 -- hammer contact (tau_0, beta, c_0), then the free fundamental
* step 4 -- as step 3, plus the inharmonic modes

    python synth.py 1|2|3|4 [E0 E2 E4 E6] [--dyn f]     -> render/{note}-{dyn}_step{n}.wav

From Python: synth.render(step, note, dyn) returns the signal, synth.recording(note, dyn)
the recording at the same level (the model is referenced to it by kappa).
"""
import os
import sys

import jax.numpy as jnp
import numpy as np
import scipy.io.wavfile as wavfile

import step1_fundamental as step1_results
from model import disp_bound, epsilon
from step2_pickup import DISP_MAX, DYNS, TABLE, VELS, eps_free, load_wave

FS, DUR, LEAD = 48000, 8.0, 10e-3     # sample rate, length (s), silence before the onset (s)
OUT = "render"


def load(step):
    return np.load({2: "results/step2_pickup.npz", 3: "results/step3_hammer.npz",
                    4: "results/step4_modes.npz"}[step], allow_pickle=True)


def _finish(x):
    """LEAD s of silence in front, cut to DUR s."""
    return np.concatenate([np.zeros(int(LEAD * FS)), x])[:int(DUR * FS)]


def step1(note, dyn):
    """Only f_0 and lambda_0 are known: A e^{-lambda_0 t} sin 2 pi f_0 t, at the level of the
    recording's fundamental (amp0 is the level of the mixed-down envelope, half the sine's).
    Where lambda_0 or amp0 was not measured, the median over the note's other dynamics
    (no decay if there is none)."""
    tab, notes, dyns = step1_results.load()
    i, j = list(notes).index(note), list(dyns).index(dyn)
    row = tab[i, j]
    lam = row["lam"] if row["ok"] else (np.nanmedian(tab["lam"][i]) if tab["ok"][i].any() else 0.0)
    amp = row["amp0"] if np.isfinite(row["amp0"]) else np.nanmedian(tab["amp0"][i])
    t = np.arange(int(DUR * FS)) / FS
    x = 2 * amp * np.exp(-lam * t) * np.sin(2 * np.pi * row["f"] * t)
    print(f"{note}-{dyn} step 1: f0 {row['f']:.2f} Hz, lambda_0 {lam:.3f} 1/s")
    return _finish(x)


def step2(note, dyn):
    fit = load(2)
    i, j = list(fit["notes"]).index(note), DYNS.index(dyn)
    t = jnp.arange(int(DUR * FS)) / FS
    A0, sig, f0 = fit["A0"][i, j], fit["sigma"][i], fit["f0"][i]
    x = float(fit["kappa"]) * np.asarray(eps_free(t, jnp.array([A0]), jnp.array([sig]), jnp.array([f0]),
                                                  fit["p_o"][i], fit["p_d"][i], TABLE))
    print(f"{note}-{dyn} step 2: f0 {f0:.1f} Hz, A_0 {A0*1e3:.2f} mm, sigma {sig:.2f} 1/s")
    return _finish(x)


def step3(note, dyn):
    fit2, fit3 = load(2), load(3)
    assert (fit2["notes"] == fit3["notes"]).all()
    i = list(fit2["notes"]).index(note)
    t = jnp.arange(int(DUR * FS)) / FS
    p = dict(c=jnp.array([fit3["c0"][i]]), lam=jnp.array([fit2["sigma"][i]]), f_modes=jnp.array([fit2["f0"][i]]),
             tau_0=float(fit3["tau0"][i]), beta=float(fit3["beta"]), p_d=float(fit2["p_d"][i]), p_o=float(fit2["p_o"][i]))
    vel = VELS[DYNS.index(dyn)]
    x = float(fit2["kappa"]) * np.asarray(epsilon(t, p, TABLE, vel=vel))
    print(f"{note}-{dyn} step 3: f0 {p['f_modes'][0]:.1f} Hz, tau {p['tau_0'] * vel ** -p['beta'] * 1e3:.2f} ms, "
          f"c_0 {p['c'][0]:.3e}")
    return _finish(x)


def c_filled(c, used):
    """c (n_dyn, N) with unknown entries (used False) taken from the nearest dynamic that
    has a value.  Lines without any value stay 0 (they are dropped anyway)."""
    c = c.copy()
    for n in range(c.shape[1]):
        have = np.flatnonzero(used[:, n])                 # dynamics with a fitted c_n
        if len(have) == 0:
            continue
        for j in range(c.shape[0]):
            if not used[j, n]:
                c[j, n] = c[have[np.argmin(np.abs(have - j))], n]
    return c


def step4(note, dyn):
    """step4_modes.npz holds one c_n per dynamic, so every dynamic is rendered with its own
    c_n.  If a mode had no frames above the noise at some dynamic, its c_n there is unknown;
    it is then taken from the nearest dynamic that has one."""
    fit2, fit3, fit4 = load(2), load(3), load(4)
    i = list(fit2["notes"]).index(note)                   # index in steps 2 and 3
    m = list(fit4["notes"]).index(note)                   # index in step 4
    keep = fit4["ok"][m]                                  # robust modes of step 4 A
    c_all = c_filled(fit4["c"][m], fit4["used"][m])       # (n_dyn, N)
    j = DYNS.index(dyn)
    t = jnp.arange(int(DUR * FS)) / FS
    # All modes of this key: the fundamental first, then the kept inharmonic modes
    p = dict(c=jnp.concatenate([jnp.array([fit3["c0"][i]]), jnp.asarray(c_all[j][keep])]),
             lam=jnp.concatenate([jnp.array([fit2["sigma"][i]]), jnp.asarray(fit4["sig"][m][keep])]),
             f_modes=jnp.concatenate([jnp.array([fit2["f0"][i]]), jnp.asarray(fit4["f_modes"][m][keep])]),
             tau_0=float(fit3["tau0"][i]), beta=float(fit3["beta"]),
             p_d=float(fit2["p_d"][i]), p_o=float(fit2["p_o"][i]))
    reach = disp_bound(p)                                 # the pickup table only covers DISP_MAX
    if reach > DISP_MAX:
        print(f"warning {note}-{dyn}: displacement up to {reach * 1e3:.2f} mm "
              f"> DISP_MAX {DISP_MAX * 1e3:.1f} mm")
    x = float(fit2["kappa"]) * np.asarray(epsilon(t, p, TABLE, vel=VELS[j]))
    print(f"{note}-{dyn} step 4: f_n/f_0 {np.round(np.asarray(p['f_modes'][1:]) / fit2['f0'][i], 3)}, "
          f"sigma_n {np.round(np.asarray(p['lam'][1:]), 2)} 1/s")
    return _finish(x)


def render(step, note, dyn="f"):
    return {1: step1, 2: step2, 3: step3, 4: step4}[step](note, dyn)


def recording(note, dyn="f"):
    """The recording, cut to DUR s."""
    x, fs = load_wave(note, dyn)
    return x[:int(DUR * fs)]


if __name__ == "__main__":
    args  = [] if "ipykernel" in sys.argv[0] else sys.argv[1:]     # cell by cell: not our command line
    step  = int(args.pop(0)) if args and args[0].isdigit() else 4
    dyns  = args[args.index("--dyn") + 1:] if "--dyn" in args else ["f"]
    notes = [a for a in args if a not in dyns and a != "--dyn"] or ["E0", "E2", "E4", "E6"]
    os.makedirs(OUT, exist_ok=True)
    for note in notes:
        for dyn in dyns:
            x = render(step, note, dyn)
            path = f"{OUT}/{note}-{dyn}_step{step}.wav"
            wavfile.write(path, FS, x.astype(np.float32))
            print(f"  -> {path}, peak {np.abs(x).max():.3f}", flush=True)
