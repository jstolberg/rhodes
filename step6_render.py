"""Render the full model of the step-6 keys as 8 s wavs: hammer contact, then the
fundamental (steps 2, 3) and the inharmonic modes (steps 5, 6) through the fitted pickup
(p_o, p_d, kappa), no RLC.

step6_fit.npz holds one c_n per dynamic, so every dynamic is rendered with its own c_n
(the measured growth of the modes from p to f, not the one the hammer model predicts).
If a mode had no frames above the noise at some dynamic, its c_n there is unknown; it is
then taken from the nearest dynamic that has one.

Writes render/{note}-{dyn}_modes.wav; compare with the recording Samples/{note}-{dyn}.wav
and with {note}-{dyn}_hammer.wav of step3_render.py (fundamental only).

    python step6_render.py                    # all keys of step 6, all dynamics
    python step6_render.py E3 A#3 --dyn p f   # selected keys and dynamics
"""
import os
import sys

import jax.numpy as jnp
import numpy as np
import scipy.io.wavfile as wavfile

from model import disp_bound, epsilon
from step2_lib import DISP_MAX, DYNS, TABLE, VELS

FS, DUR, LEAD = 48000, 8.0, 10e-3     # sample rate, length (s), silence before the onset (s)
OUT = "render"

# Fitted parameters: steps 2 and 3 (fundamental, pickup, hammer), step 6 (modes)
fit2 = np.load("step2_fit.npz", allow_pickle=True)
fit3 = np.load("step3_fit.npz", allow_pickle=True)
fit6 = np.load("step6_fit.npz", allow_pickle=True)


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


# Command line: key names, optionally followed by --dyn and dynamics
args  = sys.argv[1:]
dyns  = args[args.index("--dyn") + 1:] if "--dyn" in args else list(DYNS)
notes = [a for a in args if a not in dyns and a != "--dyn"] or list(fit6["notes"])

os.makedirs(OUT, exist_ok=True)
t = jnp.arange(int(DUR * FS)) / FS
for note in notes:
    i = list(fit2["notes"]).index(note)                   # index in steps 2 and 3
    m = list(fit6["notes"]).index(note)                   # index in step 6
    keep = fit6["ok"][m]                                  # robust modes of step 5
    c_all = c_filled(fit6["c"][m], fit6["used"][m])       # (n_dyn, N)

    for dyn in dyns:
        j = DYNS.index(dyn)
        # All modes of this key: the fundamental first, then the kept inharmonic modes
        p = dict(c=jnp.concatenate([jnp.array([fit3["c0"][i]]), jnp.asarray(c_all[j][keep])]),
                 lam=jnp.concatenate([jnp.array([fit2["sigma"][i]]), jnp.asarray(fit6["sig"][m][keep])]),
                 f_modes=jnp.concatenate([jnp.array([fit2["f0"][i]]), jnp.asarray(fit6["f_modes"][m][keep])]),
                 tau_0=float(fit3["tau0"][i]), beta=float(fit3["beta"]),
                 p_d=float(fit2["p_d"][i]), p_o=float(fit2["p_o"][i]))
        reach = disp_bound(p)                             # the pickup table only covers DISP_MAX
        if reach > DISP_MAX:
            print(f"warning {note}-{dyn}: displacement up to {reach * 1e3:.2f} mm "
                  f"> DISP_MAX {DISP_MAX * 1e3:.1f} mm")

        # Pickup voltage, with LEAD s of silence in front, cut to DUR s
        x = float(fit2["kappa"]) * np.asarray(epsilon(t, p, TABLE, vel=VELS[j]))
        x = np.concatenate([np.zeros(int(LEAD * FS)), x])[:int(DUR * FS)]
        path = f"{OUT}/{note}-{dyn}_modes.wav"
        wavfile.write(path, FS, x.astype(np.float32))
        print(f"{path}: f_n/f_0 {np.round(np.asarray(p['f_modes'][1:]) / fit2['f0'][i], 3)}, "
              f"sigma_n {np.round(np.asarray(p['lam'][1:]), 2)} 1/s, peak {np.abs(x).max():.3f}",
              flush=True)
