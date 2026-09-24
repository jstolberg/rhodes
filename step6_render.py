"""Render the full model of the step-6 keys as 8 s wavs: hammer contact, then the
fundamental (steps 2, 3) and the inharmonic modes (steps 5, 6) through the fitted pickup
(p_o, p_d, kappa), no RLC.  Writes render/{note}-{dyn}_modes.wav; compare with
{note}-{dyn}_hammer.wav of step3_render.py, which has the fundamental only.

    python step6_render.py [E3 A#3] [--dyn p mp mf f]
"""
import os
import sys

import jax.numpy as jnp
import numpy as np
import scipy.io.wavfile as wavfile

from model import disp_bound, epsilon
from step2_lib import DISP_MAX, DYNS, TABLE, VELS

FS, DUR, LEAD = 48000, 8.0, 10e-3
OUT = "render"

fit2 = np.load("step2_fit.npz", allow_pickle=True)
fit3 = np.load("step3_fit.npz", allow_pickle=True)
fit6 = np.load("step6_fit.npz", allow_pickle=True)

args  = sys.argv[1:]
dyns  = args[args.index("--dyn") + 1:] if "--dyn" in args else ["f"]
notes = [a for a in args if a not in dyns and a != "--dyn"] or list(fit6["notes"])

os.makedirs(OUT, exist_ok=True)
t = jnp.arange(int(DUR * FS)) / FS
for note in notes:
    i, m = list(fit2["notes"]).index(note), list(fit6["notes"]).index(note)
    keep = fit6["ok"][m]                                     # modes step 5 accepted
    p = dict(c=jnp.concatenate([jnp.array([fit3["c0"][i]]), jnp.asarray(fit6["c"][m][keep])]),
             lam=jnp.concatenate([jnp.array([fit2["sigma"][i]]), jnp.asarray(fit6["sig"][m][keep])]),
             f_modes=jnp.concatenate([jnp.array([fit2["f0"][i]]), jnp.asarray(fit6["f_modes"][m][keep])]),
             tau_0=float(fit3["tau0"][i]), beta=float(fit3["beta"]),
             p_d=float(fit2["p_d"][i]), p_o=float(fit2["p_o"][i]))
    reach = disp_bound(p)                                    # the table only covers DISP_MAX
    if reach > DISP_MAX:
        print(f"warning {note}: displacement up to {reach * 1e3:.2f} mm > DISP_MAX {DISP_MAX * 1e3:.1f} mm")
    for dyn in dyns:
        vel = VELS[DYNS.index(dyn)]
        x = float(fit2["kappa"]) * np.asarray(epsilon(t, p, TABLE, vel=vel))
        x = np.concatenate([np.zeros(int(LEAD * FS)), x])[:int(DUR * FS)]
        path = f"{OUT}/{note}-{dyn}_modes.wav"
        wavfile.write(path, FS, x.astype(np.float32))
        print(f"{path}: f_n/f_0 {np.round(np.asarray(p['f_modes'][1:]) / fit2['f0'][i], 3)}, "
              f"sigma_n {np.round(np.asarray(p['lam'][1:]), 2)} 1/s, peak {np.abs(x).max():.3f}")
