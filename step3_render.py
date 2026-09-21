"""Render the full step-2 + step-3 model of selected keys as 8 s wavs: hammer contact
(tau_0, beta, c_0 of step 3), then the free fundamental (sigma_0 of step 2) through the
fitted pickup (p_o, p_d, kappa), no RLC.  Writes render/{note}-{dyn}_hammer.wav.

    python step3_render.py [E0 E2 E4 E6] [--dyn p mp mf f]
"""
import os
import sys

import jax.numpy as jnp
import numpy as np
import scipy.io.wavfile as wavfile

from model import epsilon
from step2_lib import DYNS, TABLE, VELS

FS, DUR, LEAD = 48000, 8.0, 10e-3
OUT = "render"

args  = sys.argv[1:]
dyns  = args[args.index("--dyn") + 1:] if "--dyn" in args else ["f"]
notes = [a for a in args if a not in dyns and a != "--dyn"] or ["E0", "E2", "E4", "E6"]

fit2 = np.load("step2_fit.npz", allow_pickle=True)
fit3 = np.load("step3_fit.npz", allow_pickle=True)
assert (fit2["notes"] == fit3["notes"]).all()
os.makedirs(OUT, exist_ok=True)
t = jnp.arange(int(DUR * FS)) / FS
for note in notes:
    i = list(fit2["notes"]).index(note)
    p = dict(c=jnp.array([fit3["c0"][i]]), lam=jnp.array([fit2["sigma"][i]]), f_modes=jnp.array([fit2["f0"][i]]),
             tau_0=float(fit3["tau0"][i]), beta=float(fit3["beta"]), p_d=float(fit2["p_d"][i]), p_o=float(fit2["p_o"][i]))
    for dyn in dyns:
        vel = VELS[DYNS.index(dyn)]
        x = float(fit2["kappa"]) * np.asarray(epsilon(t, p, TABLE, vel=vel))
        x = np.concatenate([np.zeros(int(LEAD * FS)), x])[:int(DUR * FS)]
        path = f"{OUT}/{note}-{dyn}_hammer.wav"
        wavfile.write(path, FS, x.astype(np.float32))
        print(f"{path}: f0 {p['f_modes'][0]:.1f} Hz, tau {p['tau_0'] * vel ** -p['beta'] * 1e3:.2f} ms, "
              f"c_0 {p['c'][0]:.3e}, peak {np.abs(x).max():.3f}")
