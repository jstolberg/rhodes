"""Render the step-2 model of every fitted key (dynamic FIT_DYN) as an 8 s wav: the free
mode x(t) = p_o + A_0 e^{-sigma_0 t} sin 2 pi f_0 t through the pickup, kappa applied, no
hammer and no RLC.  Reads step2_fit.npz, writes render/{note}-{dyn}_model.wav next to
the recordings' level."""
import os

import jax.numpy as jnp
import numpy as np
import scipy.io.wavfile as wavfile

from step2_lib import DYNS, J_FIT, TABLE, eps_free

FS, DUR, LEAD = 48000, 8.0, 10e-3
OUT = "render"

fit = np.load("step2_fit.npz", allow_pickle=True)
os.makedirs(OUT, exist_ok=True)
t = jnp.arange(int(DUR * FS)) / FS
for i, note in enumerate(fit["notes"]):
    A0, sig, f0 = fit["A0"][i, J_FIT], fit["sigma"][i], fit["f0"][i]
    x = float(fit["kappa"]) * np.asarray(eps_free(t, jnp.array([A0]), jnp.array([sig]), jnp.array([f0]),
                                                  fit["p_o"][i], fit["p_d"][i], TABLE))
    x = np.concatenate([np.zeros(int(LEAD * FS)), x])[:int(DUR * FS)]
    path = f"{OUT}/{note}-{DYNS[J_FIT]}_model.wav"
    wavfile.write(path, FS, x.astype(np.float32))
    print(f"{path}: f0 {f0:.1f} Hz, A_0 {A0*1e3:.2f} mm, sigma {sig:.2f} 1/s, peak {np.abs(x).max():.3f}")
