# %% [markdown]
# # Pickup for the Rhodes model
#
# Pickup is modelled as magnetically charged surface $S$, assumed 
# to have circular cross-section at z = 0, and for z > 0, z is given 
# by a continuous function in x and y. The most simple examble being 
# a typical guitar pickup with $z$ identically given by
# $$z (x, y) = 0$$
# Modelling the tip of the Rhodes tine as a point $\alpha = (x', y', z')$, 
# we consider the magnetic effect at this point induced by a single point
# on the surface $\beta = (x,y,z) \in S$ along the z-axis,
# $$ B_z(\beta) = B_0 \frac{z' - z}{\|\alpha - \beta \|^3}.$$
# The magnetic field induced by the full surface at the tine tip $\alpha$,
# is then given by 
# $$ \mathcal{B}_z(\alpha) = \int_S \sigma B_z(\beta) \text{d}\beta $$
# where $\sigma$ is the magnetic charge density across $S$. This causes a 
# proportinal magnetisation of the tip of the tine, which in turn affects the magnetic
# field at the surface $S$, which due to the symmetry of the setting is identical
# up to a scaling factor $\gamma$, causing a magnetic flux
# $$ \Psi(\alpha) \approx \gamma \mathcal{B}_z(\alpha)^2$$
# Given an explicit expression for $S$, $\mathcal{B}_z$ can be solved numerically
# for a given point $\alpha$. The induced voltage by the pickup coil for 
# a moving tip $\alpha(t)$ is in turn given by
# $$ \epsilon = - \frac{\text{d} \Psi(\alpha)}{\text{d} t} .$$
# Assuming a simple, modal model for $\alpha(t)$ in the x-axis means
# holding $y' = 0$ and $z' = p_d$ (pickup distance parameter) fixed and letting
# $$x' = \alpha_x(t) = p_o + \sum_q A_q e^{- \lambda_q t} \sin(2\pi f_q t) $$
# where $A_q$ and $\lambda_q$ is the amplitude and decay for mode $q$ 
# with frequenzy $f_q$, and $p_o$ is the offset of the tine tip 
# from the pickup center along the x-axis.
# 
# If we fix a grid on the surface $S$, and approximate $\Psi(\alpha)$
# as the sum across this grid, it means writing $\Psi$ becomes an
# easily differentiable function in $t$, enabling us to fit the full 
# model, including modes, based on voltage outputs of the Rhodes output.
# 
# The model parameters are 
# - $N$ - Number of fitted modes.
# - $A_i, \lambda_i, f_i$ - Modal parameters for $i = 0 \ldots N-1$.
# - $\kappa$ - Single multiplicative factor, encompassing $\gamma$, $\sigma$ and $B_0$.
# - $S$ - The surface shape.
# - $p_d, p_o$ - Tine distance respectively offset w.r.t. the pickup.
# 
# As $\Psi$ depends only on the position $\alpha$, a lookup table 
# with precomputed, numerical solutions to the integral can be computed
# for positions of $\alpha$, once parameters have been fitted, enabling
# real-time synthesis of the fitted instrument.
# 
# **Missing elements:**
# - Model $\alpha$ on an arc $z' = f(x')$ instead of fixed $z'$.
# - The coil implements a RLC circuit, implicitly implying a resonant,
# low-pass filter, which could be modelled too.

# %%
from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import optax
from functools import partial
from IPython.display import Audio

jax.config.update("jax_enable_x64", True)

print("jax", jax.__version__, "devices:", jax.devices())

# %% [markdown]
# # Example: Guitar pickup, $z(x,y) = 0$
# 

# %%
def make_surface(N_r=64, N_phi=64, r_max=2e-3, z_profile=None):
    """Radial surface"""
    dr, dphi = r_max / N_r, 2*jnp.pi / N_phi
    r_c   = (jnp.arange(N_r) + 0.5) * dr
    phi_c = (jnp.arange(N_phi) + 0.5) * dphi
    R, PHI = jnp.meshgrid(r_c, phi_c, indexing='ij')

    if z_profile is None:                    # flat guitar pickup
        Z, metric = jnp.zeros_like(R), 1.0
    else:
        Z = z_profile(R)
        dzdr = jax.vmap(jax.grad(z_profile))(R.ravel()).reshape(R.shape)
        metric = jnp.sqrt(1.0 + dzdr**2)

    area = R * metric * dr * dphi            # r_bar * sqrt(1+z'^2) * dr * dphi

    return (jnp.stack([R*jnp.cos(PHI), R*jnp.sin(PHI), Z], axis=-1).reshape(-1, 3),
            area.ravel())

def alpha(t, p):
    """scalar t -> (3,) tip position"""
    A, lam, f = p['A'], p['lam'], p['f']
    disp = jnp.sum(A * jnp.exp(-lam * t) * jnp.sin(2*jnp.pi * f * t))
    x = disp + p['p_o']
    z = p['p_d'] + p['r_tine'] - jnp.sqrt(
            jnp.maximum(p['r_tine']**2 - disp**2, 0.0))
    return jnp.array([x, 0.0, z])

def B_z(a, pts, w):
    """(3,) tip position -> scalar axial field at the tip"""
    d  = a - pts                              # (M, 3)
    r3 = jnp.sum(d*d, axis=-1)**1.5
    return jnp.dot(d[:, 2] / r3, w)

def Psi(a, pts, w, gamma):
    return gamma * B_z(a, pts, w)**2

@partial(jax.jit, static_argnames=('chunk',))
def epsilon(t, p, pts, w, chunk=256):
    psi_of_t = lambda tt: Psi(alpha(tt, p), pts, w, p['gamma'])
    g = jax.grad(psi_of_t)

    T = t.shape[0]
    pad = (-T) % chunk                       # pad up to a multiple of chunk
    t_pad = jnp.concatenate([t, jnp.zeros(pad, t.dtype)])
    t_blk = t_pad.reshape(-1, chunk)         # (n_chunks, chunk)

    out = jax.lax.map(jax.vmap(g), t_blk)    # (n_chunks, chunk)
    return -out.reshape(-1)[:T]

# Example
pts, w = make_surface(r_max=3e-3)
ratios = jnp.array([0.68, 1.0, 7.11, 20.25])
f_0 = 440
p = dict(
    A=jnp.array([0.5, 1.0, 0.2, 0.1]) * 1e-4, 
    lam=jnp.array([3, 1, 10, 20]),
    f=f_0 * ratios, 
    p_d=1e-3, 
    p_o=3e-4,
    r_tine=10e-2,
    gamma=1.0)

fs = 48000.0
l = 2
t   = jnp.arange(0, l*fs) / fs
eps = epsilon(t, p, pts, w)

# %%
# Plot results

eps_np = np.asarray(eps, dtype=np.float64)

f, t_spec, Sxx = signal.spectrogram(
    eps_np, fs=fs,
    nperseg=2048, noverlap=1536,     # 75% overlap
    window='hann', scaling='spectrum'
)

Sdb = 10*np.log10(Sxx + 1e-20)       # power -> dB, epsilon guards log(0)

plt.figure(figsize=(10, 5))
plt.pcolormesh(t_spec, f, Sdb,
               vmin=Sdb.max()-80, vmax=Sdb.max(),   # 80 dB range
               shading='gouraud', cmap='magma')
plt.ylim(0, 8000)                    # partials live low; drop the empty top
plt.xlabel('time (s)'); plt.ylabel('frequency (Hz)')
plt.colorbar(label='dB')
plt.tight_layout(); plt.show()

# %%
x = np.asarray(eps, dtype=np.float64)
x = x / (np.max(np.abs(x)) + 1e-12)   # normalise to ±1

Audio(x, rate=fs) 

# %%
