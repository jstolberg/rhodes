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
# The coil itself implements a RLC filter with cutoff frequency $f_\text{filter}$
# and resonance $Q_\text{filter}$, which is applied to the output signal.
# 
# The model parameters are 
# - $N$ - Number of fitted modes.
# - $A_i, \lambda_i, f_i$ - Modal parameters for $i = 0 \ldots N-1$.
# - $\kappa$ - Single multiplicative factor, encompassing $\gamma$, $\sigma$ and $B_0$.
# - $S$ - The surface shape.
# - $p_d, p_o$ - Tine distance respectively offset w.r.t. the pickup.
# - $f_\text{filter}, Q_\text{filter}$ - The implicit RLC filter implemented by the coil circuit.
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
# Pickup surfaces

# ---------- surface definitions ----------

def z_trapz(X, Y, a=0.5e-3, m=0.3, z0=0.0):
    """Rhodes pickup: Flat plateau |x| <= a, chamfers of slope m, constant in y."""
    return z0 - m * jnp.maximum(jnp.abs(X) - a, 0.0)


def z_flat(X, Y, z0=0.0):
    """Guitar pickup: flat disc."""
    return jnp.full_like(X, z0)


def disc_mask(X, Y, r_max):
    return (X**2 + Y**2) <= r_max**2


# ---------- grid + quadrature ----------

def make_surface(z_fn, N=128, r_max=2e-3, mask_fn=disc_mask):
    e = jnp.linspace(-r_max, r_max, N + 1)
    c = 0.5 * (e[:-1] + e[1:])
    d = c[1] - c[0]
    X, Y = jnp.meshgrid(c, c, indexing='ij')

    Z = z_fn(X, Y)

    grad_z = jax.grad(z_fn, argnums=(0, 1))            # -> (dz/dx, dz/dy)
    gx, gy = jax.vmap(grad_z)(X.ravel(), Y.ravel())    # each (N*N,)
    metric = jnp.sqrt(1.0 + gx**2 + gy**2).reshape(X.shape)

    area = d * d * metric * mask_fn(X, Y, r_max)

    pts = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return pts, area.ravel()

# --------- Plotting surfaces ------------

def plot_surface(pts, area, scale=1e3, unit='mm', equal_z=True,
                 elev=22, azim=-60, cmap='viridis', ax=None):
    """Inspect the surface produced by make_surface()."""
    N = int(round(np.sqrt(pts.shape[0])))
    P = np.asarray(pts).reshape(N, N, 3)
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]

    Z = np.where(np.asarray(area).reshape(N, N) > 0, Z, np.nan)  # mask

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X * scale, Y * scale, Z * scale,
                    cmap=cmap, linewidth=0, antialiased=True,
                    rstride=2, cstride=2)

    ax.set_xlabel(f'x ({unit})'); ax.set_ylabel(f'y ({unit})')
    ax.set_zlabel(f'z ({unit})')
    ax.set_box_aspect((1, 1, 1))

    span = max(np.ptp(X), np.ptp(Y)) * scale / 2
    ax.set_xlim(-span, span); ax.set_ylim(-span, span)
    if equal_z:
        zc = np.nanmean(Z) * scale
        ax.set_zlim(zc - span, zc + span)

    ax.view_init(elev=elev, azim=azim)
    return ax

pts, w = make_surface(partial(z_trapz, a=0.5e-3, m=0.4), r_max=3e-3)
plot_surface(pts, w)

# %%

def alpha(t, p):
    """scalar t -> (3,) tip position"""
    A, lam, f = p['A'], p['lam'], p['f_modes']
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

@partial(jax.jit, static_argnames=('fs',))
def RLC(sig, p, fs=48000.0):
    """Resonant 2nd-order lowpass. Differentiable w.r.t. sig, f, Q."""
    # pre-warped analogue frequency (bilinear transform)
    f, Q = p["f_filter"], p["Q_filter"]
    w0 = 2.0 * fs * jnp.tan(jnp.pi * f / fs)
    K, K2 = w0 / (2.0 * fs), (w0 / (2.0 * fs))**2

    norm = 1.0 + K/Q + K2
    b = jnp.array([K2, 2.0*K2, K2]) / norm
    a1 = 2.0 * (K2 - 1.0) / norm
    a2 = (1.0 - K/Q + K2) / norm

    def step(state, xn):
        x1, x2, y1, y2 = state
        yn = b[0]*xn + b[1]*x1 + b[2]*x2 - a1*y1 - a2*y2
        return (xn, x1, yn, y1), yn

    _, y = jax.lax.scan(step, (0.0, 0.0, 0.0, 0.0), sig)
    return y

# Example
ratios = jnp.array([0.51, 1.0, 7.11, 20.25])
f_0 = 440
p = dict(
    A=jnp.array([0.1, 1.0, 0.1, 0.06]) * 1e-5, 
    lam=jnp.array([40, 1, 40, 60]),
    f_modes=f_0 * ratios, 
    p_d=1e-3, 
    p_o=5e-4,
    r_tine=10e-2,
    gamma=1.0,
    f_filter=5e3,
    Q_filter=2.0)

fs = 48000.0
l = 2
t   = jnp.arange(0, l*fs) / fs
eps = epsilon(t, p, pts, w)
eps = RLC(eps, p, fs)

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
