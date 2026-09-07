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


def drop_masked(pts, area):
    """Discard the points the mask zeroed.

    They cost a full distance evaluation each but contribute nothing (~21% of
    the grid for a disc).  Must run outside jit -- boolean indexing needs a
    static shape -- and after plot_surface, which wants the full N x N grid.
    """
    keep = np.asarray(area) > 0
    return pts[keep], area[keep]

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
pts, w = drop_masked(pts, w)        # idempotent, so re-running the cell is safe

# %%

def alpha(t, p):
    """scalar t -> (3,) tip position"""
    A, lam, f = p['A'], p['lam'], p['f_modes']
    disp = jnp.sum(A * jnp.exp(-lam * t) * jnp.sin(2*jnp.pi * f * t))
    x = disp + p['p_o']
    z = p['p_d']
    return jnp.array([x, 0.0, z])

# ---------- Psi lookup table ----------
# With y' = 0 and z' = p_d fixed, alpha traces a straight line in x, so Psi
# is a function of the single scalar x.  The O(M) surface sum can therefore be
# hoisted out of the per-sample loop: tabulate once, then interpolate.

@partial(jax.jit, static_argnames=('n', 'chunk'))
def psi_table(p, pts, w, n=4096, pad=1.5, chunk=256):
    """Precompute Psi on a uniform x-grid -> (x0, dx, values).

    The grid is centred on p_o and spans pad * sum|A| either side, i.e. the
    range the tip can reach if every mode peaks at once.  Grid *placement* is
    a discretisation choice and is held constant (stop_gradient); the tabulated
    *values* stay differentiable w.r.t. p_d and the surface (pts, w), so
    the table may be rebuilt inside a fitting step.

    Specialised to y' = 0: with the surface fixed, everything in the sum but
    (x - x_j)^2 is independent of x, so with num_j = w_j * dz_j and
    c_j = y_j^2 + dz_j^2 (dz_j = p_d - z_j) built once in O(M),

        Psi(x) = ( sum_j num_j / ((x - x_j)^2 + c_j)^(3/2) )^2

    which is the discretised  Psi = (int_S sigma (z'-z)/||a-b||^3 db)^2  of the
    derivation above, dropped from O(n*M) to O(M) setup + O(n*M) muls.
    """
    span = pad * jnp.sum(jnp.abs(p['A']))
    x0, x1 = jax.lax.stop_gradient(
        jnp.array([p['p_o'] - span, p['p_o'] + span]))
    dx = (x1 - x0) / (n - 1)
    xs = x0 + dx * jnp.arange(n)

    xj  = pts[:, 0]
    dz  = p['p_d'] - pts[:, 2]                        # (M,)
    num = w * dz                                      # numerator weight
    c   = pts[:, 1]**2 + dz**2                        # y^2 + dz^2

    def psi_at(x):
        r = (x - xj)**2 + c
        # rsqrt(r^3) rather than r**1.5: pow lowers ~5x slower than mul+rsqrt
        return jnp.sum(num * jax.lax.rsqrt(r * r * r))**2

    m = (-n) % chunk                                  # pad up to a chunk
    xs_pad = jnp.concatenate([xs, jnp.full(m, x0, xs.dtype)])
    vals = jax.lax.map(jax.vmap(psi_at), xs_pad.reshape(-1, chunk))

    return x0, dx, vals.reshape(-1)[:n]


def psi_lookup(a, table):
    """(..., 3) tip positions -> (...) Psi, read off the table.

    Catmull-Rom cubic, so the result is C1 in x: unlike jnp.interp, its
    derivative is continuous, which is what epsilon = -dPsi/dt needs.
    """
    x0, dx, v = table
    n = v.shape[0]

    s = jnp.clip((a[..., 0] - x0) / dx, 0.0, n - 1.0)
    i = jnp.clip(jnp.floor(s).astype(int), 1, n - 3)   # integer -> no tangent
    u = s - i                                          # so du/dx = 1/dx

    p0, p1, p2, p3 = v[i-1], v[i], v[i+1], v[i+2]
    return 0.5 * (2.0*p1
                  + (-p0 + p2) * u
                  + (2.0*p0 - 5.0*p1 + 4.0*p2 - p3) * u**2
                  + (-p0 + 3.0*p1 - 3.0*p2 + p3) * u**3)


@jax.jit
def epsilon(t, p, table):
    """time samples -> induced voltage -dPsi/dt, via the Psi table.

    Psi is scalar->scalar in t, so forward mode costs ~2x a primal eval at O(1)
    memory.  With the O(M) surface sum hoisted into the table there is nothing
    left to chunk, so the whole signal goes through one vmap.
    """
    psi_of_t = lambda tt: psi_lookup(alpha(tt, p), table)
    dpsi = lambda tt: jax.jvp(psi_of_t, (tt,), (jnp.ones_like(tt),))[1]
    return -jax.vmap(dpsi)(t)

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

    # unroll: the body is ~10 flops, so an un-unrolled scan spends all its time
    # on XLA loop-carry overhead (~950x slower here).  8 is the sweet spot;
    # 16/32 lose again to code bloat.
    _, y = jax.lax.scan(step, (0.0, 0.0, 0.0, 0.0), sig, unroll=8)
    return y

# Example
p = dict(
    A=jnp.array([0.01, 1.0]) * 5e-3, 
    lam=jnp.array([0.1, 1.0]),
    f_modes=jnp.array([60.0, 440.0]), 
    p_d=7e-3, 
    p_o=2e-3,
    f_filter=3e3,
    Q_filter=2.0)

fs = 48000.0
l = 4
t   = jnp.arange(0, l*fs) / fs
tab = psi_table(p, pts, w)          # rebuild whenever p_d or S change
eps = epsilon(t, p, tab)
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
x = x / (np.max(np.abs(x)) + 1e-12) *0.0001  # normalise to ±1

def apply_envelope(x, fs, attack=0.005, release=0.020):
    """Raised-cosine fade in/out to remove start click and end truncation."""
    x = x.copy()
    n_a, n_r = int(attack * fs), int(release * fs)

    if n_a > 0:
        x[:n_a] *= 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, n_a)))
    if n_r > 0:
        x[-n_r:] *= 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, n_r)))

    return x

x = apply_envelope(x, fs, attack = 0.015)

Audio(x, rate=fs)

# %%
# Timing a full run: table -> epsilon -> RLC
#
# JAX dispatches asynchronously, so block_until_ready is required or you time
# the queue rather than the work.  "cold" includes tracing + XLA compilation,
# which is cached per *input shape*: changing secs pays it again.  "warm" is
# what a fitting loop actually costs per step.

import time

def timed(label, fn, n=3):
    t0 = time.perf_counter(); out = fn(); jax.block_until_ready(out)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(n): out = fn(); jax.block_until_ready(out)
    warm = (time.perf_counter() - t0) / n
    print(f"  {label:12s} cold {cold:7.3f}s   warm {warm:7.4f}s")
    return out

secs = 6
t_bench = jnp.arange(0, secs * fs) / fs
print(f"{secs} s @ {fs:.0f} Hz = {t_bench.shape[0]} samples")

tab_b = timed("psi_table", lambda: psi_table(p, pts, w))
eps_b = timed("epsilon",   lambda: epsilon(t_bench, p, tab_b))
sig_b = timed("RLC",       lambda: RLC(eps_b, p, fs))

full = lambda: RLC(epsilon(t_bench, p, psi_table(p, pts, w)), p, fs)
jax.block_until_ready(full())                      # warm the cache first
t0 = time.perf_counter()
for _ in range(3): jax.block_until_ready(full())
dt = (time.perf_counter() - t0) / 3
print(f"  {'END-TO-END':12s}              warm {dt:7.4f}s   -> {secs/dt:6.1f}x realtime")

# %%
