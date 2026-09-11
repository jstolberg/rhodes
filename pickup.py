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
def calc_tau(vel, p):
    """Compute contact time for a given velocity (0-1)"""
    tau_0, beta = p['tau_0'], p['beta']
    return tau_0 * jnp.pow(vel, -beta)

def hammer(t, f, c, F_0, tau, eps=1e-9):
    """Tine displacement during contact, 0 <= t <= tau, for mode f.
    t: (N,) or scalar.  f, c: (M,).  Returns (N, M)."""
    t = jnp.asarray(t)[..., None]
    f = jnp.atleast_1d(f)
    c = jnp.atleast_1d(c)

    Om   = 2*jnp.pi/tau
    w    = 2*jnp.pi*f
    den  = w**2 - Om**2
    safe = jnp.where(jnp.abs(den) < eps, 1.0, den)

    x = (F_0/2)*((1 - jnp.cos(w*t))/w**2            # (N,1)*(M,) -> (N,M)
                 - (jnp.cos(Om*t) - jnp.cos(w*t))/safe) * c
    return jnp.where(jnp.abs(den) < eps, 0.0, x)    # (M,) mask over (N,M)

def hammer2free(f, c, F_0, tau, eps=1e-9):
    """Signed amplitude and phase of q(t) = A*sin(w*(t-tau) + phi) for t > tau."""
    Om  = 2*jnp.pi/tau
    w   = 2*jnp.pi*f
    den = w**2 - Om**2
    safe = jnp.where(jnp.abs(den) < eps, 1.0, den)
    K = -F_0 * Om**2 / (2 * w**2 * safe)
    A = 2*K*jnp.sin(w*tau/2) * c
    phi = w*tau/2
    return jnp.where(jnp.abs(den) < eps, 0.0, A), phi

def alpha(t, p, vel=1.0):
    """scalar t -> (3,) tip position

    Forced response under the hammer for 0 <= t <= tau, then the free damped
    oscillation with the amplitude and phase hammer2free hands over.
    """
    c, lam, f = p['c'], p['lam'], p['f_modes']
    tau = calc_tau(vel, p)
    F_0 = 2*vel/tau                 # impulse F_0*tau/2 = vel

    q_c = hammer(t, f, c, F_0, tau) # Contact displacement
    A, phi = hammer2free(f, c, F_0, tau) # Parameter handover
    q_f = A * jnp.exp(-lam * (t - tau)) * jnp.sin(2*jnp.pi * f * (t - tau) + phi) #Free displacement

    disp = jnp.sum(jnp.where(t < tau, q_c, q_f))
    x = disp + p['p_o']
    z = p['p_d']
    return jnp.array([x, 0.0, z])

def disp_bound(p, vel=1.0, n=4001):
    """max |disp| over the whole note -> a static float for psi_table.

    Free part: sum |A| (all modes peaking in phase).  Contact part: a stiff mode
    (f >> 1/tau) is pushed quasi-statically to ~F_0/w^2, far beyond the free
    amplitude it is left with, so the contact window is scanned numerically.
    Evaluated at the loudest velocity: displacement grows with vel in every
    regime (beta >= 0), so vel=1 bounds the whole range.
    """
    f, c = p['f_modes'], p['c']
    tau = calc_tau(vel, p)
    F_0 = 2*vel/tau
    A, _ = hammer2free(f, c, F_0, tau)
    tt = jnp.linspace(0.0, tau, n)
    contact = jnp.abs(hammer(tt, f, c, F_0, tau).sum(1)).max()
    return float(jnp.maximum(jnp.sum(jnp.abs(A)), contact))


# %%
# ---------- Psi lookup table ----------
# With y' = 0, alpha only ever visits the (x, z) plane, so Psi is a function of
# two scalars and the O(M) surface sum can be hoisted out of the per-sample
# loop entirely: tabulate once, then interpolate.  Tabulating z as well as x
# turns p_d into a runtime parameter rather than a rebuild.  The table is keyed
# on the alpha position itself, so an arc z' = f(x') would need no change here
# beyond alpha() returning the varying z.

@partial(jax.jit, static_argnames=('disp_max', 'po_max', 'n', 'n_pd', 'chunk'))
def psi_table(p, pts, w, disp_max, po_max, pd_range=None,
              n=4096, n_pd=1, pad=1.05, chunk=256):
    """Precompute Psi on a uniform (x, z) grid -> (x0, dx, z0, dz, V).

    V has shape (n_pd, n_x): row = z (i.e. p_d), column = x.

    disp_max and po_max are *promises* about the run: |disp| <= disp_max (its
    exact bound is sum|A|, every mode peaking in phase) and 0 <= p_o <= po_max.
    Both are static, so the grid never shifts as A moves during a fit.

    Resolution is set by n over the displacement span alone,

        dx = 2 * pad * disp_max / (n - 1)

    and the offset is covered by *adding* columns at that same dx rather than
    stretching n across a wider range -- so a generous po_max costs memory, not
    accuracy.  x runs from -pad*disp_max up past pad*disp_max + po_max (plus 2
    cells for the Catmull-Rom stencil), which p_o >= 0 makes one-sided.

    n_pd = 1 (default) pins z at p['p_d'] (pd_range is ignored) and costs
    exactly what a 1-D table costs -- psi_lookup drops the z axis at trace
    time.  p_d then may NOT be changed without a rebuild.  For a
    runtime-adjustable p_d pass n_pd >= 4 and an explicit pd_range; over the
    physical 1.5-3.5 mm voicing range n_pd = 48 holds the z error to ~1e-5
    (96 -> ~2e-7).  Keep pd_range tight: a uniform z grid stretched over a wide
    range fits the near end badly, where Psi curves hardest.  The x axis is
    never the limit here -- it stays at ~1e-10 even at p_d = 1.5 mm.

    Grid *placement* is a discretisation choice and is held constant
    (stop_gradient); the tabulated *values* stay differentiable w.r.t. the
    surface (pts, w), so the table may be rebuilt inside a fitting step.

    Specialised to y' = 0: with the surface fixed, everything in the sum but
    (x - x_j)^2 is independent of x, so with num_j = w_j * dz_j and
    c_j = y_j^2 + dz_j^2 (dz_j = z - z_j) built once per row in O(M),

        Psi(x, z) = ( sum_j num_j / ((x - x_j)^2 + c_j)^(3/2) )^2

    which is the discretised  Psi = (int_S sigma (z'-z)/||a-b||^3 db)^2  of the
    derivation above.
    """
    half = pad * disp_max                             # static
    dx   = 2.0 * half / (n - 1)                       # static
    n_x  = n + int(np.ceil(po_max / dx)) + 2          # + stencil margin
    xs   = -half + dx * jnp.arange(n_x)

    if n_pd == 1:                                     # pinned; pd_range moot
        z0, z1 = p['p_d'], p['p_d']
    elif pd_range is None:
        raise ValueError("n_pd > 1 needs an explicit pd_range")
    else:
        z0, z1 = pd_range
    z0 = jax.lax.stop_gradient(jnp.asarray(z0, xs.dtype))
    dz = (jax.lax.stop_gradient(jnp.asarray(z1, xs.dtype)) - z0) / (n_pd - 1) \
         if n_pd > 1 else jnp.asarray(1.0, xs.dtype)
    zs = z0 + dz * jnp.arange(n_pd)

    xj, yj2, zj = pts[:, 0], pts[:, 1]**2, pts[:, 2]

    m = (-n_x) % chunk                                # pad up to a chunk
    xs_pad = jnp.concatenate([xs, jnp.full(m, xs[0], xs.dtype)])

    def row(z):
        d   = z - zj                                  # (M,)
        num = w * d                                   # numerator weight
        c   = yj2 + d**2                              # y^2 + dz^2

        def psi_at(x):
            r = (x - xj)**2 + c
            # rsqrt(r^3) rather than r**1.5: pow lowers ~5x slower than mul+rsqrt
            return jnp.sum(num * jax.lax.rsqrt(r * r * r))**2

        v = jax.lax.map(jax.vmap(psi_at), xs_pad.reshape(-1, chunk))
        return v.reshape(-1)[:n_x]

    return -half, dx, z0, dz, jax.lax.map(row, zs)


def _catmull(p0, p1, p2, p3, u):
    """Cubic through p1 (u=0) and p2 (u=1), tangents from p0 and p3."""
    return 0.5 * (2.0*p1
                  + (-p0 + p2) * u
                  + (2.0*p0 - 5.0*p1 + 4.0*p2 - p3) * u**2
                  + (-p0 + 3.0*p1 - 3.0*p2 + p3) * u**3)


def _cell(s, n):
    """grid coordinate -> (base index with a valid 4-point stencil, fraction)"""
    s = jnp.clip(s, 0.0, n - 1.0)
    i = jnp.clip(jnp.floor(s).astype(int), 1, n - 3)   # integer -> no tangent
    return i, s - i                                    # so du/ds = 1


def psi_lookup(a, table):
    """(..., 3) tip positions -> (...) Psi, read off the table.

    Catmull-Rom, so the result is C1 in both axes: unlike jnp.interp, its
    derivative is continuous, which is what epsilon = -dPsi/dt needs -- in z as
    well as x, since a moving p_d contributes dPsi/dz * z_dot to the voltage.

    With a single z row the z axis is dropped entirely (4 gathers, not 16).
    V.shape is static, so that costs nothing at runtime.
    """
    x0, dx, z0, dz, V = table
    n_z, n_x = V.shape

    ix, ux = _cell((a[..., 0] - x0) / dx, n_x)

    if n_z == 1:                                       # p_d pinned, no z axis
        v = V[0]
        return _catmull(v[ix-1], v[ix], v[ix+1], v[ix+2], ux)

    iz, uz = _cell((a[..., 2] - z0) / dz, n_z)
    rows = [_catmull(V[iz+k, ix-1], V[iz+k, ix], V[iz+k, ix+1], V[iz+k, ix+2], ux)
            for k in (-1, 0, 1, 2)]
    return _catmull(*rows, uz)


@jax.jit
def epsilon(t, p, table, vel=1.0):
    """time samples -> induced voltage -dPsi/dt, via the Psi table.

    Psi is scalar->scalar in t, so forward mode costs ~2x a primal eval at O(1)
    memory.  With the O(M) surface sum hoisted into the table there is nothing
    left to chunk, so the whole signal goes through one vmap.
    """
    psi_of_t = lambda tt: psi_lookup(alpha(tt, p, vel), table)
    dpsi = lambda tt: jax.jvp(psi_of_t, (tt,), (jnp.ones_like(tt),))[1]
    return -jax.vmap(dpsi)(t)

# Example
p = dict(
    c=jnp.array([0.005, 1.0]) * 3.06, # per-mode gain; ~1 mm total at vel=1
    lam=jnp.array([0.1, 1.0]),
    f_modes=jnp.array([60.0, 440.0]),
    tau_0=1e-3,                       # contact time at vel=1
    beta=0.2,                         # tau ~ vel^-beta: 1.6 ms at vel=0.1
    p_d=2.5e-3,
    p_o=1.2e-3,
    f_filter=3e3,
    Q_filter=2.0)

fs = 48000.0
l = 4
t   = jnp.arange(0, l*fs) / fs

# Promises the table is built against: p_o may roam [0, po_max] with no rebuild,
# disp_max is the reach of the tine over contact and free decay (see disp_bound).
disp_max = disp_bound(p)
po_max   = 4e-3
pd_range = (1.5e-3, 3.5e-3)   # physical voicing range; pass with n_pd>=4 to
                              # make p_d adjustable too (n_pd=48 -> ~1e-5)

tab = psi_table(p, pts, w, disp_max, po_max)   # n_pd=1: p_d pinned, rebuild to change it
eps = epsilon(t, p, tab, vel=0.5)

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
# Fixed gain, referenced to the loudest velocity, so different vel are audible
# as different loudness rather than being normalised away.  epsilon is in
# arbitrary units (kappa absorbed) and peaks at ~4e3 for vel=1 with these p.
# Audio() normalises by default and would undo this -- hence normalize=False.
gain = 0.8 / float(jnp.abs(epsilon(t, p, tab, vel=1.0)).max())

x = np.asarray(eps, dtype=np.float64) * gain
Audio(x, rate=fs, normalize=False)

# %%
