# %% [markdown]
# # The Rhodes model: pickup, hammer and tine
#
# The model has three parts, and this file holds all three.
#
# ## Pickup
#
# The pickup is modelled as a magnetically charged surface $S$ with a circular
# cross-section at $z = 0$; for $z > 0$ its height is a continuous function of $x$
# and $y$.  The simplest example is a guitar pickup, $z(x, y) = 0$.  The tip of the
# tine is a point $\alpha = (x', y', z')$.  One point $\beta = (x, y, z)$ of the
# surface contributes to the field at the tip along $z$
# $$ B_z(\beta) = B_0 \frac{z' - z}{\|\alpha - \beta \|^3},$$
# and the whole surface, with charge density $\sigma$,
# $$ \mathcal{B}_z(\alpha) = \int_S \sigma B_z(\beta)\, \text{d}\beta .$$
# This magnetises the tip, which in turn acts back on the surface; by symmetry that
# is the same up to a factor $\gamma$, giving the flux through the coil
# $$ \Psi(\alpha) \approx \gamma\, \mathcal{B}_z(\alpha)^2 .$$
# The coil voltage for a moving tip $\alpha(t)$ is
# $$ \epsilon = - \frac{\text{d} \Psi(\alpha)}{\text{d} t} .$$
#
# In practice the surface is a grid of points (`make_surface`), $\Psi$ is the sum
# over the grid, and because it is needed at millions of time samples it is computed
# once on a grid of tip positions and interpolated (`psi_table`, `psi_lookup`).
# All constants ($\gamma$, $\sigma$, $B_0$) are absorbed into one overall gain
# $\kappa$ that the fits determine.
#
# ## Hammer
#
# The hammer pushes the tine with a smooth pulse of length $\tau$,
# $$ F(t) = F_0 \sin^2(\pi t / \tau), \qquad 0 \le t \le \tau, \qquad
#    F_0 = 2v/\tau \;\Rightarrow\; \int F\, \text{d}t = v, $$
# a raised-cosine (Hann) pulse whose impulse equals the velocity $v \in (0, 1]$.  A
# faster strike is also shorter: $\tau = \tau_0\, v^{-\beta}$ (`calc_tau`).  The
# force is prescribed: the tine does not act back on the hammer.
#
# ## Tine
#
# The tip moves only along $x$ ($y' = 0$, $z' = p_d$ fixed), as a sum of independent
# modes $q$ with frequency $f_q$, excitation coefficient $c_q$ and decay $\lambda_q$.
# During contact each mode is an undamped oscillator driven by $c_q F(t)$
# (`hammer`); afterwards it swings freely and decays,
# $$ x'(t) = p_o + \sum_q A_q e^{-\lambda_q (t-\tau)} \sin\big(2\pi f_q (t-\tau) + \varphi_q\big),$$
# with the amplitude $A_q$ and phase $\varphi_q$ the contact hands over
# (`hammer2free`).  $p_o$ is the offset of the tine from the pickup centre, $p_d$
# its distance.
#
# ## Parameters
#
# | Symbol | Meaning | Key in the dict `p` |
# | --- | --- | --- |
# | $f_q, c_q, \lambda_q$ | frequency, excitation, decay of each mode | `f_modes`, `c`, `lam` |
# | $\tau_0, \beta$ | contact time at $v = 1$, its velocity exponent | `tau_0`, `beta` |
# | $p_o, p_d$ | offset and distance of the tine w.r.t. the pickup | `p_o`, `p_d` |
# | $S$ | surface shape (`z_trapz`: plateau half width $a$, slope $m$, radius) | via `make_surface` |
# | $\kappa$ | overall gain (applied outside this file) | - |
#
# **Not modelled:** the coil and its circuit form a resonant low-pass (RLC)
# filter, which is not applied to $\epsilon$.

# %%
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

# The example cells below are guarded by __name__ == "__main__", so `from model import
# ...` only gets the functions.  Jupyter / VS Code cells run as __main__, so they still
# execute interactively.


# %%
# =====================================================================================
# Pickup surface
# =====================================================================================

def z_trapz(X, Y, a=0.5e-3, m=0.3, z0=0.0):
    """Rhodes pickup: flat plateau for |x| <= a, then falling off with slope m;
    constant along y."""
    return z0 - m * jnp.maximum(jnp.abs(X) - a, 0.0)


def z_flat(X, Y, z0=0.0):
    """Guitar pickup: a flat disc."""
    return jnp.full_like(X, z0)


def disc_mask(X, Y, r_max):
    """True inside the circle of radius r_max."""
    return (X**2 + Y**2) <= r_max**2


def make_surface(z_fn, N=128, r_max=2e-3, mask_fn=disc_mask):
    """Grid of N x N points on the surface z = z_fn(x, y) over the square |x|, |y| <= r_max.

    Returns pts (N*N, 3) and the area of each point's cell (N*N,).  The area includes
    the slope of the surface (a tilted cell is larger than its footprint) and is 0
    outside the mask.
    """
    e = jnp.linspace(-r_max, r_max, N + 1)              # cell edges
    c = 0.5 * (e[:-1] + e[1:])                          # cell centres
    d = c[1] - c[0]                                     # cell width
    X, Y = jnp.meshgrid(c, c, indexing='ij')
    Z = z_fn(X, Y)

    # area of a tilted cell = footprint * sqrt(1 + (dz/dx)^2 + (dz/dy)^2)
    grad_z = jax.grad(z_fn, argnums=(0, 1))
    gx, gy = jax.vmap(grad_z)(X.ravel(), Y.ravel())
    metric = jnp.sqrt(1.0 + gx**2 + gy**2).reshape(X.shape)
    area = d * d * metric * mask_fn(X, Y, r_max)

    pts = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return pts, area.ravel()


def drop_masked(pts, area):
    """Remove the points outside the mask (area 0, ~21 % of the grid for a disc).

    They would cost time in every sum but add nothing.  Call it outside jit (the result
    size depends on the data) and after plot_surface, which needs the full grid.
    """
    keep = np.asarray(area) > 0
    return pts[keep], area[keep]


def plot_surface(pts, area, scale=1e3, unit='mm', equal_z=True,
                 elev=22, azim=-60, cmap='viridis', ax=None):
    """3-D plot of the surface from make_surface() (full grid, before drop_masked)."""
    N = int(round(np.sqrt(pts.shape[0])))
    P = np.asarray(pts).reshape(N, N, 3)
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    Z = np.where(np.asarray(area).reshape(N, N) > 0, Z, np.nan)     # hide masked points

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X * scale, Y * scale, Z * scale,
                    cmap=cmap, linewidth=0, antialiased=True, rstride=2, cstride=2)
    ax.set_xlabel(f'x ({unit})'); ax.set_ylabel(f'y ({unit})'); ax.set_zlabel(f'z ({unit})')
    ax.set_box_aspect((1, 1, 1))

    span = max(np.ptp(X), np.ptp(Y)) * scale / 2
    ax.set_xlim(-span, span); ax.set_ylim(-span, span)
    if equal_z:
        zc = np.nanmean(Z) * scale
        ax.set_zlim(zc - span, zc + span)
    ax.view_init(elev=elev, azim=azim)
    return ax


if __name__ == "__main__":
    pts, w = make_surface(partial(z_trapz, a=0.5e-3, m=0.4), r_max=3e-3)
    plot_surface(pts, w)
    pts, w = drop_masked(pts, w)    # idempotent, so re-running the cell is safe


# %%
# =====================================================================================
# Hammer and tine motion
# =====================================================================================

def calc_tau(vel, p):
    """Contact time for velocity vel (0-1): tau = tau_0 vel^-beta."""
    tau_0, beta = p['tau_0'], p['beta']
    return tau_0 * jnp.pow(vel, -beta)


def _pulse(f, vel, tau, eps):
    """Quantities shared by hammer and hammer2free: peak force F_0, pulse angular
    frequency Om = 2 pi / tau, mode angular frequency w, w^2 - Om^2 and a version of it
    that is never 0 (exact resonance w = Om would divide by 0)."""
    F_0  = 2*vel/tau
    Om   = 2*jnp.pi/tau
    w    = 2*jnp.pi*f
    den  = w**2 - Om**2
    safe = jnp.where(jnp.abs(den) < eps, 1.0, den)
    return F_0, Om, w, den, safe


def hammer(t, f, c, vel, tau, eps=1e-9):
    """Displacement of each mode during contact, 0 <= t <= tau.

    Force F(t) = F_0 sin^2(pi t / tau) = F_0/2 (1 - cos(Om t)), with F_0 = 2 vel / tau so
    that the impulse F_0 tau / 2 is vel.  Each mode is an undamped oscillator
    x'' + w^2 x = c F(t) starting at rest; this is its exact solution.

    t: scalar or (N,).  f, c: (M,).  Returns (M,) for scalar t, else (N, M).
    At exact resonance (w = Om) it returns 0; the true value is finite, but that case
    does not occur in practice.
    """
    t = jnp.asarray(t)[..., None]
    f = jnp.atleast_1d(f)
    c = jnp.atleast_1d(c)
    F_0, Om, w, den, safe = _pulse(f, vel, tau, eps)

    x = (F_0/2)*((1 - jnp.cos(w*t))/w**2                # response to the constant part
                 - (jnp.cos(Om*t) - jnp.cos(w*t))/safe) * c   # ... and to the cosine part
    return jnp.where(jnp.abs(den) < eps, 0.0, x)


def hammer2free(f, c, vel, tau, eps=1e-9):
    """Amplitude A (with sign) and phase phi of the free oscillation after contact,
    q(t) = A sin(w (t - tau) + phi) for t > tau.

    Chosen so that position and velocity at t = tau equal those of hammer(): with
    K = -F_0 Om^2 / (2 w^2 (w^2 - Om^2)), hammer gives x(tau) = K (1 - cos w tau) and
    x'(tau) = K w sin w tau, which is A sin(phi), A w cos(phi) for the A, phi below.
    """
    F_0, Om, w, den, safe = _pulse(f, vel, tau, eps)
    K = -F_0 * Om**2 / (2 * w**2 * safe)
    A = 2*K*jnp.sin(w*tau/2) * c
    phi = w*tau/2
    return jnp.where(jnp.abs(den) < eps, 0.0, A), phi


def alpha(t, p, vel=1.0):
    """Tip position (x, 0, p_d) at scalar time t (t = 0: the hammer touches).

    During contact (t < tau) the modes follow hammer(); afterwards each swings freely
    with the amplitude and phase hammer2free hands over and decays with lam.
    """
    c, lam, f = p['c'], p['lam'], p['f_modes']
    tau = calc_tau(vel, p)

    q_c = hammer(t, f, c, vel, tau)                                     # during contact
    A, phi = hammer2free(f, c, vel, tau)
    q_f = A * jnp.exp(-lam * (t - tau)) * jnp.sin(2*jnp.pi * f * (t - tau) + phi)   # after

    disp = jnp.sum(jnp.where(t < tau, q_c, q_f))                        # sum of all modes
    return jnp.array([disp + p['p_o'], 0.0, p['p_d']])


def disp_bound(p, vel=1.0, n=4001):
    """Largest |displacement| of the whole note (a plain float), to size psi_table.

    After contact at most sum |A| (all modes peaking together).  During contact a stiff
    mode (f >> 1/tau) is pushed much further (~F_0 / w^2) than the amplitude it keeps,
    so the contact is scanned at n points.  Taken at the loudest velocity (vel = 1),
    which moves the tine furthest.
    """
    f, c = p['f_modes'], p['c']
    tau = calc_tau(vel, p)
    A, _ = hammer2free(f, c, vel, tau)
    tt = jnp.linspace(0.0, tau, n)
    contact = jnp.abs(hammer(tt, f, c, vel, tau).sum(1)).max()
    return float(jnp.maximum(jnp.sum(jnp.abs(A)), contact))


# %%
# =====================================================================================
# Psi lookup table
# =====================================================================================
# The tip only ever moves in the (x, z) plane (y' = 0), so Psi is a function of two
# numbers.  Summing over all surface points at every time sample would be far too slow,
# so Psi is computed once on a grid of (x, z) and interpolated.  Because z is a grid
# axis too, p_d can change without rebuilding the table.

@partial(jax.jit, static_argnames=('disp_max', 'po_max', 'n', 'n_pd', 'chunk'))
def psi_table(p, pts, w, disp_max, po_max, pd_range=None,
              n=4096, n_pd=1, pad=1.05, chunk=256):
    """Psi on a uniform (x, z) grid.  Returns (x0, dx, z0, dz, V), V of shape (n_pd, n_x):
    row = z (i.e. p_d), column = x.

    x grid.  disp_max and po_max are promises about how the table will be used:
    |displacement| <= disp_max (see disp_bound) and 0 <= p_o <= po_max.  The spacing is
    set by the displacement alone, dx = 2 pad disp_max / (n - 1); the p_o range is covered
    by adding columns at the same spacing (a large po_max costs memory, not accuracy).
    x runs from -pad disp_max to past pad disp_max + po_max (+ 2 cells for the
    interpolation stencil).  With po_max = None the grid is n points centred on p['p_o']:
    cheaper, but p_o can then not change without a rebuild.

    z grid.  n_pd = 1: one row at p['p_d'] (pd_range ignored); p_d then can not change
    without a rebuild, but the table stays differentiable w.r.t. p_d.  n_pd >= 4 with
    pd_range = (lo, hi): rows from lo to hi, p_d free at run time.  Keep the range tight;
    Psi curves most at small distances.  (Over 1.5-3.5 mm, n_pd = 48 gives ~1e-5 error,
    96 ~2e-7; the x axis is never the limit.)

    The grid positions are held fixed for differentiation (stop_gradient); the values
    stay differentiable w.r.t. the surface (pts, w), so the table may be rebuilt inside
    a fit.

    Formula.  With y' = 0 and the surface fixed, only (x - x_j)^2 depends on x:
        Psi(x, z) = ( sum_j w_j (z - z_j) / ((x - x_j)^2 + y_j^2 + (z - z_j)^2)^(3/2) )^2,
    the discretised Psi = B_z^2 of the header.
    """
    # x grid
    half = pad * disp_max
    dx   = 2.0 * half / (n - 1)
    if po_max is None:                                  # p_o fixed: centre on it
        n_x = n
        x0  = jax.lax.stop_gradient(jnp.asarray(p['p_o'], float)) - half
    else:                                               # p_o free in [0, po_max]
        n_x = n + int(np.ceil(po_max / dx)) + 2
        x0  = -half
    xs = x0 + dx * jnp.arange(n_x)

    # z grid
    if n_pd == 1:                                       # one row at p_d, keeps its gradient
        z0 = jnp.asarray(p['p_d'], xs.dtype)
        dz = jnp.asarray(1.0, xs.dtype)
    elif pd_range is None:
        raise ValueError("n_pd > 1 needs an explicit pd_range")
    else:
        z0 = jax.lax.stop_gradient(jnp.asarray(pd_range[0], xs.dtype))
        dz = (jax.lax.stop_gradient(jnp.asarray(pd_range[1], xs.dtype)) - z0) / (n_pd - 1)
    zs = z0 + dz * jnp.arange(n_pd)

    xj, yj2, zj = pts[:, 0], pts[:, 1]**2, pts[:, 2]

    # x values are processed in chunks of `chunk` to limit memory; pad up to a multiple
    m = (-n_x) % chunk
    xs_pad = jnp.concatenate([xs, jnp.full(m, xs[0], xs.dtype)])

    def row(z):                                         # one z: Psi at all x
        d   = z - zj                                    # height above each surface point
        num = w * d
        c   = yj2 + d**2                                # the part of the distance^2 without x

        def psi_at(x):
            r = (x - xj)**2 + c                         # distance^2 to every surface point
            return jnp.sum(num * jax.lax.rsqrt(r * r * r))**2   # rsqrt(r^3) = r^-3/2, fast

        v = jax.lax.map(jax.vmap(psi_at), xs_pad.reshape(-1, chunk))
        return v.reshape(-1)[:n_x]

    return x0, dx, z0, dz, jax.lax.map(row, zs)


def _catmull(p0, p1, p2, p3, u):
    """Catmull-Rom cubic between p1 (u = 0) and p2 (u = 1), slopes from p0 and p3."""
    return 0.5 * (2.0*p1
                  + (-p0 + p2) * u
                  + (2.0*p0 - 5.0*p1 + 4.0*p2 - p3) * u**2
                  + (-p0 + 3.0*p1 - 3.0*p2 + p3) * u**3)


def _cell(s, n):
    """Grid coordinate s -> (index i with neighbours i-1 .. i+2 inside the grid, fraction)."""
    s = jnp.clip(s, 0.0, n - 1.0)
    i = jnp.clip(jnp.floor(s).astype(int), 1, n - 3)
    return i, s - i


def psi_lookup(a, table):
    """Tip positions a (..., 3) -> Psi (...), interpolated from the table.

    Catmull-Rom instead of linear interpolation, so the slope is continuous as well:
    epsilon = -dPsi/dt needs a smooth derivative (in z too, where p_d moves).
    With a single z row only x is interpolated (4 values instead of 16).
    """
    x0, dx, z0, dz, V = table
    n_z, n_x = V.shape

    ix, ux = _cell((a[..., 0] - x0) / dx, n_x)
    if n_z == 1:
        v = V[0]
        return _catmull(v[ix-1], v[ix], v[ix+1], v[ix+2], ux)

    # interpolate along x in the four neighbouring rows, then along z
    iz, uz = _cell((a[..., 2] - z0) / dz, n_z)
    rows = [_catmull(V[iz+k, ix-1], V[iz+k, ix], V[iz+k, ix+1], V[iz+k, ix+2], ux)
            for k in (-1, 0, 1, 2)]
    return _catmull(*rows, uz)


# %%
# =====================================================================================
# Pickup voltage
# =====================================================================================

@jax.jit
def epsilon(t, p, table, vel=1.0):
    """Coil voltage -dPsi/dt at the time samples t (without kappa).

    dPsi/dt is computed exactly by forward-mode differentiation (jvp) of
    t -> Psi(alpha(t)), about twice the cost of Psi itself.
    """
    psi_of_t = lambda tt: psi_lookup(alpha(tt, p, vel), table)
    dpsi = lambda tt: jax.jvp(psi_of_t, (tt,), (jnp.ones_like(tt),))[1]
    return -jax.vmap(dpsi)(t)


# Example: two modes, 4 s at vel = 0.5
if __name__ == "__main__":
    p = dict(
        c=jnp.array([0.005, 1.0]) * 3.06, # per-mode gain; ~1 mm total at vel = 1
        lam=jnp.array([0.1, 1.0]),
        f_modes=jnp.array([60.0, 440.0]),
        tau_0=1e-3,                       # contact time at vel = 1
        beta=0.2,                         # tau ~ vel^-beta: 1.6 ms at vel = 0.1
        p_d=2.5e-3,
        p_o=1.2e-3)

    fs = 48000.0
    l = 4
    t = jnp.arange(0, l*fs) / fs

    disp_max = disp_bound(p)              # how far the tine moves
    po_max   = 4e-3                       # p_o may be anywhere in [0, po_max]

    tab = psi_table(p, pts, w, disp_max, po_max)        # n_pd = 1: p_d fixed
    # tab = psi_table(p, pts, w, disp_max, None)        # p_o fixed too: smallest table
    # tab = psi_table(p, pts, w, disp_max, po_max, pd_range=(1.5e-3, 3.5e-3), n_pd=48)  # p_d free
    eps = epsilon(t, p, tab, vel=0.5)

# %%
# Spectrogram of the example
if __name__ == "__main__":
    from scipy import signal

    eps_np = np.asarray(eps, dtype=np.float64)
    f, t_spec, Sxx = signal.spectrogram(eps_np, fs=fs, nperseg=2048, noverlap=1536,
                                        window='hann', scaling='spectrum')
    Sdb = 10*np.log10(Sxx + 1e-20)

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t_spec, f, Sdb, vmin=Sdb.max()-80, vmax=Sdb.max(),   # 80 dB range
                   shading='gouraud', cmap='magma')
    plt.ylim(0, 8000)
    plt.xlabel('time (s)'); plt.ylabel('frequency (Hz)')
    plt.colorbar(label='dB')
    plt.tight_layout(); plt.show()

# %%
# Listen to the example.  The gain is fixed by the loudest velocity, so different vel
# sound differently loud; normalize=False keeps Audio() from undoing that.
if __name__ == "__main__":
    from IPython.display import Audio, display

    gain = 0.8 / float(jnp.abs(epsilon(t, p, tab, vel=1.0)).max())
    x = np.asarray(eps, dtype=np.float64) * gain
    display(Audio(x, rate=fs, normalize=False))
