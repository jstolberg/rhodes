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
# field at the surface, which due to the symmetry of the setting is identical
# up to a scaling factor $\gamma$, causing a magnetic flux
# $$ \Psi(\alpha) \approx \gamma \mathcal{B}_z(\alpha)^2$$
# Given an explicit expression for $S$, $\mathcal{B}_z$ can be solved numerically
# for a given point $\alpha$. The induced voltage by the pickup coil for 
# a moving tip $\alpha(t)$ is in turn given by
# $$ \epsilon = - \frac{\text{d} \Psi(\alpha)}{\text{d} t} .$$
# Assuming a simple, modal model for $\alpha(t)$ in the x-axis means
# holding $y'$ and $z'$ fixed and letting
# $$x' = \alpha_x(t) = \sum_q A_q e^{- \lambda_q t} \sin(2\pi f_q t) $$
# where $A_q$ and $\lambda_q$ is the amplitude and decay for mode $q$ 
# with frequenzy $f_q$.
# If we fix a grid on the surface $S$, and approximate $\Psi_{B_z}(\alpha)$
# as the sum across this grid, it means writing $\Psi_{B_z}$ becomes an
# easily differentiable function in $t$, enabling us to fit the full 
# model, including modes, based on voltage readings of the Rhodes output.
# 
# The model parameters are 
# - $N$ - Number of fitted modes.
# - $A_i, \lambda_i, f_i$ - Modal parameters for $i = 0 \ldots N-1$.
# - $\kappa$ - Single multiplicative factor, encompassing $\gamma$, $\sigma$ and $B_0$.
# - $S$ - The surface shape.
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
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

print("jax", jax.__version__, "devices:", jax.devices())

