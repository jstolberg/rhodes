# rhodes
Physical modelling sound synthesis targeting the Rhodes piano

## Physical Model

# Pickup for the Rhodes model

Pickup is modelled as magnetically charged surface $S$, assumed 
to have circular cross-section at z = 0, and for z > 0, z is given 
by a continuous function in x and y. The most simple examble being 
a typical guitar pickup with $z$ identically given by

$$z (x, y) = 0$$

Modelling the tip of the Rhodes tine as a point $\alpha = (x', y', z')$, 
we consider the magnetic effect at this point induced by a single point
on the surface $\beta = (x,y,z) \in S$ along the z-axis,

$$ B_z(\beta) = B_0 \frac{z' - z}{\|\alpha - \beta \|^3}.$$

The magnetic field induced by the full surface at the tine tip $\alpha$,
is then given by 

$$ \mathcal{B}_z(\alpha) = \int_S \sigma B_z(\beta) \text{d}\beta $$

where $\sigma$ is the magnetic charge density across $S$. This causes a 
proportinal magnetisation of the tip of the tine, which in turn affects the magnetic
field at the surface $S$, which due to the symmetry of the setting is identical
up to a scaling factor $\gamma$, causing a magnetic flux

$$ \Psi(\alpha) \approx \gamma \mathcal{B}_z(\alpha)^2$$

Given an explicit expression for $S$, $\mathcal{B}_z$ can be solved numerically
for a given point $\alpha$. The induced voltage by the pickup coil for 
a moving tip $\alpha(t)$ is in turn given by

$$ \epsilon = - \frac{\text{d} \Psi(\alpha)}{\text{d} t} .$$

Assuming a simple, modal model for $\alpha(t)$ in the x-axis means
holding $y'$ and $z'$ fixed and letting

$$x' = \alpha_x(t) = \sum_q A_q e^{- \lambda_q t} \sin(2\pi f_q t) $$

where $A_q$ and $\lambda_q$ is the amplitude and decay for mode $q$ 
with frequenzy $f_q$.
If we fix a grid on the surface $S$, and approximate $\Psi(\alpha)$
as the sum across this grid, it means writing $\Psi$ becomes an
easily differentiable function in $t$, enabling us to fit the full 
model, including modes, based on voltage outputs of the Rhodes output.

The model parameters are 
- $N$ - Number of fitted modes.
- $A_i, \lambda_i, f_i$ - Modal parameters for $i = 0 \ldots N-1$.
- $\kappa$ - Single multiplicative factor, encompassing $\gamma$, $\sigma$ and $B_0$.
- $S$ - The surface shape.

As $\Psi$ depends only on the position $\alpha$, a lookup table 
with precomputed, numerical solutions to the integral can be computed
for positions of $\alpha$, once parameters have been fitted, enabling
real-time synthesis of the fitted instrument.

**Missing elements:**
- Model $\alpha$ on an arc $z' = f(x')$ instead of fixed $z'$.
- The coil implements a RLC circuit, implicitly implying a resonant,
low-pass filter, which could be modelled too.

## Tine
Differential equation for shear beam from Pfeifle 2017,

$$
\rho \mathbf{u}_{tt} + [EI\mathbf{u}_{xx}]_{xx} - EA\frac{1}{2}\mathbf{u}_{xx} \cdot K(\mathbf{u}) - \kappa \mathbf{u}_{2x2t} - F(\mathbf{u}^{V}[x], t) = 0
$$

## Tentative Plan
- Model pickup from Pfeile (2017) (NN or Lookup table or both)
- Reverse engineer modes (frequency, amplitude, decay) from sample pack, using pickup model, and compare with Gabrielli (2020)
- Model excitation signal based on idealized clamped bar (using measurements? See below). Calculate for each key.
- Rhodes real-time synthesis: Add modal synthesis with excitation signal, and feed through pickup model.

## Rhodes measurements
[https://www.fenderrhodes.com/org/manual/ch6.html](https://www.fenderrhodes.com/org/manual/ch6.html)

## Relevant Literature
- [Real-time Physical Model of A Wurlitzer and Rhodes Electric Piano](https://dafx17.eca.ed.ac.uk/papers/DAFx17_paper_79.pdf)
- [The Rhodes electric piano: Analysis and simulation of the inharmonic overtones](https://pubs.aip.org/asa/jasa/article/148/5/3052/631688/The-Rhodes-electric-piano-Analysis-and-simulation)
- [Rhodes Service Manual](https://dn760106.eu.archive.org/0/items/fender_Rhodes_Keyboard_Instruments_Service_Manual/Rhodes_Keyboard_Instruments_Service_Manual_text.pdf)
- M. Muenster and F. Pfeifle - Non-Linear Behaviour in Sound Production of the Rhodes Piano
- S. Bilbao - Numerical Sound Synthesis; Chapter 7
- [Modeling the magnetic pickup of an electric guitar](https://users.manchester.edu/facstaff/gwclark/PHYS301/AJP%20Articles/AJP%20Electric%20Guitar%20pickup.pdf)
