# rhodes
Physical modelling sound synthesis targeting the Rhodes piano

## Physical Model

## Tine
Differential equation for shear beam from Pfeifle 2017,
$$
\rho\mathbf{u}_{tt} + [EI\mathbf{u}_{xx}]_{xx} - EA\frac{1}{2}\mathbf{u}_{xx} \cdot K(\mathbf{u}) - \kappa\mathbf{u}_{2x2t} - F(\mathbf{u}^{V}[x], t) = 0 \tag{2}
$$

## Relevant Literature
- [Real-time Physical Model of A Wurlitzer and Rhodes Electric Piano](https://dafx17.eca.ed.ac.uk/papers/DAFx17_paper_79.pdf)
- [The Rhodes electric piano: Analysis and simulation of the inharmonic overtones](https://pubs.aip.org/asa/jasa/article/148/5/3052/631688/The-Rhodes-electric-piano-Analysis-and-simulation)
- [Rhodes Service Manual](https://dn760106.eu.archive.org/0/items/fender_Rhodes_Keyboard_Instruments_Service_Manual/Rhodes_Keyboard_Instruments_Service_Manual_text.pdf)
- M. Muenster and F. Pfeifle - Non-Linear Behaviour in Sound Production of the Rhodes Piano
- S. Bilbao - Numerical Sound Synthesis; Chapter 7
