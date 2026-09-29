# rhodes
Physical modelling sound synthesis targeting the Rhodes piano

## Pipeline
One script per step, run in order from the repo root. Each step reads the results of the
steps before it from `results/`. `pipeline.py` runs all steps, shows the results and renders
the model after each step.

| Step | Script | Estimates |
| --- | --- | --- |
| 1. Fundamental frequency and decay | `python step1_fundamental.py` | $f_0$, $\lambda_0$ |
| 2. Pickup and free oscillation | `python step2_pickup.py real 1` | $p_d$, $p_o$, $\kappa$, $A_0$, $\sigma_0$ |
| 3. Hammer parameters | `python step3_hammer.py` | $\tau_0$, $\beta$, $c_0$ |
| 4. Inharmonic modes | `python step4_modes.py` | $f_n$, $\sigma_n$, $c_n$ |

The model itself is in `model.py`; `python synth.py <step> [notes] [--dyn ...]` renders it
after a step. `python results.py` computes the numbers and figures for the paper from
`results/` (writes `figures/` and `results_numbers.md`). The samples go in `Samples/` as
`{note}-{dyn}.wav`.

## Rhodes measurements
[https://www.fenderrhodes.com/org/manual/ch6.html](https://www.fenderrhodes.com/org/manual/ch6.html)

## Relevant Literature
- [Real-time Physical Model of A Wurlitzer and Rhodes Electric Piano](https://dafx17.eca.ed.ac.uk/papers/DAFx17_paper_79.pdf)
- [The Rhodes electric piano: Analysis and simulation of the inharmonic overtones](https://pubs.aip.org/asa/jasa/article/148/5/3052/631688/The-Rhodes-electric-piano-Analysis-and-simulation)
- [Rhodes Service Manual](https://dn760106.eu.archive.org/0/items/fender_Rhodes_Keyboard_Instruments_Service_Manual/Rhodes_Keyboard_Instruments_Service_Manual_text.pdf)
- M. Muenster and F. Pfeifle - Non-Linear Behaviour in Sound Production of the Rhodes Piano
- S. Bilbao - Numerical Sound Synthesis; Chapter 7
- [Modeling the magnetic pickup of an electric guitar](https://users.manchester.edu/facstaff/gwclark/PHYS301/AJP%20Articles/AJP%20Electric%20Guitar%20pickup.pdf)
