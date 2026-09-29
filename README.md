# rhodes
A physical model of the Rhodes electric piano, with its parameters estimated from
recordings. The model has three parts: the tine as a sum of decaying modes, the hammer as
a force pulse, and the magnetic pickup. The parameters are estimated in four steps, each
of which holds the results of the steps before it fixed.

## Getting started
```
uv sync
uv run python pipeline.py
```
`pipeline.py` goes through all four steps: it shows each step's results and plays the model
next to the recording. It uses the saved results in `results/` and the six recordings in
`examples/`, so it runs without the sample pack. Set `RERUN` at its top to run a step
again.

To rerun the steps you need the recordings (Matt's Fender Rhodes on
[Pianobook](https://www.pianobook.co.uk/packs/matts-fender-rhodes/)) in `Samples/` as
`{note}-{dyn}.wav`, with dynamics `p`, `mp`, `mf`, `f`. The note names are the sample
library's, one octave below the usual ones: `E0` is the lowest key (41.2 Hz).

## The steps
Run from the repo root, in order. Each step writes its result to `results/`. The steps,
`model.py`, `pipeline.py` and `results.py` are split into `# %%` cells, so they can also
be run cell by cell in an interactive window (VS Code, Jupyter).

| Step | Script | Estimates |
| --- | --- | --- |
| 1. Fundamental frequency and decay | `python step1_fundamental.py` | $f_0$, $\lambda_0$ |
| 2. Pickup and free oscillation | `python step2_pickup.py real 1` | $p_d$, $p_o$, $\kappa$, $A_0$, $\sigma_0$ |
| 3. Hammer | `python step3_hammer.py` | $\tau_0$, $\beta$, $c_0$, velocities |
| 4. Inharmonic modes | `python step4_modes.py` | $f_n$, $\sigma_n$, $A_n/A_0$ |


## Other files
- `model.py`: the model itself (tine, hammer, pickup)
- `synth.py`: renders a note after any step, e.g. `python synth.py 4 A#3 --dyn p f`
- `results.py`: the numbers and figures for the paper (writes `figures/`, `results_numbers.md`)
- `explore/`: analyses that are not part of the pipeline

## Relevant Literature
- [Real-time Physical Model of A Wurlitzer and Rhodes Electric Piano](https://dafx17.eca.ed.ac.uk/papers/DAFx17_paper_79.pdf)
- [The Rhodes electric piano: Analysis and simulation of the inharmonic overtones](https://pubs.aip.org/asa/jasa/article/148/5/3052/631688/The-Rhodes-electric-piano-Analysis-and-simulation)
- [Rhodes Service Manual](https://dn760106.eu.archive.org/0/items/fender_Rhodes_Keyboard_Instruments_Service_Manual/Rhodes_Keyboard_Instruments_Service_Manual_text.pdf)
- M. Muenster and F. Pfeifle - Non-Linear Behaviour in Sound Production of the Rhodes Piano
- S. Bilbao - Numerical Sound Synthesis; Chapter 7
- [Modeling the magnetic pickup of an electric guitar](https://users.manchester.edu/facstaff/gwclark/PHYS301/AJP%20Articles/AJP%20Electric%20Guitar%20pickup.pdf)
