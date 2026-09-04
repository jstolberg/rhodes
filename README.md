# rhodes
Physical modelling sound synthesis targeting the Rhodes piano

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
