# Reconstructing Nuclear Fuel Cycle Operations with Nuclear Archaeology
This repository contains the code used in _Reconstructing Nuclear Fuel Cycle
Operations with Nuclear Archaeology_ by Max Schalz and Malte Göttsche, presented
at the INMM & ESARDA Joint Annual Meeting in Vienna, 2023.

## Overview
- `simulation/`:
  This directory contains the driver files to start the forward and
  reconstruction runs, as well as the definitions of the parameter
  distributions, the template Cyclus input file and data used in the case study.
- `analysis/`:
  This directory mostly contains scripts needed to create the figures used in
  the presentation and/or the paper.

## Software Requirements
This repository uses [Bicyclus](https://github.com/Nuclear-Verification-and-Disarmament/bicyclus).

Additionally, the following Python3 modules are needed:
[Scipy](https://docs.scipy.org/doc/scipy/index.html),
[Pandas](https://pandas.pydata.org/),
[matplotlib](https://matplotlib.org/),
[seaborn](https://seaborn.pydata.org/),
[Arviz](https://python.arviz.org/en/latest/index.html),
[NumPy](https://numpy.org/doc/stable/index.html),
[Aesara](https://aesara.readthedocs.io/en/lates/) (only needed for the inference
runs).

If you install Bicyclus using the `plotting` option (`pip3 install .[plotting]`),
then all of these packages will be installed anyway.
