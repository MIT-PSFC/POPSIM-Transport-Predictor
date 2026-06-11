# POPSIM-Transport-Predictor

Study that investigates using POPSIM for full-shot profile prediction.

# Installation

This repo should be able to completely reproduce all plots in the ICDDPS presentation, and all plots in the upcoming paper provided the person trying to do the reproduction has proper data access.

A C-Mod dataset is provided, and the MAST dataset can be created using the provided scripts.

The number of people who have access to C-Mod, DIII-D, and TCV data is extremely small (might just be you), so there should be plots that demonstrate performance on only a subset of the above devices (probably C-Mod since we're aiming to release all of it anyway)

A simple `uv sync` should work. However, there might be some finagling you have to do to get the GPU dependencies working.
I think this is a symptopm of POPSIM having a slightly older JAX dependency. Will get around to fixing that at some point in the future (likely after APS 2026).
For the ORCD cluster in particular, the required dependency can be installed with `uv sync --group gpu-orcd`.

# Repo organization:

## Datasets
Creation of the datasets for C-Mod, DIII-D, and TCV
Requires proper data access

## Modules
All the modules implemented as part of this study

`power_balance`: Used for power balance transfer learning

`profile_trajectory`: Used for DIII-D trajectory optimization
- Has a slightly different version of the profile predictor module

# Generative AI Disclosure

Github Copilot and Claude Code were used for code completion, snippet generation, and code review. However, the results of this were carefully vetted. A human has read and understands every line in this repo.