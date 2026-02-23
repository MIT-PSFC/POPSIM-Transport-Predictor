# POPSIM-Transport-Predictor

Study that investigates using POPSIM for full-shot profile prediction.

# Installation

This repo should be able to completely reproduce all plots in the paper, provide the person trying to do the reproduction has proper data access.
The number of people who have access to C-Mod, DIII-D, and TCV data is extremely small (might just be you), so there should be plots that demonstrate performance on only a subset of the above devices (probably C-Mod since we're aiming to release all of it anyway)

# Repo organization:

## Datasets
Creation of the datasets for C-Mod, DIII-D, and TCV
Requires proper data access

## Modules
All the modules implemented as part of this study

`power_balance`: Used for power balance transfer learning

`profile_trajectory`: Used for DIII-D trajectory optimization
- Has a slightly different version of the profile predictor module