# POPSIM-Transport-Predictor

Study that investigates using POPSIM for full-shot profile prediction.

Repo utilizes uv workspaces to organize dependencies. We utilize on Disruption-Py for dataset creation, which relies on MDSPlus, which requires numpy < 2. This means it can't use the same .venv as the ML training pipeline since POPSIM uses a JAX version that requires numpy >=2.

# Installation

This repo should be able to completely reproduce all plots in the paper, provide the person trying to do the reproduction has proper data access.
The number of people who have access to C-Mod, DIII-D, and TCV data is extremely small (might just be you), so there should be plots that demonstrate performance on only a subset of the above devices (probably C-Mod since we're aiming to release all of it anyway)

