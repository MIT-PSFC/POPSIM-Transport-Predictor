# POPSIM-Transport-Predictor

Study that investigates using POPSIM for full-shot profile prediction.

# Installation

This repo should be able to completely reproduce all plots in the ICDDPS presentation, and all plots in the upcoming paper provided the person trying to do the reproduction has proper data access.

A C-Mod dataset is provided, and the MAST dataset can be created using the provided scripts.

The number of people who have access to C-Mod, DIII-D, and TCV data is extremely small (might just be you), so there should be plots that demonstrate performance on only a subset of the above devices (probably C-Mod since we're aiming to release all of it anyway)

A simple `uv sync` should work. However, there might be some finagling you have to do to get the GPU dependencies working.
I think this is a symptopm of POPSIM having a slightly older JAX dependency. Will get around to fixing that at some point in the future (likely after APS 2026).
For the ORCD cluster in particular, the required dependency can be installed with `uv sync --group gpu-orcd`.

# Studies

The goal of each of these study types is to compare the performance of various architectures when trained on a lot of historic data and a small amount of data from the target device.

In particular, we aim to answer the following questions:

### 1: Which model architecture is best at domain adaptation?

My hypothesis is that SciML / hybird physics models will do well when there is little data from the target domain. However, if there is plentiful target data then the expressiveness of purely data-driven models will likely enable them to do better.

### 2: Is historic data needed at all / can we reach good predictability quickly from nothing (exnihilo) on the target device?

Obtaining historic datasets that can be mapped into the operating regime of the target device is challenging. Need to deal with differences in geometries, diagnostics, actuators, and data sharing agreements. If we could get away with training our models on one or two shots of low-performance data from the target device that would be excellent, I just don't think it is possible.

### 3: If we have a lot of historic data, how many shots are needed from the target device to achieve good predictions?

To answer this question, for each model we do training with an increasing number of shots from the target device. (e.g. 0, 1, 3, 10, 32, 100, 316, etc.). This is on a per-shot basis because that is the meaningful unit for tokamak operations, since if you run a shot, you get all the data from that shot. When commissioning future tokamaks this is also important because we would like to have all our predictive models trained before going to more dangerous high-performance operating scenarios.

### 4: How should we be normalizing our input features?

The raw input features from different devices can have little domain overlap, which can be problematic for data-driven models. There are many ways to process the input features to maintain the same dimensionality but have better overlap, and we will compare the results for the following:
- Raw: No normalization. Ip, Wtot, etc. are in their original units
- Physics: Convert to dimensionless parameters like beta, q95, f_G, etc.
- Z-Score: Within each device, normalize each variable to zero mean and unit variance.
- CORAL: Correlation alignment algorithm (https://arxiv.org/abs/1612.01939) to align the covariances of source and target domains

### 5: How should we be training our networks for a new domain?

Many methods to take a data-driven model trained in one domain and adapt it to a new domain, we investigate two:
- Data mixing: The model is trained once on a dataset which includes both historic data and target data. The samples are weighted such that the target's are more important.
- Transfer learning: The model is trained for many epochs on historic data, and the training is continued with a reduced learning rate using target data.

## Power Balance Transfer

Time-dependent stored energy prediction. Given the present timestep's stored energy and controllable input signals, predict the change in stored energy to the next timestep.

The input signals are Ip_MA, B0, ne20, P_aux_MW, and shaping (R0, a_minor, kappa).
These signals were chosen since they are present in the H89/H98 scaling laws which we are comparing against.

Four model architectures investigated:

1: Scaling Law, predicting energy confinement time with H89/H98 scaling laws, and a P_LH scaling law to switch between them. Since the H98 scaling law takes a 'P_abs' but we can only directly control 'P_aux', it also has a network which predicts the ohmic power from the remaining inputs (essentially Ip * network).

2: SciML, predicting energy confinement time with a neural network. Change in stored energy is P_oh + P_aux - P_rad - P_cond, where P_cond = Wtot / tau_e. tau_e very difficult to figure out from first principles, so use a neural network. In addition, P_oh and P_rad can't be directly controlled from our inputs, so have networks which predict their values (essentially Ip * network and ne * network).

3: Unstructured NN, directly predict change in stored energy from a neural network (in this case, simple MLP). Has all the same inputs as the other models, but the architecture itself has no physics information.

4: Transformer architecture. Again, just directly predicting change in stored energy with a purely data-driven model, but a different one to compare against the MLP.

We also compare the different data normalization methods, where normalization is implemented as a POPSIM module. This is done since each model above may need real-valued units as well as those which are normalized for input to a neural network. Dimensionality is maintained so no new information is being provided. In addition, the normalization methods are set up via configuration parameters that are chosen based on only the available training data so there is no out-of-domain information obtained from this processing.

## Profile Transfer

In the power balance transfer study we determined that 'physics' normalization performed best, so we only look at that method here.
We still compare the differences between 'mixing' and 'transfer' domain adaptations.

## Transport Transfer

Again, we only consider 'physics' normalization.
The above two studies indicated there is little difference between 'mixing' and 'transfer' domain adaptation, so we only use 'mixing' here.


# Repo organization:

## Datasets
Creation of the datasets for C-Mod, DIII-D, and TCV
Requires proper data access

## Modules

All the POPSIM modules and configs used in the various studies

`power_balance`: Used for power balance transfer learning

`profile_trajectory`: Used for DIII-D trajectory optimization
- Has a slightly different version of the profile predictor module

## Orchestration

Helper scripts for organizing training cases, launching SLURM jobs, and hyperparameter tuning.

## Trajectory Optimization

This was briefly attempted on DIII-D in March 2026, though results were inconclusive due to difficulties in reproducing the target scenario.

# Generative AI Disclosure

Github Copilot and Claude Code were used for code completion, snippet generation, and code review. However, the results of this were carefully vetted. A human has read and understands every line in this repo.