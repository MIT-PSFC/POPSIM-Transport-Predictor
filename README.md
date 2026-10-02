# POPSIM-Transport-Predictor

Benchmarking hybrid physics architectures for tasks related to full-shot transport prediction.

We desire models which can predict the evolution of the plasma state according to future actuator control signals.
Such control-oriented models enable using Model Predictive Control (MPC) and Reinforcement Learning (RL) to design feedforward trajectory targets which avoid instabilities.
This has been demonstrated to work in the past for several different tasks on multiple tokamaks.
However, these studies trained and developed their models using data from the same tokamak on which they were deployed (thousands of shots in some instances). This is only possible because the consequences of failure on present-day experimental tokamaks are low. This will *not* be the case for next-generation tokamaks which operate at significantly higher plasma current and stored energy.
Before using MPC+RL for trajectory optimization on a next-generation tokamak, we must have confidence that the models we are using can achieve good predictive performance without training data from hazardous operating regimes (and preferably with as little data from the target device as possible).

My work focuses on Neoclassical Tearing Modes (NTMs), which are pressure-driven instances of magnetic reconnection that grow into magnetic islands.
NTM stability is influenced by kinetic profiles of temperature and density.
The idea is that if we can understand how actuators available on a next-generation tokamak influence the kinetic profiles, we can design trajectories which are more passively stable against NTM onset.
(Note that this also requires a model relating profiles to NTM stability, which is ongoing work)

Predicting kinetic profiles is essentially predicting particle and energy transport, hence the name of this repo.

# Studies

The goal of each of these study types is to compare the performance of various model architectures when trained on a lot of data from several devices and a small amount of data from a separate target device.

In particular, we aim to answer the following questions:

### 1: Which model architecture is best at domain adaptation?

The Scientific Machine Learning (SciML) paradigm describes model architectures which have some physics structure included in them (like a neural differential equation). The idea is that this physics structure enables these models to train faster and/or extrapolate better outside their training domain than an unstructured model (like a multilayer perceptron).

My hypothesis is that SciML / hybrid physics models will do well when there is little data from the target domain. However, if there is plentiful target data then the expressiveness of purely data-driven models will likely enable them to do better.

### 2: Is data from other devices needed at all / can we reach good predictability quickly from nothing (exnihilo) on the target device?

Curating multi-device datasets is extremely challenging. Considerations must be taken to handle differences in geometries, diagnostics, actuators, and (most difficult of all) data sharing agreements. If we could get away with training models on a few low-hazard shots from the target device, that would be excellent. I'm not expecting this to be the case, but it's worth investigating.

### 3: If we have a lot of historic data, how many shots are needed from the target device to achieve good predictions?

To answer this question, for each model we do training with an increasing number of shots from the target device. (e.g. 0, 1, 3, 10, 32, 100, etc.). This is on a per-shot basis because that is the meaningful unit for tokamak operations. If you run a shot, you get all the data from that shot. When commissioning future tokamaks this is also important because we would like to have all our predictive models trained before going to more hazardous operating scenarios.

### 4: How should we be normalizing our input features?

The raw input features (plasma current, toroidal field, major and minor radii, etc.) from different devices are wildly different, which can be problematic for data-driven models. There are many ways to process the input features to maintain the same dimensionality but have better domain overlap, and we will compare the results for the following:
- Raw: No normalization. Ip, Wtot, R_geo, a_minor, etc. are in their original units
- Physics: Convert to dimensionless parameters like q95, betan, epsilon, etc.
- Z-Score: Within each device, normalize each variable to zero mean and unit variance.
- CORAL: Correlation alignment algorithm (https://arxiv.org/abs/1612.01939) to align the covariances of source and target domains

### 5: How should we be training our networks for a new domain?

Many methods exist to take a data-driven model trained in one domain and adapt it to a new domain. Here, we investigate three:
- Addition: The model is trained once on a dataset which includes both historic data and target data. Samples are unweighted.
- Weighted: The model is trained once on a dataset which includes both historic data and target data. The samples are weighted such that the target ones are more important.
- Transfer: The model is trained for many epochs on historic data, and the training is continued with a reduced learning rate using target data.

## Study 1: Time-dependent power balance

Time-dependent stored energy prediction. Given the present timestep's stored energy and controllable input signals, predict the change in stored energy to the next timestep.

The input signals are Ip_MA, B_geo, ne20, P_aux_MW, and shaping (R_geo, a_minor, kappa).
These signals were chosen since they are present in the H89/H98 scaling laws which we are comparing against.

Four model architectures investigated:

1: SciML-ScalingLaw: Predicting energy confinement time with H89/H98 scaling laws, and a P_LH scaling law to switch between them. Since the H98 scaling law takes a 'P_abs' but we can only directly control 'P_aux', it also has a network which predicts the ohmic power from the remaining inputs (essentially Ip * network).

2: SciML-NN: Predicting energy confinement time with a neural network. Change in stored energy is P_oh + P_aux - P_rad - P_cond, where P_cond = Wtot / tau_e. tau_e very difficult to figure out from first principles, so use a neural network. In addition, P_oh and P_rad can't be directly controlled from our inputs, so have networks which predict their values (essentially Ip * network and ne * network).

3: MLP: Directly predict change in stored energy from a neural network (in this case, a simple multilayer perceptron). Has all the same inputs as the other models, but the architecture itself has no physics information.

4: Transformer: Again, just directly predicting change in stored energy with a purely data-driven model, but a different one to compare against the MLP. This network attends over its previous predictions with an embedding to provide information on ordering (most recent vs most distant predictions).

We also compare the different data normalization methods, where normalization is implemented as a POPSIM module. This is done since each model above may need real-valued units as well as those which are normalized for input to a neural network. Dimensionality is maintained so no new information is being provided. In addition, the normalization methods are set up via configuration parameters that are chosen based on only the available training data so there is no out-of-domain information obtained from this processing.

## Study 2: Time-independent profile prediction

For this study, the inputs are Ip_MA, B_geo, ne20, beta, and shaping (R_geo, a_minor, kappa, delta_top, delta_bot).
This is similar to the plasma control system setup you might see on a present-day experimental device like DIII-D.
Feedforward targets are given for each of the above quantities, so you have a reasonably good idea of what they will be in the future.
The goal of the model is then to predict what the 1D profiles of temperature and density will be for a given set of inputs.

Four model architectures are investigated:

1: Shape-init (K-Means or PCA): From only the training data, determine the dominant profile shapes. Then use a small neural network to determine how they are added together to produce the predicted profile. Since the shapes are normalized, they are scaled back to physical quantities using the inputs (line-averaged density and an estimate of temperature from prescribed betan).

2: MLP: Go directly from inputs to output profiles with an unstructured neural network.

3: Reservoir: Go directly from inputs to output profiles with a reservoir architecture. This has been used for a similar problem setup in the past for DIII-D. The reservoir itself is untrained, only the weights in the readout layer are updated.

4: TORAX: Utilize a fast, differentiable transport code to predict profiles. The inputs to TORAX are geometry, boundary conditions, particle/energy sources and sinks, and the transport coefficients. These are computed from the input features where possible, and predicted with small neural networks otherwise. The kinetic profiles start as an initial parabolic guess, and then the transport solver iterates them for a few timesteps until they relax into the final shape which is taken as the prediction.

## Study 3: Time-dependent profile prediction

TODO(ZanderKeith)

# Repo Organization:

For robots, see `CLAUDE.md`.

For humans, read on. (or check out `.ua/`)

## `transport_study/datasets`

Creation of the datasets for C-Mod, MAST, DIII-D, and TCV.
There's a general `DataWorkflow` class which organizes the steps in making these datasets, however running it for a particular machine requires data access.

MAST is publicly hosted so anyone can create the dataset for that device.
In the near future `TODO(ZanderKeith)` I will be releasing the full C-Mod and MAST datasets used in the studies presented at ICDDPS 2026.

It is unlikely I will be able to release the DIII-D and TCV datasets.

See `transport_study/datasets/dataset_creation.md` for more info.

## `transport_study/modules`

All the POPSIM modules, training run builders, and configs used in the three studies.

## `transport_study/orchestration`

The `Study` class is the workflow that goes from datasets to trained models and their performance statistics. Each study type described above implements a subclass of `Study` to handle its unique model and data setup.

This section also has helper functions for launching SLURM jobs and hyperparameter tuning using WandB.

## `transport_study/power_balance_transfer`, `transport_study/profile_transfer`, `transport_study/transport_transfer`

Everything unique to each study type, primarily module setup and plotting.

## Trajectory Optimization

This was briefly attempted on DIII-D in March 2026, though results were inconclusive due to difficulties in reproducing the target scenario. Again, out of date, but keeping around as an example

# Misc. Notes from the Author

I am not completely satisfied with how things are set up right now. The workflows are a consequence of my development environment and as such might be tricky to adapt. I've attempted to make the structure general where possible, but as it stands this is a project I made to facilitate my own work on a tight schedule. This whole thing could be iterated upon to smooth out the sharp edges, I simply do not have the bandwidth to do so. I'm still going ahead and releasing this on the off chance someone finds it useful. If you do, please cite the [POPSIM Repo](https://github.com/MIT-PSFC/POPSIM-Public) and its related [arXiv publication](https://arxiv.org/abs/2509.10244).

The C-Mod and MAST datasets are built from the published stores of [transport-validation-datasets](https://github.com/MIT-PSFC/transport-validation-datasets), which is where the computationally intensive GP fitting of their Thomson scattering data now lives. DIII-D is pulled on OMEGA, which has the DIII-D data servers and the IDA profile database, and TCV is built from DEFUSE exports.

I run the training orchestration on an interactive node on a compute cluster (either Engaging or OMEGA). This is convenient for testing because whenever something breaks I can just run the study serially to step through the logic with a debugger.

For installation, run `git submodule update --init` (popsim, torax and transport-validation-datasets install from their submodules), then a simple `uv sync` should work, and `uv sync --group gpu` adds the CUDA 13 JAX plugin for GPU nodes. With that plugin installed, anything that imports JAX hangs on a node without a GPU unless `JAX_PLATFORMS=cpu` is set (the test suite's conftest sets it automatically).

# Generative AI Disclosure

GitHub Copilot and Claude Code were used for code completion, snippet generation, and code review. However, the results of this were carefully vetted. A human has read and understands every line in this repo (except for plots, which were fully vibe coded).