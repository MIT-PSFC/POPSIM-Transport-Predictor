Every device store shares one on-disk schema (`transport_study/signals.py`, `STORE_SIGNALS`):
IMAS names in SI units on `(shot, time_idx[, rho_tor_norm])`, with `time` on `(shot, time_idx)`.
The studies convert to their working units on load (`signals.convert_to_working_units`).
Dataset creation is done in slightly different ways for each device:

1. C-Mod and MAST

Both are built from the published Zarr stores of
[transport-validation-datasets](https://github.com/MIT-PSFC/transport-validation-datasets) ("zk" GP fit method),
which handles the source pulls (MDSplus through disruption-py for C-Mod, the STFC S3 store for MAST),
the filtering, and the GP profile fitting on rho_tor_norm.
The build only selects the stored signals, trims each shot's trailing NaN padding,
takes ip and b0 as magnitudes (published datasets keeps the source sign, which isn't needed here),
and zero-fills `power_ec`, which neither device has.
Shot quality is the published store's responsibility, so nothing is culled here.

```bash
python -m transport_study.datasets.cli cmod <data_assembly_dir> <cmod_published.zarr>
python -m transport_study.datasets.cli mast <data_assembly_dir> <mast_published.zarr>
```

2. DIII-D

Pulled with disruption-py 0.14.0 from PyPI, in the main uv venv.
disruption-py 0.14.0 pins numpy < 2, which `[tool.uv] override-dependencies` in `pyproject.toml` lifts
(none of the code paths used here need numpy < 2).
The OMEGA system MDSplus cannot import under numpy >= 2, so disruption-py falls back to the mdsthin thin client on its own,
with the login PYTHONPATH left as it is.
Everything comes from one `get_shots_data` call per shot, built-in methods (EFIT scalars, Ip)
plus the custom ones in `d3d/physics_methods.py`:

- EFIT: every EFIT signal comes from the shot's latest code_rundb run tagged `DISPY` (the 1 kHz disruption-efit).
  A shot without one, or whose EFIT is slower than 1 kHz, is skipped (`DispyEfitNicknameSetting`, `Uniform1kHzTimeSetting`).
  disruption-py's own EFIT selection falls back to the 50 Hz efit01 and forces runtag DIS under pytest, so it is not used.
- Profiles: Te/ne from IDA, searched in the priority order of the databases in `d3d/config.toml`:
  `HBP_database` (98 shots), the VVUQ files (4), then the general-purpose `TMDB_V1c`, `TMDB_V1a`, and `TokaMaker_database`,
  which only serve shots on `d3d/HBP_shotlist_2013_2025` (105 more). The default shotlist is the union, 207 shots.
  The first two are full IDA, the general-purpose ones are IDA-lite runs with fewer diagnostics.
  On the 38 shots both `HBP_database` and `TMDB_V1c` have, the two agree to a few percent, well within the error bars.
  No GP fitting is needed since IDA profiles are already a Bayesian fit with errors.
  IDA files are on psi_n (poloidal flux) and carry no rho coordinate, so each IDA slice is mapped to rho_tor_norm
  through the q profile of the nearest DISPY EFIT slice, with the same definition and secant extension past the LCFS
  as the published C-Mod/MAST stores (`d3d/profiles.py`).
  The profiles go onto rho_tor_norm = linspace(0, 1.1, 56), the published C-Mod grid (MAST's has 67 points over the same range),
  and each slice is held onto the 1 kHz timebase until the next one, for at most 3 median IDA steps.
  IDA gives no point covariance, so the gradient errors assume independent points.
- `n_e_line_average` is `\density` of the DISPY tree, or the PCS estimate `dssdenest` where the tree has none (199264),
  which matches `\density` within 1 percent where both exist.
  disruption-py's `get_density_parameters` is not used, since it falls back to `\denv2`, which reads about 3x higher.
  Interferometer fringe jumps (201907, 199126) are dropped by the `n_e_line_average` range filter.
- `power_ohm` is `poh` of the DISPY EFIT, Ip V_surf - dW_pol/dt with V_surf = -2 pi dpsi_bdy/dt, smooth at 1 kHz.
  disruption-py's `get_ohmic_parameters` is not used:
  its 20 kHz `vloopb` with a 0.55 ms median filter is noise at 1 kHz (p1 / p99 of -4 / +4.6 MW on a 0.05 MW median).
- `power_radiated` is `\bolom::prad_tot`, the standard bolometer analysis total including the divertor,
  sampled every 4 ms and smoothed non-causally over 50 ms. Its units label reads MW, but the values are W.
  disruption-py's `pwrmix` (`get_power_parameters`) is not used:
  it is a causal 10 ms sum of the 48 raw channels that resolves ELMs and goes negative,
  so after the clip at 0 it cycled between 0 and 4-9 MW in heated H-modes.
  The NBI (`pinj`) and ECH (`echpwrc`) powers are read from the same nodes as `get_power_parameters` (`get_heating_powers`).

Processing writes two stores on the same shot / time_idx layout, both in IMAS names and SI units, with description,
units, and ref (IMAS path) attributes on every variable: `ds.zarr`, the shared schema, and `trajopt.zarr`,
the PCS programmed targets (feedforward control) and the measured signals only the trajectory optimization reads.
Each raw file records the IDA file its profiles came from (`ida_path` attribute), and both stores carry `ida_source`,
a JSON object mapping every stored shot to the folder of its IDA file (`json.loads(ds.attrs["ida_source"])`).
Three PCS pointnames of unverified meaning (`bttbt`, `dstdenp`, `ieeseg07`) are kept under their raw names and units,
see the TODO in `d3d/d3d_dataset.py`.

Filtering and culls in processing (`D3DDataWorkflow`, `RawFileWorkflow.filter_ds`):

- The plasma ends at the last slice with ip >= 0.2 MA, and each shot is cut 50 ms before that.
  DIII-D ip reads near 0 out to the 8 s end of the timebase, so the last finite ip is not the end.
- A P_oh transient cuts the rest of the shot: its centered 5 ms boxcar above 2 MW.
  The EFIT P_oh stays below 0.6 MW through the ramp-up and only crosses 2 MW at disruptive terminations.
  P_rad has no such cut, since values up to 16 MW occur mid-shot in high-power shots.
- Slices with elongation below 1.3 (deep limited rampdowns, the worst stored energy glitches)
  or a triangularity at the EFIT saturation (|delta| > 0.99) are dropped.
- `excluded_shots` in `d3d/config.toml` culls 203549, 203551 and 203554,
  failed beam shots with HFS pellets from run 20250529 (see the logbook).
- Known and kept as is:
  - The DISPY EFIT scalars carry single-slice spikes (Wtot and triangularity).
    EFIT chisq does not flag them, and they are not median-filtered.
  - The first ~80 ms of the TMDB_V1c IDA-lite records (0.11-0.2 s, Ip ~0.4 MA) are often hollow in both Te and ne.
    They are kept as ramp-up data.
disruption-py writes no netCDF (its output setting has `path=False`), only a small `config.json` per call
under `$LOCALSCRATCH/$USER/disruption-py/`.

```bash
python -m transport_study.datasets.cli d3d <data_assembly_dir> --mode raw
JAX_PLATFORMS=cpu python -m transport_study.datasets.cli d3d <data_assembly_dir> --mode process
```

Processing imports popsim and with it JAX, so on a node without a GPU it needs `JAX_PLATFORMS=cpu`.

3. TCV

Read straight from the DEFUSE exports (`TCVno{shot}.h5`, MATLAB v7.3, read with h5py)
and the LIUQE reconstructions of the MEQ databases (`TCV{shot}_meqdb.mat`), both paths in `tcv/config.toml`.

- 0D signals: DEFUSE, SI apart from NBI, NBI2 and ECRH, which it stores in MW.
  Newer shots carry ECRH as one row per gyrotron, and the rows are summed.
  A heating system a shot does not have is an empty placeholder in its export and counts as zero.
- Profiles: the DEFUSE Te/ne fits, which are on rho_pol = sqrt(psi_N), not rho_tor_norm
  (the raw Thomson positions match sqrt(psi_N) of LIUQE within 0.006, and miss rho_tor_norm by 0.05-0.08).
  Each fit slice maps to rho_tor_norm through the q profile of the nearest LIUQE reconstruction within 2 ms (`tcv/profiles.py`).
  Phi_N is the integral of q over psi_N, the same definition as the other stores.
  q diverges at the LCFS of a diverted plasma,
  so past the last surface of finite q it is integrated analytically as q = a - b ln(1 - psi_N), fit to the four surfaces inside it.
  LIUQE's own enclosed toroidal flux (FtPQ) is not used,
  it is quantized: staircased near the axis, and jittering 0.002-0.003 in Phi_N between 1 ms reconstructions at the edge.
  Only shots with a MEQ database are built, 964 shots from 60001 to 82878.
- The profiles go onto rho_tor_norm = linspace(0, 1.1, 56), the DIII-D and C-Mod grid,
  and are NaN past the LCFS, where the DEFUSE fits end.
  Each slice is held onto the 1 kHz timebase until the next one (every ~17 ms), for at most 1.5 median DEFUSE steps.
  A slice with a NaN fit point, or without both a Te and an ne fit, is dropped.
- The gradient companions are taken on the DEFUSE fit points before regridding.
  The DEFUSE fits wiggle on the scale of their own grid, so the gradients are noisy.
  DEFUSE gives no uncertainty, so the error companions are the 0 sentinel.

Processing adds two shot culls to the DIII-D ones, both opt-in in `RawFileWorkflow`:
a radiated fraction floor (a mean P_rad below 2.5 percent of the mean input power is a dead bolometer)
and the density ratio check of the published stores
(the shot median over fresh slices of mean(n_e for rho_tor_norm <= 1) / n_e_line_average must sit in 0.7-1.3).
On a 20-shot sample the radiated fractions were 0.08-0.61 and the density ratios 0.88-1.12.

```bash
python -m transport_study.datasets.cli tcv <data_assembly_dir> --mode raw
JAX_PLATFORMS=cpu python -m transport_study.datasets.cli tcv <data_assembly_dir> --mode process
```

The raw stage takes about 10 s per shot, most of it loading the MEQ database (about 250 MB in memory).
Shots whose DEFUSE export has no profile fit (a uint64 [0 0] placeholder) are skipped.
