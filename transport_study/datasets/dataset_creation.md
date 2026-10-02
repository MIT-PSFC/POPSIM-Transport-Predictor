Every device store shares one on-disk schema (transport-validation-datasets `store_schema.STORE_SIGNAL_ATTRS`):
IMAS names in SI units on `(shot, time_idx[, rho_tor_norm])`, with `time` on `(shot, time_idx)`.
The studies convert to their working units on load (`signals.convert_to_working_units`).

Every stored value is causal, no time draws on a later sample, except power_ohm and power_radiated.
Unsmoothed, both are noise-dominated at 1 kHz, so they are stored smoothed:
DIII-D's at the source (below),
and the rest with the DIII-D bolometer kernel, a centered 50 ms boxcar applied twice
(transport-validation-datasets `smoothed_power`, which TCV's computed `power_ohm` uses too).
TCV `PradTot` is the exception, it already reads smooth at its ~17 ms DEFUSE cadence and is not smoothed further.
It is held causally for at least 60 ms over its skipped samples (TCV section below),
a zero-order hold rather than the 50 ms triangle.
A 0D signal is never interpolated onto the 1 kHz grid (`signal_on_grid`).
One sampled faster than the grid is averaged over each grid step, time t taking the mean of (t - 1 ms, t].
One sampled slower is held forward from its last finite sample for at most 1.5 of its own median sample steps,
or for at least 10 ms when it comes from the equilibrium reconstruction (`EQUILIBRIUM_HOLD_FLOOR`),
so a few missing reconstructions do not cut the shot.
`fresh_equilibrium` marks the grid times a reconstruction usable for the profile mapping lands on.
The 0D equilibrium signals are placed from their own samples, so they can update where it is 0.
Derivatives are backward differences, apart from those DIII-D's EFIT takes for `poh`.
b0 is the vacuum toroidal field at the fixed major radius r0 on every device, as IMAS defines it.
The study derives the field at the geometric axis on load, b_geo = b0 r0 / geometric_axis_r (`convert_to_working_units`).
beta_tor_norm is normalized with B_geo on every device, in beta_tor = 2 mu0 <p> / B_geo^2 and in a B_geo / Ip,
which is what the study's beta inversions assume, and not the IMAS b0 at r0.
MAST efm and DIII-D EFIT store it so, C-Mod rebuilds it from EFIT betat (its betan takes the total field at the axis),
and TCV computes it from LIUQE Wtot and Vol (DEFUSE BETAN normalizes by the volume-averaged vacuum field and BZERO).
power_radiated is the total radiated power including the divertor,
and power_ohm is Ip V_loop - dW_pol/dt.
rho_tor_norm = sqrt(Phi_N), with Phi_N the integral of q over psi_N (`phi_n_map`),
imported from transport-validation-datasets like every helper the devices share (`machine/generic.py`).

One filter spec, imported from transport-validation-datasets (`filters.py`), which applies it to C-Mod and MAST
(`RawFileWorkflow.filter_ds` and `cull_shot` for DIII-D and TCV).
Every check cuts the times it fails out as a gap:

- the end of the shot: the plasma ends at the last time |ip| reaches its min threshold,
  and everything after `end_margin` before that is cut (`end_of_shot_index`)
- a 0D signal of `DATASET_0D_SIGNALS` that is not finite
- a `min_filter` signal below its threshold (ip as |ip|), or a `max_filter` signal above it, on the raw samples.
  `greenwald_fraction` = n_e_line_average / n_GW with n_GW = Ip / (pi a^2) is derived for the max filter and not stored
- a `transient_filter` signal above its threshold after a centered 5 ms boxcar,
  which only selects times and smooths no stored value
- the 20 ms before any failed time short of the end-of-shot cut (`FAILURE_MARGIN`),
  since the non-causally smoothed P_oh and P_rad rise ahead of the event that ends a segment

Only the longest contiguous segment is kept, and a shot is culled when that segment spans less than `min_pulse_length`,
its mean P_rad is below `min_radiated_fraction` of its mean input power (a dead bolometer)
or above `max_radiated_fraction` of it (more radiated than put in),
its stored energy rises by more than 1.05 times the input energy (a broken power record),
or the median over fresh slices of mean(n_e for rho_tor_norm <= 1) / n_e_line_average is outside `density_ratio_bounds`.
Powers are clipped at 0 after filtering.
Every shot's unfiltered 0D signals are plotted once, with transport-validation-datasets' `plot_unprocessed_data`, as for C-Mod and MAST:
into `<dataset dir>/accepted_shots/` with the kept segment shaded green when the shot reaches the stores,
into `<dataset dir>/rejected_shots/` when the filter or a cull drops it,
with the transients shaded red, the end-of-shot cut, the thresholds, and dots where a profile is measured.

| Threshold | C-Mod | MAST | DIII-D | TCV |
| --- | --- | --- | --- | --- |
| min ip | 100 kA | 210 kA | 200 kA | 50 kA |
| min energy_mhd | 2.7 kJ | 5 kJ | 10 kJ | 1 kJ |
| min n_e_line_average | 1e19 m^-3 | 3e18 m^-3 | 5e17 m^-3 | 2e18 m^-3 |
| max greenwald_fraction | 2.0 | 2.0 | 2.0 | 2.0 |
| transient power_ohm | 5 MW | 5 MW | 2 MW | 2 MW |
| transient power_radiated | 5.5 MW | 3 MW | 17 MW | 5 MW |
| end_margin | 20 ms | 40 ms | 50 ms | 50 ms |
| min_pulse_length | 0.5 s | 0.2 s | 0.5 s | 0.5 s |
| min_radiated_fraction | 0.01 | 0.025 | 0.025 | 0.025 |
| max_radiated_fraction | 1.0 | 1.0 | 1.0 | 1.0 |
| density_ratio_bounds | 0.72-1.3 | 0.7-1.3 | 0.7-1.3 | 0.7-1.3 |

Dataset creation is done in slightly different ways for each device:

1. C-Mod and MAST

Both are built from the published Zarr stores of
[transport-validation-datasets](https://github.com/MIT-PSFC/transport-validation-datasets) ("zk" GP fit method),
which handles the source pulls (MDSplus through disruption-py for C-Mod, the STFC S3 store for MAST),
the filtering, and the GP profile fitting on rho_tor_norm.
The build only selects the stored signals, trims each shot's trailing NaN padding,
and takes ip and b0 as magnitudes (the published stores keep the source sign, which isn't needed here).
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
Everything comes from one `get_shots_data` call per shot, through the custom methods in `d3d/physics_methods.py`.
The disruption-py built-ins interpolate onto the timebase, so the EFIT scalars and Ip are read by custom methods too,
and every signal is placed on the timebase causally instead (`signal_on_grid`, averaged per grid step when faster, held when slower):

- EFIT: every EFIT signal comes from the shot's latest code_rundb run tagged `DISPY` (the 1 kHz disruption-efit).
  A shot without one, or whose EFIT is slower than 1 kHz, is skipped (`DispyEfitNicknameSetting`, `Uniform1kHzTimeSetting`).
  disruption-py's own EFIT selection falls back to the 50 Hz efit01 and forces runtag DIS under pytest, so it is not used.
  Failed reconstructions are missing or fail chisq > 50, so every EFIT signal is held for at least 10 ms,
  and `fresh_equilibrium` marks the grid times a usable slice (chisq, mappable q) lands on,
  while the EFIT 0D signals take every slice that passes chisq.
- Profiles: Te/ne from IDA, searched in the priority order of the databases in `d3d/config.toml`:
  `HBP_database` (98 shots), the VVUQ files (4), then the general-purpose `TMDB_V1c`, `TMDB_V1a`, and `TokaMaker_database`,
  which only serve shots on `d3d/HBP_shotlist_2013_2025` (105 more). The default shotlist is the union, 207 shots.
  The first two are full IDA, the general-purpose ones are IDA-lite runs with fewer diagnostics.
  On the 38 shots both `HBP_database` and `TMDB_V1c` have, the two agree to a few percent, well within the error bars.
  No GP fitting is needed since IDA profiles are already a Bayesian fit with errors.
  IDA files are on psi_n (poloidal flux) and carry no rho coordinate, so each IDA slice is mapped to rho_tor_norm
  through the q profile of the nearest DISPY EFIT slice, with the same definition and secant extension past the LCFS
  as the published C-Mod/MAST stores (`d3d/profiles.py`).
  IDA's psi_n comes from its own reconstruction, which the files do not name, while q comes from the DISPY EFIT.
  The profiles go onto rho_tor_norm = linspace(0, 1.1, 56), the published C-Mod grid (MAST's has 67 points over the same range),
  and each slice is held onto the 1 kHz timebase until the next one, for at most 3 median steps of the usable slices (`hold_onto_grid`).
  A slice with no valid EFIT nearby, or without both a Te and an ne fit, is dropped before the hold,
  so the slice before it holds over it, and `fresh_profile` marks where the usable slices land.
  IDA gives no point covariance, so the gradient errors assume independent points.
- `b0` is mu0 144 bcoil / (2 pi r0) from the TF coil current (PTDATA `bcoil`), with r0 = 1.6955 m.
  That is how EFIT computes `bcentr`, and the two agree to 0.01 percent,
  but `bcoil` has no EFIT dropouts.
  PTDATA `bt` reads 2.4-2.8 percent above it, at a reference radius nowhere documented.
- `n_e_line_average` is `\density` of the DISPY tree, or the PCS estimate `dssdenest` where the tree has none (199264),
  which matches `\density` within 1 percent where both exist.
  disruption-py's `get_density_parameters` is not used, since it falls back to `\denv2`, which reads about 3x higher.
  A gained fringe count (201907, up to 3.9e20) is caught by the Greenwald fraction max,
  and a lost one (199126, down to -1.7e20) by the 5e17 floor.
- `power_ohm` is `poh` of the DISPY EFIT, Ip V_surf - dW_pol/dt with V_surf = -2 pi dpsi_bdy/dt.
  EFIT takes both derivatives (PSIBDYDOT, WBDOT) as centered least-squares slopes over +-100 ms,
  so each sample draws on slices up to 100 ms later, and it is not smoothed further.
  Rebuilt from PSIBDY, WB and IPMEAS with +-100 ms slopes, it is 0.995-1.014 of the node on 4 shots.
  No shorter window works on beam-heated shots, where P_oh is ~0.05 MW and each term swings far more over 10 ms.
  disruption-py's `get_ohmic_parameters` is not used:
  its 20 kHz `vloopb` with a 0.55 ms median filter is noise at 1 kHz (p1 / p99 of -4 / +4.6 MW on a 0.05 MW median).
- `power_radiated` is `\bolom::prad_tot`, the standard bolometer analysis total including the divertor,
  sampled every 4 ms from raw channels smoothed by a centered 50 ms boxcar applied twice,
  a triangle 100 ms wide at its base with a 50 ms FWHM, so it draws on raw data up to 50 ms later.
  It is not smoothed further, and the other devices' powers get the same kernel.
  Its units label reads MW, but the values are W.
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
A trajopt signal outside its range in `TRAJOPT_VALID_RANGES` is NaN, it is not part of the filter spec.

DIII-D specifics of the filter spec (`D3DDataWorkflow`):

- DIII-D ip reads near 0 out to the 8 s end of the timebase, which is why the end of the shot is the last |ip| above its threshold.
- The P_oh transient is a 5 ms boxcar above 2 MW.
  The EFIT P_oh stays below 0.6 MW through the ramp-up and only crosses 2 MW at disruptive terminations.
  The P_rad transient is a 5 ms boxcar above 17 MW, above the 16.0 MW the iteration_3 store reaches.
- n_e_line_average below 5e17 m^-3 is a gap.
  Ramp-ups reach 1e18 (199121), single-slice dropouts in the rampdown read below 3e17 (201914, 201935, 203836, 204188),
  and a lost fringe count drives it negative (199126 after 4.41 s), which the Greenwald fraction max lets through.
- `excluded_shots` in `d3d/config.toml` culls 203549, 203551 and 203554,
  failed beam shots with HFS pellets from run 20250529 (see the logbook).
- Known and kept as is:
  - The DISPY EFIT scalars carry single-slice spikes (Wtot and triangularity).
    EFIT chisq does not flag them, and they are not median-filtered.
    Triangularity also saturates at exactly 1 for a few ms at a time (runs up to 138 ms in 204191),
    and deep limited rampdowns (elongation ~1.2, 199243, 199244, 203460) carry the largest Wtot spikes.
    Neither is filtered, since every filtered slice is a gap that splits the shot.
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
  The LIUQE signals (Wtot, Vol, a_minor, R_geom, KAPPA, DELTA_TOP, DELTA_BOTTOM, LI, BZERO) are held for at least 10 ms.
- `power_radiated` is `PradTot`, the total including the divertor like DIII-D and MAST.
  `PradBulk`, the confined plasma only, is 0.43 of it at the median over 81 shots (0.27-0.69 for 5-95 percent).
  It follows the Thomson cadence (~17 ms) but often skips one or two samples, leaving 33-50 ms steps,
  and 61056 comes in bursts of three samples every 50 ms.
  The 1.5-step hold (25.5 ms) left NaN gaps that cut ~10 percent of the kept time (5.5 s) and all of 61056,
  so it is held for at least 60 ms (`PRAD_TOT_HOLD_FLOOR_S`), which bridges both.
  The hold is causal, a zero-order hold rather than the 50 ms triangle of the other devices' P_rad.
  PradTot is not smoothed further, since it already reads smooth at its own cadence (DEFUSE documents no kernel).
- `fresh_equilibrium` marks the grid times a usable LIUQE reconstruction of the MEQ database lands on,
  while the LIUQE 0D signals come from the DEFUSE export on its own times.
- `power_ohm` is computed, Ip V_loop - d/dt(mu0 R_geo li Ip^2 / 4) from DEFUSE `I_P`, `Vloop`, `LI` and the geometric major radius `R_geom`
  (`ohmic_power`), then smoothed by the centered 50 ms boxcar applied twice (`smoothed_power`), as on C-Mod and MAST.
  DEFUSE `Vloop` has the opposite sign to `I_P` (Ip Vloop < 0 at flat-top on 39 of 39 shots of both polarities), so it is flipped.
  DEFUSE `POHM` has no documented definition and reads 0.9-1.0 of Ip Vloop at flat-top.
- `b0` is `BZERO`, LIUQE's rBt / r0 with r0 = 0.88 m.
- `beta_tor_norm` is computed on the timebase, 100 beta_tor a B_geo / Ip[MA] with beta_tor = 2 mu0 <p> / B_geo^2,
  <p> = 2 Wtot / (3 Vol) and B_geo = |BZERO| r0 / R_geom (`_normalized_beta`).
  DEFUSE `BETAN` is LIUQE's, which normalizes beta_tor by the volume-averaged vacuum field and multiplies by |BZERO| at r0,
  and reads a median 5.6 percent below the B_geo value.
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
  Each slice is held onto the 1 kHz timebase until the next one (every ~17 ms), for at most 1.5 median steps of the usable slices (`hold_onto_grid`).
  A slice with no usable reconstruction nearby or a NaN fit point is dropped before the hold,
  so the slice before it holds over it, and `fresh_profile` marks where the usable n_e slices land.
  A time without both a Te and an ne fit has no profiles.
- The gradient companions are taken on the DEFUSE fit points before regridding.
  The DEFUSE fits wiggle on the scale of their own grid, so the gradients are noisy.
  DEFUSE gives no uncertainty, so the error companions are the 0 sentinel.

TCV specifics of the filter spec (`TCVDataWorkflow`):

- The transients are 5 ms boxcars above 2 MW for P_oh, as on DIII-D, and above 5 MW for P_rad.
  In 246 shots only 75026 has a 5 ms P_rad peak above 3 MW (12 MW),
  and it also radiates 4.1x its input, which `max_radiated_fraction` culls.
- The raw stage removes the FIR fringe jumps from the 20-25 kHz NEavg samples before they go onto the grid (`remove_fringe_jumps`, non-causal).
  A jump is a shift of at least 1e19 m^-3 between the medians of the 0.25 ms on either side of a sample,
  which no real density change is fast enough for.
  Shifts within 5 ms of each other are one episode (a dropout and its recovery).
  The level change across an episode is taken between the medians from 2 to 10 ms on either side,
  so a spike that decays back is not removed as a jump, and the samples inside the episode become a straight line.
  Three corrected episodes within 50 ms, or one episode longer than that, means the FIR lost count,
  and NEavg is NaN from there on, which the filter cuts.
  Against Thomson `TS_nel` on 185 shots, the times more than 25 percent off drop from 3.9 to 2.0 percent
  (the earlier correction on the 1 kHz grid split a jump across two grid times and subtracted decaying spikes).
  A slip spread over ~3 ms (74207) looks like a real fast density drop and is left in.
  The density ratio cull catches that shot.

```bash
python -m transport_study.datasets.cli tcv <data_assembly_dir> --mode raw
JAX_PLATFORMS=cpu python -m transport_study.datasets.cli tcv <data_assembly_dir> --mode process
```

The raw stage takes about 10 s per shot, most of it loading the MEQ database (about 250 MB in memory).
Shots whose DEFUSE export has no profile fit (a uint64 [0 0] placeholder) are skipped.
