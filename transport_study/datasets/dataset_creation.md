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
Everything comes from one `get_shots_data` call per shot, built-in methods (EFIT scalars, powers, Ip)
plus the custom ones in `d3d/physics_methods.py`:

- EFIT: every EFIT signal comes from the shot's latest code_rundb run tagged `DISPY` (the 1 kHz disruption-efit).
  A shot without one, or whose EFIT is slower than 1 kHz, is skipped (`DispyEfitNicknameSetting`, `Uniform1kHzTimeSetting`).
  disruption-py's own EFIT selection falls back to the 50 Hz efit01 and forces runtag DIS under pytest, so it is not used.
- Profiles: Te/ne from the IDA database (`IDA_{shot}_.cdf`, plus the VVUQ files, see `d3d/config.toml`).
  No GP fitting is needed since IDA profiles are already a Bayesian fit with errors.
  IDA files are on psi_n (poloidal flux) and carry no rho coordinate, so each IDA slice is mapped to rho_tor_norm
  through the q profile of the nearest DISPY EFIT slice, with the same definition and secant extension past the LCFS
  as the published C-Mod/MAST stores (`d3d/profiles.py`).
  The profiles go onto rho_tor_norm = linspace(0, 1.1, 56), the published C-Mod grid (MAST's has 67 points over the same range),
  and each slice is held onto the 1 kHz timebase until the next one, for at most 3 median IDA steps.
  IDA gives no point covariance, so the gradient errors assume independent points.
- `n_e_line_average` is `\density` of the DISPY tree. disruption-py's `get_density_parameters` is not used,
  since it falls back to `\denv2`, which reads about 3x higher.

Processing writes two stores on the same shot / time_idx layout, both in IMAS names and SI units, with description,
units, and ref (IMAS path) attributes on every variable: `ds.zarr`, the shared schema, and `trajopt.zarr`,
the PCS programmed targets (feedforward control) and the measured signals only the trajectory optimization reads.
Three PCS pointnames of unverified meaning (`bttbt`, `dstdenp`, `ieeseg07`) are kept under their raw names and units,
see the TODO in `d3d/d3d_dataset.py`.
disruption-py writes no netCDF (its output setting has `path=False`), only a small `config.json` per call
under `$LOCALSCRATCH/$USER/disruption-py/`.

```bash
python -m transport_study.datasets.cli d3d <data_assembly_dir> --mode raw
JAX_PLATFORMS=cpu python -m transport_study.datasets.cli d3d <data_assembly_dir> --mode process
```

Processing imports popsim and with it JAX, so on a node without a GPU it needs `JAX_PLATFORMS=cpu`.

3. TCV

TCV has DEFUSE which is in Matlab. Take h5 files output from DEFUSE, convert them into an Xarray-friendly format,
and do the remainder of the data preparation workflow from there.
DEFUSE profiles are on rho_tor_norm and carry no error bars or gradients,
so the TCV store leaves these out and the studies fill gradients and heuristic errors on load
(`organize_data.add_missing_profile_companions`).
