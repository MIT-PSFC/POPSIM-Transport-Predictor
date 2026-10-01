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

MDSPlus on Omega requires numpy < 2 which is incompatible with POPSIM, so we have a separate virtual environment
that only has the requirements for Disruption-Py to first pull the data. (set up this venv with `make_d3d_venv.sh`)
Further filtering and preprocessing is done on the uv-managed venv with numpy >= 2.
All signals come from Disruption-Py in one call per shot: 1 kHz EFIT via the DISPY runtag trees, PTDATA and
pedestal-tree signals via the custom physics methods in `d3d/physics_methods.py`, and Te/ne profiles from the
IDA database (`IDA_{shot}_.cdf` files) interpolated from IDA's own rho_tor_norm onto a uniform grid.
No GP fitting is needed since IDA profiles are already a Bayesian fit with errors.
Processing writes two stores on the same shot / time_idx layout:
`ds.zarr`, the shared schema, and `trajopt.zarr`, the PCS programmed targets (feedforward control)
and the measured signals only the trajectory optimization reads, under their pre-IMAS names and units.
Note for `--mode process` on Omega: the login environment exports PYTHONPATH pointing at the system MDSplus,
which cannot import under numpy >= 2 and makes disruption-py refuse to import. Strip it for the processing
step: `PYTHONPATH= python -m transport_study.datasets.cli d3d <data_assembly_dir> --mode process ...`

3. TCV

TCV has DEFUSE which is in Matlab. Take h5 files output from DEFUSE, convert them into an Xarray-friendly format,
and do the remainder of the data preparation workflow from there.
DEFUSE profiles are on rho_tor_norm and carry no error bars or gradients,
so the TCV store leaves these out and the studies fill gradients and heuristic errors on load
(`organize_data.add_missing_profile_companions`).
