Every device is built by [transport-validation-datasets](https://github.com/MIT-PSFC/transport-validation-datasets)
(TVD, the `submodules/transport-validation-datasets` submodule):
the source pulls, the filtering, the profile fits and the GEQDSK equilibrium all live there,
and its README documents each device's sources, filter thresholds and known limitations.

| Device | TVD store the study reads | Profiles |
| --- | --- | --- |
| C-Mod | `<ds>_published.zarr` | GP fit (zk) of the core and edge Thomson channels |
| MAST | `<ds>_published.zarr` | GP fit (zk) of the AYC Thomson channels |
| TCV | `<ds>_internal.zarr` | GP fit (zk) of the raw DEFUSE Thomson channels |
| DIII-D | `<ds>_internal.zarr` | IDA's own GP fits, carried onto rho_tor_norm (ida) |

TCV and DIII-D data has no release permission, so TVD stops them at the internal store.

Every TVD store shares one schema (TVD `store_schema.STORE_SIGNAL_ATTRS`):
IMAS names in SI units on `(shot, time_idx[, rho_tor_norm])`, with `time` on `(shot, time_idx)`.
The build here (`workflow.StoreWorkflow`) keeps only the signals the studies read (`signals.STUDY_STORE_SIGNALS`,
the shared schema less `fresh_equilibrium`), trims each shot's trailing NaN padding,
and takes ip and b0 as magnitudes, since TVD keeps their source sign.
Shot quality is the TVD store's responsibility, so nothing is culled here.
The studies convert to their working units on load (`signals.convert_to_working_units`).

```bash
python -m transport_study.datasets.cli cmod <data_assembly_dir> <cmod_published.zarr>
python -m transport_study.datasets.cli mast <data_assembly_dir> <mast_published.zarr>
python -m transport_study.datasets.cli tcv <data_assembly_dir> <tcv_internal.zarr>
python -m transport_study.datasets.cli d3d <data_assembly_dir> <d3d_internal.zarr>
```

The build imports popsim and with it JAX, so on a node without a GPU it needs `JAX_PLATFORMS=cpu`.
It writes `<data_assembly_dir>/<ds_name>/dataset_full/ds.zarr` (or `dataset_<max_num_shots>`)
and diagnostic plots next to it: per-shot time traces, profile heatmaps, and a summary PDF.
