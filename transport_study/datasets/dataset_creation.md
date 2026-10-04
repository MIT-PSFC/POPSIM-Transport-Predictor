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
IMAS names in SI units on `(shot, time_idx[, rho_tor_norm])`, with `time` on `(shot, time_idx)`,
one contiguous 1 kHz segment per shot padded with NaN to the longest shot,
and ip and b0 with their source sign (the per-shot `cocos` records the convention).
The build here (`workflow.StoreWorkflow`) is a lazy xarray pass over that store:
it keeps only the signals the studies read (`signals.STUDY_STORE_SIGNALS`, the shared schema less `fresh_equilibrium`),
cuts the time axis back to the longest shot, rechunks into `ds.zarr` with TVD's chunked writer
and checks the written shot count against the source.
It refuses a store whose `fit_mode` is not `sample`, since the windowed fit modes are not contiguous in time.
Shot quality is the TVD store's responsibility, so nothing is culled here.
The studies convert to their working units and take |ip| and |b0| on load (`signals.convert_to_working_units`).

```bash
python -m transport_study.datasets.cli build --device cmod --store <cmod_published.zarr> --data_assembly_dir <dir>
python -m transport_study.datasets.cli build --device mast --store <mast_published.zarr> --data_assembly_dir <dir>
python -m transport_study.datasets.cli build --device tcv --store <tcv_internal.zarr> --data_assembly_dir <dir>
python -m transport_study.datasets.cli build --device d3d --store <d3d_internal.zarr> --data_assembly_dir <dir>
```

`--ds_name` renames the output directory (the device name by default), `--max_num_shots N` builds only the first N shots
into `dataset_N`, and `--clean` deletes an earlier build first.
The build needs neither popsim nor JAX.
It writes `<data_assembly_dir>/<ds_name>/dataset_full/ds.zarr` (through `ds.partial.zarr` while incomplete, so a crashed run leaves no half store)
and diagnostic plots next to it: per-shot time traces, profile heatmaps, and a summary PDF.
