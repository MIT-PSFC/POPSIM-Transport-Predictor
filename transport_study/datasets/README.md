Dataset creation is done in slightly different ways for each device:

1. C-Mod

MDSPlus on the PSFC cluster supports numpy >= 2, so this is straightforward.
Use Disruption-Py to get the 0D scalars and the 1D data separately, do some preprocessing, and bring them together.

2. DIII-D

MDSPlus on Omega requires numpy < 2 which is incompatible with POPSIM, so we have a separate virtual environment
that only has the requirements for Disruption-Py to first pull the data. (set up this venv with `make_d3d_venv.sh`)
Further filtering and preprocessing is done on the uv-managed venv with numpy >= 2.

3. TCV

TCV has DEFUSE which is in Matlab. Take h5 files output from DEFUSE, convert them into an Xarray-friendly format,
and do the remainder of the data preparation workflow from there.

4. MAST

Reads 0D signals and raw Thomson channel data from the STFC ECHO S3 open-access Zarr store (no credentials needed),
maps Thomson positions to psi_n using the level1 EFM equilibrium, and GP fits profiles the same way as C-Mod.

## Distributed GP profile fitting (C-Mod and MAST)

GP profile fitting with gptools takes ~10 minutes per shot, so fitting a ~1000 shot dataset serially takes about
a week. The cmod and mast workflows can dispatch the fitting to a SLURM cluster while keeping data retrieval and
dataset assembly local (the cluster has no access to the C-Mod data source). Without any cluster options the
workflows run fully serially, so the datasets remain reproducible without a cluster.

1. Prepare (local): download and validate each shot, cache the source data in `<ds_name>/fit_staging/`, and
   extract the Thomson channel arrays that the fit needs.
2. Fit (cluster): shots are packed into one npz per batch (avoids many-small-file transfers), uploaded together
   with the self-contained `gp_fitting/fit_worker.py`, and fit by one CPU job per batch. Jobs have deterministic
   names (`gpfit-<device>-<batch-hash>`) so a restarted workflow adopts in-flight jobs instead of resubmitting.
3. Assemble (local): results are pulled back and combined with the staged data into the same raw netCDF files
   the serial workflow produces.

One-time setup per cluster:

```bash
# Register the SSH profile (host comes from ~/.ssh/config)
srunx ssh profile add engaging --ssh-host eofe10.mit.edu

# Build the minimal fitting venv (numpy + scipy + gptools) on the cluster scratch space
bash transport_study/datasets/gp_fitting/bootstrap_remote.sh eofe10.mit.edu /pool001/$USER/gpfit
```

Then, for example:

```bash
python transport_study/datasets/cli.py cmod \
    --data_assembly_dir /usr/local/mfe/ml_data_dump/POPSIM/popsim_studies/icddps \
    --cluster_profile engaging \
    --cluster_partition mit_normal \
    --cluster_remote_workdir /pool001/$USER/gpfit \
    --cluster_venv /pool001/$USER/gpfit/.venv
```

Notes:
- The fitting partition should be a CPU partition since gptools cannot use GPUs.
- `--cluster_max_jobs` (default 8) caps concurrent jobs; `--cluster_shots_per_batch` (default 50) and
  `--cluster_cpus_per_job` (default 32) set batch size, roughly 1 hour of wall time per job at the defaults.
- For MAST running on the cluster itself, pass `--cluster_profile local` to submit sbatch jobs directly on the
  shared filesystem.
- For serial runs on a beefy machine, `--fit_workers N` parallelizes the fitting over N local processes.