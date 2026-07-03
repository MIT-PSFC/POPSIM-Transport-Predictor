Dataset creation is done in slightly different ways for each device:

1. C-Mod

MDSPlus on the PSFC cluster supports numpy >= 2, so this is straightforward.
Use Disruption-Py to get the 0D scalars and the 1D data separately, do some preprocessing, and bring them together.

2. DIII-D

MDSPlus on Omega requires numpy < 2 which is incompatible with POPSIM, so we have a separate virtual environment
that only has the requirements for Disruption-Py to first pull the data. (set up this venv with `make_d3d_venv.sh`)
Further filtering and preprocessing is done on the uv-managed venv with numpy >= 2.

3. MAST

Reads 0D signals and raw Thomson channel data from the STFC ECHO S3 open-access Zarr store,
maps Thomson positions to the normalized minor radius rho (midplane distance from the magnetic axis divided by the
axis-to-LCFS distance, from the level1 EFM equilibrium), and GP fits profiles the same way as C-Mod. Profiles are
fit and stored in rho, not psi_n, because psi_n squishes the core in real space.

4. TCV

TCV has DEFUSE which is in Matlab. Take h5 files output from DEFUSE, convert them into an Xarray-friendly format,
and do the remainder of the data preparation workflow from there.


## Distributed GP profile fitting (C-Mod and MAST)

GP profile fitting with mkgp takes ~10 minutes per shot, so fitting a ~1000 shot dataset serially takes about
a week. The C-Mod and MAST workflows can dispatch the fitting to a SLURM cluster while keeping data retrieval and
dataset assembly local (the cluster has no access to the C-Mod data source). Without any cluster options the
workflows run fully serially, so the datasets remain reproducible (for anyone patient enough).

1. Prepare (local): download and validate each shot, cache the source data in `<ds_name>/fit_staging/`, and
   extract the Thomson channel arrays that the fit needs.
2. Fit (cluster): shots are packed into one npz per batch (avoids many-small-file transfers), uploaded together
   with the self-contained `gp_fitting/fit_worker.py`, and fit by one CPU job per batch. Jobs have deterministic
   names (`gpfit-<device>-<batch-hash>`) so a restarted workflow adopts in-flight jobs instead of resubmitting.
3. Assemble (local): results are pulled back and combined with the staged data into the same raw netCDF files
   the serial workflow produces.

One-time setup per cluster:

The `~/.ssh/config` entry for the cluster host must have an `IdentityFile` line. The python-SLURM library srunx connects with paramiko, which only uses the key named there.
It does not fall back to the default keys in `~/.ssh/`, and with `ProxyJump` it refuses to connect without an explicit key.

```
Host <ssh-host>
  HostName <hostname>
  User <username>
  ProxyJump <jump-host>            # if the cluster is behind a login node
  IdentityFile ~/.ssh/id_ed25519   # required for srunx
```

```bash
# Register the SSH profile (host comes from ~/.ssh/config)
srunx ssh profile add <profile-name> --ssh-host <ssh-host>

# Build the minimal fitting venv (numpy + scipy + mkgp) on the cluster scratch space
bash transport_study/datasets/gp_fitting/bootstrap_remote.sh <ssh-host> <path-to-venv>
```

Then, for example:

```bash
python transport_study/datasets/cli.py cmod \
    --data_assembly_dir /usr/local/mfe/ml_data_dump/POPSIM/popsim_studies/icddps \
    --cluster_profile <profile-name> \
    --cluster_partitions "sched_psfc_mit_r8@8:00:00,mit_preemptable@8:00:00@rocky8" \
    --cluster_remote_workdir <remote-workdir> \
    --cluster_venv <path-to-venv>/.venv
```

Notes:
- `--cluster_partitions` is an ordered preference list of `name@time_limit` or
  `name@time_limit@constraint` entries. Killed jobs are retried up to
  `--cluster_max_retries` (default 2) times, and both retries and jobs stuck PENDING past
  `--cluster_pending_timeout_s` (default 1800) move to the next partition in the list
  (wrapping around). Each time limit must not exceed the partition's MaxTime
  (`scontrol show partition <name> | grep MaxTime`).
- The fitting partitions should be CPU partitions since mkgp cannot use GPUs.
- Configure the following arguments according to your cluster's resources and user quotas
    - `--cluster_max_jobs` (default 8)
    - `--cluster_shots_per_batch` (default 10)
    - `--cluster_cpus_per_job` (default 32)
- For MAST running on the cluster itself, pass `--cluster_profile local` to submit sbatch jobs directly on the
  shared filesystem.
- For serial runs on a beefy machine, `--fit_workers N` parallelizes the fitting over N local processes.