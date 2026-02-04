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