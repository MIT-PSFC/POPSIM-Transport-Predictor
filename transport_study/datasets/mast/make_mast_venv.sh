# Create .venv_mast for fetching MAST data from the STFC ECHO S3 open-access store.
# Requires s3fs + zarr>=3 which are not in the main project venv.
# Run from the repo root: bash transport_study/datasets/mast/make_mast_venv.sh

python3 -m venv .venv_mast
source .venv_mast/bin/activate
echo $(which python)
pip install --upgrade pip
pip install s3fs
pip install "zarr>=3.0"
pip install "fsspec[s3]>=2025.9.0"
pip install aiohttp
pip install xarray
pip install scipy
pip install numpy
pip install netcdf4
pip install loguru
pip install matplotlib
pip install dynaconf
pip install pydantic-settings
# popsim-public needed for transport_study.config; use --no-deps to avoid
# version conflicts with the deep jax/equinox dependency tree.
# The MAST data workflow only needs popsim.data path helpers, which fall
# back to /tmp if popsim is unavailable, so this step is optional.
pip install --no-deps -e submodules/popsim-public
# Install the transport study without pulling in the full dep tree
pip install --no-deps -e .
