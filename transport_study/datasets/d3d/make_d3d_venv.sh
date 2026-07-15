# Make a .venv_d3d for getting specifically the raw D3D data
# This installation will likely spit out a LOT of errors, but it still works I swear
# Required until MDSplus on OMEGA supports numpy >= 2

source .venv/bin/activate
echo $(python --version)
python -m venv .venv_d3d
deactivate
source .venv_d3d/bin/activate
echo $(which python)
pip install --upgrade pip
pip install numpy==1.26.4
pip install -e submodules/disruption-py
pip install mkgp  # not used for D3D, but the dataset CLI imports the C-Mod workflow
pip install -e .
pip install numpy==1.26.4
pip install "jax==0.4.35" "jaxlib==0.4.35"  # last jax compatible with numpy < 2
pip install pytest  # for the live test in tests/datasets/test_d3d_dataset.py
pip install fsspec
pip install joblib
pip install bottleneck
pip install "numcodecs<0.16"
pip install zarr==2.18.3
