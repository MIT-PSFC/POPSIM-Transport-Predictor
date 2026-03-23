# Make a .venv_d3d for getting specifically the raw D3D data
# This installation will likely spit out a LOT of errors, but it still works I swear
# It's just a hack until disruption-py updates to numpy >= 2 on OMEGA

source .venv/bin/activate
echo $(python --version)
python -m venv .venv_d3d
deactivate
source .venv_d3d/bin/activate
echo $(which python)
pip install --upgrade pip
pip install numpy==1.26.4
pip install -e submodules/disruption-py
pip install -e submodules/gptools
pip install -e submodules/popsim
pip install -e .
pip install numpy==1.26.4
pip install freeqdsk
pip install fsspec
pip install joblib
pip install ray
pip install pyspark
pip install bottleneck
pip install "numcodecs<0.16"
pip install zarr==2.18.3
pip install -e submodules/toksearch
