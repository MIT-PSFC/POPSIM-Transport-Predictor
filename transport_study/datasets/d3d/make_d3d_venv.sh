# Make a venv for getting specifically the raw D3D data

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