#!/usr/bin/env bash
# Launches the raw TCV dataset build under nohup.
# Reads the DEFUSE exports and the LIUQE MEQ databases from the paths in datasets/tcv/config.toml.
# Sequential and I/O bound, about 10 s and 250 MB of memory per shot.
set -euo pipefail

WORKSPACE="/home/zkeith/proj/POPSIM_dirs/POPSIM-Transport-Predictor"
LOG="${WORKSPACE}/transport_study/datasets/tcv/raw_tcv_dataset.log"
DATA_ASSEMBLY_DIR="/usr/local/mfe/ml_data_dump/POPSIM/popsim_studies/aps2026/datasets/iteration_2"

export JAX_PLATFORMS="cpu"
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK="1"

cd "${WORKSPACE}"
nohup uv run --frozen python -m transport_study.datasets.cli \
    tcv "${DATA_ASSEMBLY_DIR}" \
    --ds_name tcv \
    --mode raw \
    > "${LOG}" 2>&1 &

echo "Launched raw TCV dataset job, PID $!"
echo "Logs: ${LOG}"
