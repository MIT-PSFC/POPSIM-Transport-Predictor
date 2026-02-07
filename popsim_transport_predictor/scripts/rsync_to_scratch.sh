#!/bin/bash

# This script places the completed datasets into a scratch directory for use in training and evaluation.
# For clusters (like ORCD and OMEGA), the scratch is on a much faster filesystem than the home directory
# this greatly speeds up training and evaluation for large datasets

# For this study we have C-Mod, D3D, and TCV datasets
# Define a .env_ds with something like
# export PTPS_CMOD_DATASET=/data/zkeith/cmod/dataset.zarr
# export PTPS_CMOD_SCRATCH_DIR=/scratch/zkeith/cmod
# And this script will do the rest

echo "Rsyncing datasets to scratch for training and evaluation"

if [ -z "$PTPS_CMOD_DATASET" ] || [ -z "$PTPS_CMOD_SCRATCH_DIR" ] ; then
    echo "PTPS_CMOD_DATASET or PTPS_CMOD_SCRATCH_DIR not set, skipping C-Mod rsync"
else
    echo "C-Mod"
    rsync -az --info=progress2 --info=name0 $PTPS_CMOD_DATASET $PTPS_CMOD_SCRATCH_DIR
fi

if [ -z "$PTPS_D3D_DATASET" ] || [ -z "$PTPS_D3D_SCRATCH_DIR" ] ; then
    echo "PTPS_D3D_DATASET or PTPS_D3D_SCRATCH_DIR not set, skipping D3D rsync"
else
    echo "D3D"
    rsync -az --info=progress2 --info=name0 $PTPS_D3D_DATASET $PTPS_D3D_SCRATCH_DIR
fi

if [ -z "$PTPS_TCV_DATASET" ] || [ -z "$PTPS_TCV_SCRATCH_DIR" ] ; then
    echo "PTPS_TCV_DATASET or PTPS_TCV_SCRATCH_DIR not set, skipping TCV rsync"
else
    echo "TCV"
    rsync -az --info=progress2 --info=name0 $PTPS_TCV_DATASET $PTPS_TCV_SCRATCH_DIR
fi
