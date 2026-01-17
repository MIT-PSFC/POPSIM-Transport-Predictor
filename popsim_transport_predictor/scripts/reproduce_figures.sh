export PTPS_CMOD_DATASET_PATH=/usr/local/mfe/ml_data_dump/studies/transport_predictor/cmod/cmod_200_processed.zarr
export PTPS_TCV_DATASET_PATH=/usr/local/mfe/ml_data_dump/TCV/zkeith/TCV_transport_dataset/full/dataset.nc

uv run popsim_transport_predictor/popsim_study/run_study.py \
--project_name popsim_transport_predictor \
--cmod_dataset_path /usr/local/mfe/ml_data_dump/studies/transport_predictor/cmod/cmod_200_processed.zarr \
--tcv_dataset_path /usr/local/mfe/ml_data_dump/TCV/zkeith/TCV_transport_dataset/full/dataset.nc \
--clean_models False \
--clean_results False \
--clean_figures False
