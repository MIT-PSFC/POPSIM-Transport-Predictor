source .venv/bin/activate
export MPLBACKEND=Agg

python -m transport_study.datasets.cli mast \
    --shotlist_file transport_study/datasets/mast/mast_shotlist \
    --mode raw \
    --data_assembly_dir scratch/datasets \
    --max_num_shots 400

# python -m transport_study.datasets.cli mast \
#     --shotlist_file transport_study/datasets/mast/mast_shotlist \
#     --mode process \
#     --data_assembly_dir scratch/datasets \
#     --max_num_shots 400