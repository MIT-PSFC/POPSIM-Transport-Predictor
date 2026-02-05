import xarray as xr
import shutil

CHECK_FILE = "/fusion/projects/results/ida-results/HBP_database/IDA_199051_.cdf"

def read_hbp_data() -> xr.Dataset:
    """Read HBP data from CHECK_FILE and return as xarray Dataset."""
    ds = xr.open_dataset(CHECK_FILE)
    print(ds)
    return ds

if __name__ == "__main__":
    # read_hbp_data()
    ds_path = "/fusion/projects/disruption_warning/data/popsim/tpt_d3d_final/d3d_hp.zarr"
    ds = xr.open_zarr(ds_path)
    logical_size = ds.nbytes / (1024 ** 3)
    print(f"Dataset logical size: {logical_size:.2f} GB")

    total, used, free = shutil.disk_usage(ds_path)
    print(f"Disk space at {ds_path}:")
    print(f"  Total: {total // (2**30)} GiB")
    print(f"  Used: {used // (2**30)} GiB")
    print(f"  Free: {free // (2**30)} GiB")

    shot_ds = ds.sel(shot=199051)

    print(ds)