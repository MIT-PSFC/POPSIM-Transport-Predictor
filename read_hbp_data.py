import xarray as xr

CHECK_FILE = "/fusion/projects/results/ida-results/HBP_database/IDA_199051_.cdf"

def read_hbp_data() -> xr.Dataset:
    """Read HBP data from CHECK_FILE and return as xarray Dataset."""
    ds = xr.open_dataset(CHECK_FILE)
    print(ds)
    return ds

if __name__ == "__main__":
    read_hbp_data()