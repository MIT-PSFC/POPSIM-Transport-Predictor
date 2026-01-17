import xarray as xr


def performance_extrapolation_plot(
    save_path: str,
    ds_list: list[xr.Dataset],
    labels: list[str],
    performance_metric: str = "performance",
    x_var: str = "Ip_MA",
    y_var: str = "Wtot_MJ",
):
    """
    Generate performance extrapolation plots
    """
