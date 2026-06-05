from pathlib import Path

import fire
import matplotlib.pyplot as plt
import xarray as xr


def scope_shot(thomson_dir: Path | str, profile_dir: Path | str, figure_dir: Path | str, shot: int):
    """Make a plot for a single shot comparing Thomson and profile data."""
    Path(figure_dir).mkdir(parents=True, exist_ok=True)

    ds_thomson = xr.open_dataset(Path(thomson_dir) / f"{shot}.nc")
    ds_profile = xr.open_dataset(Path(profile_dir) / f"{shot}.nc")
    # Remove the shot dimension since we're only looking at one shot
    ds_thomson = ds_thomson.isel(shot=0).drop_vars("shot")
    ds_profile = ds_profile.isel(shot=0).drop_vars("shot")

    ts_times = ds_thomson.time.values

    for t_idx, time in enumerate(ts_times):
        fig, (ax_te, ax_ne) = plt.subplots(1, 2, figsize=(12, 6))

        ds_thomson_time = ds_thomson.sel(time=time)
        if "time" in ds_profile.indexes:
            ds_profile_time = ds_profile.sel(time=time, method="nearest")
        else:
            time_coord = ds_profile["time"]
            time_dim = time_coord.dims[0]
            nearest_idx = abs(time_coord - time).argmin(dim=time_dim)
            ds_profile_time = ds_profile.isel({time_dim: nearest_idx})

        psi_gp = ds_profile_time["psi"].values
        te_gp = ds_profile_time["Te_keV_psi"].values
        ne_gp = ds_profile_time["ne20_psi"].values

        ds_core = ds_thomson_time.where(ds_thomson_time["ts_array"] == "core", drop=True)
        ds_edge = ds_thomson_time.where(ds_thomson_time["ts_array"] == "edge", drop=True)

        rho_ts_core = ds_core["ts_channel_rho"].values
        te_ts_core = ds_core["ts_channel_te"].values
        te_ts_err_core = ds_core["ts_channel_te_error"].values
        ne_ts_core = ds_core["ts_channel_ne"].values / 1e20
        ne_ts_err_core = ds_core["ts_channel_ne_error"].values / 1e20

        rho_ts_edge = ds_edge["ts_channel_rho"].values
        te_ts_edge = ds_edge["ts_channel_te"].values
        te_ts_err_edge = ds_edge["ts_channel_te_error"].values
        ne_ts_edge = ds_edge["ts_channel_ne"].values / 1e20
        ne_ts_err_edge = ds_edge["ts_channel_ne_error"].values / 1e20

        # Te profile, TS data and GP fit
        ax_te.plot()
        ax_te.plot(psi_gp, te_gp, label="GP Mean", color="blue")
        ax_te.errorbar(
            rho_ts_core,
            te_ts_core,
            yerr=te_ts_err_core,
            fmt="o",
            color="black",
            label="TS Core",
        )
        ax_te.errorbar(
            rho_ts_edge,
            te_ts_edge,
            yerr=te_ts_err_edge,
            fmt="o",
            color="orange",
            label="TS Edge",
        )

        ax_te.set_xlabel("rho")
        ax_te.set_ylabel("Te [keV]")
        ax_te.set_title("Te Profile")
        ax_te.set_ylim(bottom=0)
        ax_te.grid()
        ax_te.legend()

        # ne profiles, TS data and GP fit
        ax_ne.plot()
        ax_ne.plot(psi_gp, ne_gp, label="GP Mean", color="red")
        ax_ne.errorbar(
            rho_ts_core,
            ne_ts_core,
            yerr=ne_ts_err_core,
            fmt="o",
            color="black",
            label="TS Core",
        )
        ax_ne.errorbar(
            rho_ts_edge,
            ne_ts_edge,
            yerr=ne_ts_err_edge,
            fmt="o",
            color="orange",
            label="TS Edge",
        )

        ax_ne.set_xlabel("rho")
        ax_ne.set_ylabel("ne [10^20 m^-3]")
        ax_ne.set_title("ne Profile")
        ax_ne.set_ylim(bottom=0)
        ax_ne.grid()
        ax_ne.legend()

        fig.suptitle(f"C-Mod Shot {shot} Time {ds_thomson_time['time'].values:.3f} s")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        fig.savefig(Path(figure_dir) / f"time_{t_idx:03d}.png")
        plt.close(fig)


def scope_all_shots_freestyle(profile_dir: Path | str):
    """Plot all shots in the dataset."""

    thomson_dir = "/usr/local/mfe/ml_data_dump/POPSIM/old_studies/transport_predictor/cmod_100/cmod_thomson_raw"
    figure_dir = Path(profile_dir) / "profile_scopes"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fitted_shot_data_files = list(Path(profile_dir).glob("*.nc"))
    fitted_shots = [int(f.stem) for f in fitted_shot_data_files]

    for shot in fitted_shots:
        shot_dir = figure_dir / str(shot)
        scope_shot(thomson_dir, profile_dir, shot_dir, shot)


def scope_all_shots(save_dir: Path | str):
    """Plot all shots in the dataset."""

    thomson_dir = Path(save_dir) / "cmod_thomson_raw"
    profile_dir = Path(save_dir) / "cmod_profiles_raw"
    figure_dir = Path(save_dir) / "cmod_scope_profiles"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fitted_shot_data_files = list(Path(profile_dir).glob("*.nc"))
    fitted_shots = [int(f.stem) for f in fitted_shot_data_files]

    for shot in fitted_shots:
        shot_dir = figure_dir / str(shot)
        scope_shot(thomson_dir, profile_dir, shot_dir, shot)


if __name__ == "__main__":
    fire.Fire(
        {
            "scope_all_shots": scope_all_shots,
            "scope_all_shots_freestyle": scope_all_shots_freestyle,
            "scope_shot": scope_shot,
        }
    )
