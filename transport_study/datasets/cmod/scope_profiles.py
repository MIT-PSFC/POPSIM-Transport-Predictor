import glob
import os

import fire
import matplotlib.pyplot as plt
import xarray as xr


def scope_shot(thomson_dir: str, profile_dir: str, figure_dir: str, shot: int):  # noqa: PLR0915
    """Make a plot for a single shot comparing Thomson and profile data."""
    os.makedirs(figure_dir, exist_ok=True)

    ds_thomson = xr.open_dataset(os.path.join(thomson_dir, f"{shot}.nc"))
    ds_profile = xr.open_dataset(os.path.join(profile_dir, f"{shot}.nc"))
    ds_shot = xr.merge([ds_thomson, ds_profile], compat="no_conflicts")
    # Remove the shot dimension since we're only looking at one shot
    ds_shot = ds_shot.isel(shot=0).drop_vars("shot")

    time = ds_shot["time"].values

    for t_idx in range(len(time)):
        fig, (ax_te, ax_ne) = plt.subplots(1, 2, figsize=(12, 6))

        ds_time = ds_shot.isel(time=t_idx)

        rho_gp = ds_time["gp_fit_rho"].values
        te_gp = ds_time["gp_fit_te"].values
        te_gp_err = ds_time["gp_fit_te_error"].values
        ne_gp = ds_time["gp_fit_ne"].values
        ne_gp_err = ds_time["gp_fit_ne_error"].values

        ds_core = ds_time.where(ds_time["ts_array"] == "core", drop=True)
        ds_edge = ds_time.where(ds_time["ts_array"] == "edge", drop=True)

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
        ax_te.plot(rho_gp, te_gp, label="GP Mean", color="blue")
        ax_te.fill_between(
            rho_gp,
            te_gp - 2 * te_gp_err,
            te_gp + 2 * te_gp_err,
            color="blue",
            alpha=0.2,
            label="GP 95% CI",
        )
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
        ax_ne.plot(rho_gp, ne_gp, label="GP Mean", color="red")
        ax_ne.fill_between(
            rho_gp,
            ne_gp - 2 * ne_gp_err,
            ne_gp + 2 * ne_gp_err,
            color="red",
            alpha=0.2,
            label="GP 95% CI",
        )
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

        fig.suptitle(f"C-Mod Shot {shot} Time {ds_time['time'].values:.3f} s")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        fig.savefig(os.path.join(figure_dir, f"time_{t_idx:03d}.png"))
        plt.close(fig)


def scope_all_shots(save_dir: str):
    """Plot all shots in the dataset."""

    thomson_dir = os.path.join(save_dir, "cmod_thomson_raw")
    profile_dir = os.path.join(save_dir, "cmod_profiles_raw")
    figure_dir = os.path.join(save_dir, "cmod_scope_profiles")
    os.makedirs(figure_dir, exist_ok=True)

    fitted_shot_data_files = glob.glob(os.path.join(profile_dir, "*.nc"))
    fitted_shots = [
        int(os.path.basename(f).split("/")[-1].split(".")[0])
        for f in fitted_shot_data_files
    ]

    for shot in fitted_shots:
        shot_dir = os.path.join(figure_dir, str(shot))
        # if not os.path.exists(figure_dir):
        scope_shot(thomson_dir, profile_dir, shot_dir, shot)


if __name__ == "__main__":
    fire.Fire(scope_all_shots)
