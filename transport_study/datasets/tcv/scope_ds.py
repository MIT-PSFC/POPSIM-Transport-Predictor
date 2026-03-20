import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

BACKGROUND_COLOR = "#2F2F2F"
FACE_COLOR = "#1A1A1A"
TEXT_COLOR = "white"

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18

DS_CANDIDATES = {
    # Can't be any of these because don't have enough signals for transport predictor
    "Taue_10": "/orcd/nese/psfc/001/allenw/ml_data_dump/TCV/zkeith/TCV_TauE_dataset/tcv_dataset_10_shots.nc",
    "Taue_151": "/orcd/nese/psfc/001/allenw/ml_data_dump/TCV/zkeith/TCV_TauE_dataset/tcv_dataset_151_shots.nc",
    "Taue_389": "/orcd/nese/psfc/001/allenw/ml_data_dump/TCV/zkeith/TCV_TauE_dataset/tcv_dataset_389_shots.nc",
    "Transport_full": "/orcd/nese/psfc/001/allenw/ml_data_dump/TCV/zkeith/TCV_transport_dataset/full/dataset.nc",
    "Transport_max": "/orcd/nese/psfc/001/allenw/ml_data_dump/TCV/zkeith/TCV_transport_dataset/max/dataset.nc",
    "Transport_test": "/orcd/nese/psfc/001/allenw/ml_data_dump/TCV/zkeith/TCV_transport_dataset/test/dataset.nc",
}

DS_PATH_SRC = DS_CANDIDATES["Transport_full"]
DS_PATH_TCV = "/home/zkeith/proj/POPSIM_dirs/POPSIM_transport_checkpoint/popsim/data/tcv/transport_sample.nc"


def ds_sizes():
    for name, path in DS_CANDIDATES.items():
        print("================================")
        print(f"{name}: {path}")
        print(f"Number of shots: {int(xr.open_dataset(path).sizes['shot'])}")


def ds_target_info(ds_target: str):
    ds_path = DS_CANDIDATES.get(ds_target)

    ds = xr.open_dataset(ds_path)
    shotlist = ds["shot"].values.tolist()
    data_vars = list(ds.data_vars)

    return shotlist, data_vars


def transfer_ds():  # noqa: PLR0915 PLR0912
    # I know the dataset at "/orcd/nese/psfc/001/allenw/ml_data_dump/TCV/zkeith/TCV_transport_dataset/full/dataset.nc" is right
    ds_path_src = "/orcd/nese/psfc/001/allenw/ml_data_dump/TCV/zkeith/TCV_transport_dataset/full/dataset.nc"

    ds = xr.open_dataset(ds_path_src)
    ds = ds.rename(
        {
            "RMAG": "R0",
            "KAPPA": "kappa",
            "DELTA_TOP": "delta_top",
            "DELTA_BOTTOM": "delta_bottom",
            "P_LH": "LH_transition_threshold_MW",
        }
    )

    # Conversions
    ds["B0"] = np.abs(ds["BZERO"])
    ds["Ip_MA"] = np.abs(ds["I_P"]) * 1e-6
    ds["P_oh_MW"] = ds["POHM"] * 1e-6
    ds["P_rad_MW"] = ds["PradBulk"] * 1e-6
    ds["ne20_line_avg"] = ds["NEavg"] * 1e-20
    ds["Wtot_MJ"] = ds["Wtot"] * 1e-6
    ds["ne20_rho"] = ds["Ne_rho"] * 1e-20
    ds["Te_keV_rho"] = ds["Te_rho"] * 1e-3

    # Simple fringe-jump correction for ne20_line_avg
    # Detect large step changes and remove the offset for the remainder of the trace
    ne_values = ds["ne20_line_avg"].values
    time_values = ds["time"].values
    for shot_idx in range(ne_values.shape[0]):
        trace = ne_values[shot_idx].copy()
        if np.all(np.isnan(trace)):
            continue
        diff = np.diff(trace)
        median_abs_diff = np.nanmedian(np.abs(diff))
        jump_threshold = max(0.1, 5.0 * median_abs_diff)

        # Cumulative offset after each detected jump
        offset = 0.0
        corrected = trace.copy()
        for i in range(1, trace.size):
            if np.isnan(trace[i - 1]) or np.isnan(trace[i]):
                corrected[i] = trace[i] - offset
                continue
            step = trace[i] - trace[i - 1]
            if np.abs(step) >= jump_threshold:
                offset += step
            corrected[i] = trace[i] - offset

        ne_values[shot_idx] = corrected
    ds["ne20_line_avg"] = (ds["ne20_line_avg"].dims, ne_values)

    # If the signal is not present, create it as zeros up to shape of Ip_MA
    # But where Ip_MA is NaN, keep it NaN
    ds["P_NBI_MW"] = xr.where(ds["Ip_MA"].notnull(), ds["NBI"].fillna(0.0), np.nan)
    ds["P_ECRH_MW"] = xr.where(ds["Ip_MA"].notnull(), ds["ECRH"].fillna(0.0), np.nan)
    ds["P_LH_MW"] = xr.where(ds["Ip_MA"].notnull(), 0.0, np.nan)
    ds["P_ICRF_MW"] = xr.where(ds["Ip_MA"].notnull(), 0.0, np.nan)

    # Set the region 20ms before Ip transitions to NaN to NaN for all signals to avoid training on data during disruption
    for shot in ds["shot"].values:
        shot_idx = int(np.where(ds["shot"].values == shot)[0])
        ip_values = ds["Ip_MA"].isel(shot=shot_idx).values
        time_values = ds["time"].values

        # Find where Ip_MA transitions to NaN (goes from not NaN to NaN)
        ip_notnan = ~np.isnan(ip_values)
        transitions = np.where(~ip_notnan & np.roll(ip_notnan, 1))[0]

        for trans_idx in transitions:
            trans_time = time_values[trans_idx]
            # Create mask for times from 20ms before transition
            mask = time_values >= trans_time - 0.02
            mask_indices = np.where(mask)[0]

            # Apply mask to all data variables that have shot and time dimensions
            for var in ds.data_vars:
                if "shot" in ds[var].dims and "time" in ds[var].dims:
                    ds[var].values[shot_idx, mask_indices] = np.nan

    # Reject unrealistic data points and data points at low temperature and density, when Thomson scattering can be unreliable
    # Only keep data where NEavg is above 0.1e20
    ds["ne20_line_avg"] = xr.where(
        ds["ne20_line_avg"] > 0.1, ds["ne20_line_avg"], np.nan
    )

    # Only keep data where Wtot is above 1e3 J
    ds["Wtot_MJ"] = xr.where(ds["Wtot_MJ"] > 1e-3, ds["Wtot_MJ"], np.nan)

    # Only keep data where Te_rho at rho=1.0 is less than 0.25 keV and above 0.0
    ds["Te_keV_rho"] = xr.where(
        (ds["Te_keV_rho"].sel(rho=1.0) < 0.25) & (ds["Te_keV_rho"].sel(rho=1.0) > 0.0),
        ds["Te_keV_rho"],
        np.nan,
    )

    # # Only keep data where ne20_rho at rho=1.0 is less than 0.4e20 and above 0..0
    ds["ne20_rho"] = xr.where(
        (ds["ne20_rho"].sel(rho=1.0) < 0.4) & (ds["ne20_rho"].sel(rho=1.0) > 0.0),
        ds["ne20_rho"],
        np.nan,
    )

    # Drop old names
    ds = ds.drop_vars(
        [
            "BZERO",
            "I_P",
            "POHM",
            "PradBulk",
            "NEavg",
            "Wtot",
            "Te_rho",
            "Ne_rho",
            "NBI",
            "ECRH",
        ]
    )

    # Drop shots where ne20_line_avg still has a median value over 2e21
    shots_before = ds.shot
    ds = ds.where(ds["ne20_line_avg"].median(dim="time") <= 20, drop=True)
    dropped_shots = set(shots_before.values) - set(ds.shot.values)
    if dropped_shots:
        print(f"Dropped shots with broken high density: {dropped_shots}")

    # Drop shots where radiated power exceeds 300% of input power for 20% of the valid time points
    shots_before = ds.shot
    power_in = (
        ds["P_NBI_MW"]
        + ds["P_ECRH_MW"]
        + ds["P_LH_MW"]
        + ds["P_ICRF_MW"]
        + ds["P_oh_MW"]
    )
    power_rad_frac = ds["P_rad_MW"] / power_in
    ds = ds.where(
        (power_rad_frac >= 3.0).sum(dim="time") / ds["Ip_MA"].notnull().sum(dim="time")
        <= 0.2,
        drop=True,
    )
    dropped_shots = set(shots_before.values) - set(ds.shot.values)
    if dropped_shots:
        print(f"Dropped shots with excessive radiated power: {dropped_shots}")

    # Drop shots where P_rad_MW is NaN for all time points
    shots_before = ds.shot
    ds = ds.where(~np.isnan(ds["P_rad_MW"]).all(dim="time"), drop=True)
    dropped_shots = set(shots_before.values) - set(ds.shot.values)
    if dropped_shots:
        print(f"Dropped shots with all NaN radiated power: {dropped_shots}")

    # Drop shots where `ne20_line_avg` is NaN for more than 10% of valid time points
    shots_before = ds.shot
    ds = ds.where(
        ds["ne20_line_avg"].notnull().sum(dim="time")
        / ds["Ip_MA"].notnull().sum(dim="time")
        >= 0.9,
        drop=True,
    )
    dropped_shots = set(shots_before.values) - set(ds.shot.values)
    if dropped_shots:
        print(f"Dropped shots with excessive NaN density: {dropped_shots}")

    # Drop additional shots that I say are sus
    sus_shots = [
        85117,  # Missing input power signal?
        83412,  # Missing input power signal?
        # 76567, # Radiated power too high the whole time, kind of an outlier though
    ]

    # Drop shots where profile data is all NaN
    shots_before = ds.shot
    null_time_mask = ds["ne20_rho"].isnull().sum(dim="rho") > 0
    null_shot_mask = null_time_mask.all(dim="time")
    ds = ds.where(~null_shot_mask, drop=True)
    if dropped_shots:
        print(f"Dropped shots with all NaN profile data: {dropped_shots}")

    ds = ds.where(~ds["shot"].isin(sus_shots), drop=True)

    print("Shots remaining after cleaning:", ds.sizes["shot"])

    # Save as netcdf xarray in new location
    ds.to_netcdf(DS_PATH_TCV)


def ds_time_plot(fig_dir: str, num_shots: int = 2):  # noqa: PLR0915
    """Plot time traces of signals from the dataset"""
    ds = xr.open_dataset(DS_PATH_TCV)

    # Remove existing directory if present
    if os.path.exists(fig_dir):
        shutil.rmtree(fig_dir)
    os.makedirs(fig_dir)

    # Compute global y-limits across all shots for consistent axes
    ylim_ip = (0, float(np.nanmax(np.abs(ds["Ip_MA"].values))) * 1.1)
    ylim_wtot = (0, float(np.nanmax(ds["Wtot_MJ"].values)) * 1.1)

    power_signals = [
        "P_oh_MW",
        "P_rad_MW",
        "P_NBI_MW",
        "P_ECRH_MW",
        "P_LH_MW",
        "P_ICRF_MW",
    ]
    power_max = max(float(np.nanmax(ds[sig].values)) for sig in power_signals)
    lh_thresh_max = float(np.nanmax(ds["LH_transition_threshold_MW"].values / 1e6))
    ylim_power = (0, max(power_max, lh_thresh_max) * 1.1)

    ylim_ne = (0, float(np.nanmax(ds["ne20_line_avg"].values)) * 1.1)

    shape_signals = ["a_minor", "kappa", "delta_top", "delta_bot"]
    shape_min = min(float(np.nanmin(ds[sig].values)) for sig in shape_signals)
    shape_max = max(float(np.nanmax(ds[sig].values)) for sig in shape_signals)
    ylim_shape = (
        shape_min * 0.9 if shape_min > 0 else shape_min * 1.1,
        shape_max * 1.1,
    )

    for shot in ds["shot"].data[:num_shots]:
        shot_ds = ds.sel(shot=shot)

        fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
        fig.patch.set_facecolor(BACKGROUND_COLOR)

        fig.suptitle(f"TCV Shot {shot}", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)

        # Ip and Wtot
        ax_ip = axes[0]
        # Put Ip on the left y axis and Wtot on the right y axis
        ax_ip.plot(shot_ds["time"], shot_ds["Ip_MA"], label="Ip [MA]", color="cyan")
        ax_ip.set_ylabel("Ip [MA]", fontsize=LABEL_FONTSIZE, color="cyan")
        ax_ip.set_ylim(ylim_ip)
        ax_wtot = ax_ip.twinx()
        ax_wtot.plot(
            shot_ds["time"], shot_ds["Wtot_MJ"], label="Wtot [MJ]", color="red"
        )
        ax_wtot.set_ylabel("Wtot [MJ]", fontsize=LABEL_FONTSIZE, color="red")
        ax_wtot.set_ylim(ylim_wtot)
        ax_wtot.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)

        # powers
        ax_power = axes[1]
        ax_power.plot(
            shot_ds["time"], shot_ds["P_oh_MW"], label="P_oh [MW]", color="orange"
        )
        ax_power.plot(
            shot_ds["time"], shot_ds["P_rad_MW"], label="P_rad [MW]", color="red"
        )
        ax_power.plot(
            shot_ds["time"], shot_ds["P_NBI_MW"], label="P_NBI [MW]", color="cyan"
        )
        ax_power.plot(
            shot_ds["time"], shot_ds["P_ECRH_MW"], label="P_ECRH [MW]", color="lime"
        )
        ax_power.plot(
            shot_ds["time"],
            shot_ds["LH_transition_threshold_MW"] / 1e6,
            label="LH_Thresh [MW]",
            color="white",
            linestyle="--",
        )
        ax_power.set_ylabel("Power [MW]", fontsize=LABEL_FONTSIZE, color="white")
        ax_power.set_ylim(ylim_power)
        ax_power.legend(
            fontsize=LEGEND_FONTSIZE,
            facecolor=BACKGROUND_COLOR,
            edgecolor=BACKGROUND_COLOR,
            loc="upper left",
        )

        # density and gas valve
        ax_ne = axes[2]
        ax_ne.plot(
            shot_ds["time"],
            shot_ds["ne20_line_avg"],
            label="ne20_line_avg [m^-3]",
            color="white",
        )
        ax_ne.set_ylabel("ne20_line_avg [m^-3]", fontsize=LABEL_FONTSIZE, color="white")
        ax_ne.set_ylim(ylim_ne)

        # Shaping
        ax_shape = axes[3]
        ax_shape.plot(shot_ds["time"], shot_ds["a_minor"], label="a_minor", color="red")
        ax_shape.plot(shot_ds["time"], shot_ds["kappa"], label="kappa", color="yellow")
        ax_shape.plot(
            shot_ds["time"], shot_ds["delta_top"], label="delta_top", color="lime"
        )
        ax_shape.plot(
            shot_ds["time"],
            shot_ds["delta_bot"],
            label="delta_bot",
            color="green",
        )
        ax_shape.set_ylabel("Shaping", fontsize=LABEL_FONTSIZE, color="white")
        ax_shape.set_ylim(ylim_shape)
        ax_shape.set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE, color="white")
        ax_shape.legend(
            fontsize=LEGEND_FONTSIZE,
            facecolor=BACKGROUND_COLOR,
            edgecolor=BACKGROUND_COLOR,
            loc="upper left",
        )

        for ax in axes:
            ax.set_facecolor(FACE_COLOR)
            ax.grid(True, color="gray", linestyle="--", linewidth=0.1)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
            try:
                for text in ax.get_legend().get_texts():
                    text.set_color(TEXT_COLOR)
            except AttributeError:
                pass

        fig.tight_layout()
        fig.savefig(f"{fig_dir}/{shot}.png")
        plt.close(fig)


def ds_profile_plot(profile_dir: str, num_shots: int = 2):
    """Plot individual profile signals from the dataset"""
    ds = xr.open_dataset(DS_PATH_TCV)

    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)

    # Compute global y-limits across all shots for consistent axes
    ylim_ne = (0, float(np.nanmax(ds["ne20_rho"].values)) * 1.1)
    ylim_te = (0, float(np.nanmax(ds["Te_keV_rho"].values)) * 1.1)

    for shot in ds["shot"].data[:num_shots]:
        shot_ds = ds.sel(shot=shot)
        shot_dir = os.path.join(profile_dir, str(shot))
        if not os.path.exists(shot_dir):
            os.makedirs(shot_dir)

        # Get rho coordinates
        rho = shot_ds["rho"].values

        # Track previous profiles to avoid plotting duplicates from rectilinear interpolation
        prev_ne_profile = None
        prev_te_profile = None

        for _, time in enumerate(shot_ds["time"].values):
            time_ds = shot_ds.sel(time=time)

            ne_profile = time_ds["ne20_rho"].values
            te_profile = time_ds["Te_keV_rho"].values

            # Skip if profiles are identical to previous (rectilinear interpolation duplicates)
            ne_same = prev_ne_profile is not None and np.allclose(
                ne_profile, prev_ne_profile, equal_nan=True
            )
            te_same = prev_te_profile is not None and np.allclose(
                te_profile, prev_te_profile, equal_nan=True
            )
            if ne_same and te_same:
                continue

            prev_ne_profile = ne_profile.copy()
            prev_te_profile = te_profile.copy()

            # Skip if profiles are all NaN
            if np.all(np.isnan(ne_profile)) and np.all(np.isnan(te_profile)):
                continue

            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            fig.patch.set_facecolor(BACKGROUND_COLOR)

            fig.suptitle(
                f"TCV Shot {shot} @ t={time:.3f}s",
                fontsize=TITLE_FONTSIZE,
                color=TEXT_COLOR,
            )

            # Density profile
            ax_ne = axes[0]
            ax_ne.plot(rho, ne_profile, label="ne20", color="cyan", linewidth=2)
            ax_ne.set_ylabel(
                r"$n_e$ [$10^{20}$ m$^{-3}$]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR
            )
            ax_ne.set_ylim(ylim_ne)
            ax_ne.legend(
                fontsize=LEGEND_FONTSIZE,
                facecolor=BACKGROUND_COLOR,
                edgecolor=BACKGROUND_COLOR,
                loc="upper right",
            )

            # Temperature profile
            ax_te = axes[1]
            ax_te.plot(rho, te_profile, label="Te", color="red", linewidth=2)
            ax_te.set_ylabel(r"$T_e$ [keV]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
            ax_te.set_xlabel(r"$\rho$", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
            ax_te.set_ylim(ylim_te)
            ax_te.legend(
                fontsize=LEGEND_FONTSIZE,
                facecolor=BACKGROUND_COLOR,
                edgecolor=BACKGROUND_COLOR,
                loc="upper right",
            )

            for ax in axes:
                ax.set_facecolor(FACE_COLOR)
                ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
                ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
                ax.set_xlim(0, 1)
                for spine in ax.spines.values():
                    spine.set_color(TEXT_COLOR)
                try:
                    for text in ax.get_legend().get_texts():
                        text.set_color(TEXT_COLOR)
                except AttributeError:
                    pass

            fig.tight_layout()
            fig.savefig(os.path.join(shot_dir, f"t{time:.3f}.png"))
            plt.close(fig)


def ds_profile_time_plot(fig_dir: str, num_shots: int = 2):
    """Plot 2D heatmaps of density and temperature profiles over time.

    X-axis: rho (radial coordinate)
    Y-axis: time
    Color: density/temperature values
    Timesteps with any NaN values are colored bright pink.
    """
    ds = xr.open_dataset(DS_PATH_TCV)

    # Remove existing directory if present
    if os.path.exists(fig_dir):
        shutil.rmtree(fig_dir)
    os.makedirs(fig_dir)

    for shot in ds["shot"].data[:num_shots]:
        shot_ds = ds.sel(shot=shot)

        rho = shot_ds["rho"].values
        time = shot_ds["time"].values

        # Extract 2D arrays for density and temperature
        ne_data = shot_ds["ne20_rho"].values.T  # shape: (rho, time) - transposed
        te_data = shot_ds["Te_keV_rho"].values.T  # shape: (rho, time) - transposed

        # Create masks for timesteps with NaN values
        ne_nan_mask = np.isnan(ne_data).any(axis=0)  # True if any NaN in that timestep
        te_nan_mask = np.isnan(te_data).any(axis=0)  # True if any NaN in that timestep

        fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        fig.suptitle(
            f"TCV Shot {shot} Profile Heatmaps",
            fontsize=TITLE_FONTSIZE,
            color=TEXT_COLOR,
        )

        ax_ne = axes[0]
        ax_te = axes[1]

        # Plot density heatmap
        ne_plot_data = ne_data.copy()
        ne_plot_data[:, ne_nan_mask] = (
            np.nan
        )  # Set NaN timesteps to NaN for proper masking
        im_ne = ax_ne.imshow(
            ne_plot_data,
            cmap="viridis",
            aspect="auto",
            origin="lower",
            extent=[time.min(), time.max(), rho.min(), rho.max()],
        )

        # Overlay bright pink for NaN timesteps
        if np.any(ne_nan_mask):
            ne_pink_data = np.full_like(ne_data, np.nan)
            ne_pink_data[:, ne_nan_mask] = 1.0  # Use constant value for bright color
            ax_ne.imshow(
                ne_pink_data,
                cmap="Reds",
                aspect="auto",
                origin="lower",
                alpha=0.8,
                extent=[time.min(), time.max(), rho.min(), rho.max()],
            )

        ax_ne.set_ylabel(r"$\rho$", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        ax_ne.set_title(
            r"$n_e$ [$10^{20}$ m$^{-3}$]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR
        )
        cbar_ne = plt.colorbar(im_ne, ax=ax_ne)
        cbar_ne.ax.tick_params(labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)

        # Plot temperature heatmap
        te_plot_data = te_data.copy()
        te_plot_data[:, te_nan_mask] = (
            np.nan
        )  # Set NaN timesteps to NaN for proper masking
        im_te = ax_te.imshow(
            te_plot_data,
            cmap="plasma",
            aspect="auto",
            origin="lower",
            extent=[time.min(), time.max(), rho.min(), rho.max()],
        )

        # Overlay bright pink for NaN timesteps
        if np.any(te_nan_mask):
            te_pink_data = np.full_like(te_data, np.nan)
            te_pink_data[:, te_nan_mask] = 1.0  # Use constant value for bright color
            ax_te.imshow(
                te_pink_data,
                cmap="Reds",
                aspect="auto",
                origin="lower",
                alpha=0.8,
                extent=[time.min(), time.max(), rho.min(), rho.max()],
            )

        ax_te.set_ylabel(r"$\rho$", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        ax_te.set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        ax_te.set_title(r"$T_e$ [keV]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        cbar_te = plt.colorbar(im_te, ax=ax_te)
        cbar_te.ax.tick_params(labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)

        for ax in axes:
            ax.set_facecolor(FACE_COLOR)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
            ax.set_xlim(0, 1)
            for spine in ax.spines.values():
                spine.set_color(TEXT_COLOR)

        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{shot}.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    scope_dir = "scratch/scoping"
    num_shots = 999
    ds_time_plot(os.path.join(scope_dir, "time_plots"), num_shots=num_shots)
    ds_profile_plot(os.path.join(scope_dir, "profile_plots"), num_shots=num_shots)
    ds_profile_time_plot(
        os.path.join(scope_dir, "profile_heatmaps"), num_shots=num_shots
    )
