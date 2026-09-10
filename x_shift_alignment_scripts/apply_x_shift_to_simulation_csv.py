#!/usr/bin/env python3
"""
apply_x_shift_to_simulation_csv.py

Reads the best x-shift estimated by diagnose_x_shift_least_error.py,
applies it to the simulation CSV, and writes a shifted simulation CSV.

Sign convention
---------------
From the diagnostic script:

    x_sim_shifted_over_Lp = x_sim_over_Lp + best_shift_over_Lp

In physical coordinates, this script writes:

    x_shifted = x_original + best_shift_over_Lp * Lp

Outputs
-------
1. shifted simulation CSV with physical and normalized coordinates
2. shifted 2D comparison CSV on normalized coordinates
3. shifted 2D comparison plot
4. shifted centerline comparison plot and CSV
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# ==================================================
# User inputs
# ==================================================
FILE_FOLDER = "7500iter_SST"
REPO_DIR = Path(__file__).parent.parent
PROJECT = "2023_Aviation"

SIM_CSV = "panel_pressure_sim.csv"
DIG_CSV = "digitized_pressure_field_2023Avia.csv"

project_dir = REPO_DIR / "csv_files" / PROJECT
sim_csv = project_dir / f"{FILE_FOLDER}_{SIM_CSV}"
dig_csv = project_dir / DIG_CSV

alignment_folder = REPO_DIR / "results" / PROJECT / FILE_FOLDER / "x_shift_alignment"
shift_summary_csv = alignment_folder / "x_shift_diagnostic_summary.csv"

output_folder = REPO_DIR / "results" / PROJECT / FILE_FOLDER / "x_shifted_simulation"
os.makedirs(output_folder, exist_ok=True)

# Simulation pressure column
sim_pressure_column = "pressure"

# If True, use shift from diagnostic summary.
# If False, use manual_shift_over_Lp below.
read_shift_from_diagnostic_summary = True
manual_shift_over_Lp = 0.0

# If None, Lp and x_min are read from diagnostic summary when available.
# Only set manually if you are not using diagnostic summary.
Lp_manual = None
x_min_manual = None
y_center_manual = None
p_inf_manual = 48400.0

# Common comparison grid for plotting after shift
nx = 400
ny = 200

x_norm_min = 0.0
x_norm_max = 1.0
y_norm_min = -0.25
y_norm_max = 0.25

# Centerline extraction
centerline_y = 0.0
centerline_band_half_width = 0.0025
use_band_average_for_centerline = True

# Plot settings
cmap_pressure = "jet"
cmap_difference = "bwr"
pressure_plot_min = 0.7
pressure_plot_max = 1.7
difference_limit_manual = None


# ==================================================
# Helper functions
# ==================================================
def require_columns(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def interpolate_to_grid(df, x_col, y_col, value_col, X, Y, method="linear"):
    points = (df[x_col].values, df[y_col].values)
    values = df[value_col].values
    Z = griddata(points, values, (X, Y), method=method)
    Z_nearest = griddata(points, values, (X, Y), method="nearest")
    Z = np.where(np.isnan(Z), Z_nearest, Z)
    return Z


def centerline_from_grid(X, Y, Z, y0=0.0, band_half_width=0.0025, use_band=True):
    x_1d = X[0, :]

    if use_band:
        mask = np.abs(Y - y0) <= band_half_width
        if not np.any(mask):
            raise ValueError("No grid points found inside centerline band.")

        z_line = np.full_like(x_1d, np.nan, dtype=float)
        for j in range(len(x_1d)):
            vals = Z[:, j][mask[:, j]]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                z_line[j] = np.mean(vals)
    else:
        row_idx = np.argmin(np.abs(Y[:, 0] - y0))
        z_line = Z[row_idx, :]

    return x_1d, z_line


# ==================================================
# Main
# ==================================================
def main():
    if read_shift_from_diagnostic_summary:
        if not shift_summary_csv.exists():
            raise FileNotFoundError(
                f"Could not find shift summary CSV:\n{shift_summary_csv}\n"
                "Run diagnose_x_shift_least_error.py first, or set "
                "read_shift_from_diagnostic_summary = False and provide manual_shift_over_Lp."
            )

        summary = pd.read_csv(shift_summary_csv).iloc[0]
        shift_over_Lp = float(summary["best_shift_over_Lp"])
        Lp = float(summary["Lp_m"])
        x_min = float(summary["x_min_original_m"])
        y_center = float(summary["y_center_m"])
        p_inf = float(summary["p_inf_Pa"])

    else:
        shift_over_Lp = float(manual_shift_over_Lp)

        sim_tmp = pd.read_csv(sim_csv)
        require_columns(sim_tmp, ["x", "y", "z", sim_pressure_column], "Simulation CSV")

        if Lp_manual is None:
            Lp = float(sim_tmp["x"].max() - sim_tmp["x"].min())
        else:
            Lp = float(Lp_manual)

        if x_min_manual is None:
            x_min = float(sim_tmp["x"].min())
        else:
            x_min = float(x_min_manual)

        if y_center_manual is None:
            y_center = 0.5 * float(sim_tmp["y"].min() + sim_tmp["y"].max())
        else:
            y_center = float(y_center_manual)

        p_inf = float(p_inf_manual)

    shift_physical_m = shift_over_Lp * Lp

    print("\nApplying x-shift to simulation CSV")
    print(f"  shift_over_Lp: {shift_over_Lp:+.8f}")
    print(f"  shift_physical_m: {shift_physical_m:+.8e}")
    print("  x_shifted = x_original + shift_physical_m")

    # Read raw simulation and preserve all rows/columns.
    sim_raw = pd.read_csv(sim_csv)
    require_columns(sim_raw, ["x", "y", "z", sim_pressure_column], "Simulation CSV")

    shifted = sim_raw.copy()
    shifted["x_original"] = shifted["x"]
    shifted["x_shift_m"] = shift_physical_m
    shifted["x"] = shifted["x_original"] + shift_physical_m

    # Add normalized coordinates for plotting and downstream comparisons.
    # Normalization is done using the original panel reference, not recomputed after shifting.
    shifted["x_over_Lp_original"] = (shifted["x_original"] - x_min) / Lp
    shifted["x_over_Lp_shifted"] = shifted["x_over_Lp_original"] + shift_over_Lp
    shifted["y_over_Lp"] = (shifted["y"] - y_center) / Lp
    shifted["p_over_pinf"] = shifted[sim_pressure_column] / p_inf

    shifted_csv = output_folder / f"{FILE_FOLDER}_panel_pressure_sim_x_shifted.csv"
    shifted.to_csv(shifted_csv, index=False)
    print(f"\nSaved shifted simulation CSV:\n  {shifted_csv}")

    # For grid interpolation, average duplicate shifted x-y locations.
    sim_plot_df = (
        shifted[["x_over_Lp_shifted", "y_over_Lp", "z", sim_pressure_column, "p_over_pinf"]]
        .groupby(["x_over_Lp_shifted", "y_over_Lp"], as_index=False)
        .agg({"p_over_pinf": "mean", sim_pressure_column: "mean", "z": "mean"})
    )

    dig_df = pd.read_csv(dig_csv)
    require_columns(
        dig_df,
        ["x_over_Lp", "y_over_Lp", "pressure_ratio", "pressure_ratio_uncertainty"],
        "Digitized CSV",
    )

    x_grid = np.linspace(x_norm_min, x_norm_max, nx)
    y_grid = np.linspace(y_norm_min, y_norm_max, ny)
    X, Y = np.meshgrid(x_grid, y_grid)

    P_sim_shifted = interpolate_to_grid(
        sim_plot_df,
        x_col="x_over_Lp_shifted",
        y_col="y_over_Lp",
        value_col="p_over_pinf",
        X=X,
        Y=Y,
    )
    P_dig = interpolate_to_grid(
        dig_df,
        x_col="x_over_Lp",
        y_col="y_over_Lp",
        value_col="pressure_ratio",
        X=X,
        Y=Y,
    )
    U_dig = interpolate_to_grid(
        dig_df,
        x_col="x_over_Lp",
        y_col="y_over_Lp",
        value_col="pressure_ratio_uncertainty",
        X=X,
        Y=Y,
    )

    Delta = P_sim_shifted - P_dig
    AbsDelta = np.abs(Delta)
    GapBeyondUncertainty = np.maximum(AbsDelta - U_dig, 0.0)
    SignedGapBeyondUncertainty = np.sign(Delta) * GapBeyondUncertainty

    shifted_comparison_df = pd.DataFrame({
        "x_over_Lp": X.ravel(),
        "y_over_Lp": Y.ravel(),
        "p_sim_shifted_over_pinf": P_sim_shifted.ravel(),
        "p_digitized_over_pinf": P_dig.ravel(),
        "digitization_uncertainty": U_dig.ravel(),
        "difference_shifted_sim_minus_digitized": Delta.ravel(),
        "absolute_difference": AbsDelta.ravel(),
        "gap_beyond_digitization_uncertainty": GapBeyondUncertainty.ravel(),
        "signed_gap_beyond_digitization_uncertainty": SignedGapBeyondUncertainty.ravel(),
    })

    shifted_comparison_csv = output_folder / "pressure_field_comparison_after_x_shift.csv"
    shifted_comparison_df.to_csv(shifted_comparison_csv, index=False)
    print(f"Saved shifted comparison grid CSV:\n  {shifted_comparison_csv}")

    # 2D comparison plot
    if difference_limit_manual is None:
        diff_lim = np.nanpercentile(np.abs(Delta), 99)
        if diff_lim <= 0:
            diff_lim = 1.0
    else:
        diff_lim = difference_limit_manual

    gap_lim = np.nanpercentile(np.abs(SignedGapBeyondUncertainty), 99)
    if gap_lim <= 0:
        gap_lim = diff_lim

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), constrained_layout=True)

    c1 = axes[0, 0].contourf(
        X, Y, P_sim_shifted,
        levels=np.linspace(pressure_plot_min, pressure_plot_max, 101),
        cmap=cmap_pressure,
        vmin=pressure_plot_min,
        vmax=pressure_plot_max,
        extend="both",
    )
    fig.colorbar(c1, ax=axes[0, 0])
    axes[0, 0].set_title(r"Shifted simulation: $P/P_\infty$")
    axes[0, 0].set_xlabel(r"$x/L_p$")
    axes[0, 0].set_ylabel(r"$y/L_p$")

    c2 = axes[0, 1].contourf(
        X, Y, P_dig,
        levels=np.linspace(pressure_plot_min, pressure_plot_max, 101),
        cmap=cmap_pressure,
        vmin=pressure_plot_min,
        vmax=pressure_plot_max,
        extend="both",
    )
    fig.colorbar(c2, ax=axes[0, 1])
    axes[0, 1].set_title(r"Digitized paper: $P/P_\infty$")
    axes[0, 1].set_xlabel(r"$x/L_p$")
    axes[0, 1].set_ylabel(r"$y/L_p$")

    c3 = axes[1, 0].contourf(
        X, Y, Delta,
        levels=np.linspace(-diff_lim, diff_lim, 101),
        cmap=cmap_difference,
        vmin=-diff_lim,
        vmax=diff_lim,
        extend="both",
    )
    fig.colorbar(c3, ax=axes[1, 0])
    axes[1, 0].set_title(r"Difference after shift: simulation $-$ digitized")
    axes[1, 0].set_xlabel(r"$x/L_p$")
    axes[1, 0].set_ylabel(r"$y/L_p$")

    c4 = axes[1, 1].contourf(
        X, Y, SignedGapBeyondUncertainty,
        levels=np.linspace(-gap_lim, gap_lim, 101),
        cmap=cmap_difference,
        vmin=-gap_lim,
        vmax=gap_lim,
        extend="both",
    )
    fig.colorbar(c4, ax=axes[1, 1])
    axes[1, 1].set_title(r"Signed gap beyond uncertainty after shift")
    axes[1, 1].set_xlabel(r"$x/L_p$")
    axes[1, 1].set_ylabel(r"$y/L_p$")

    for ax in axes.ravel():
        ax.set_xlim(x_norm_min, x_norm_max)
        ax.set_ylim(y_norm_min, y_norm_max)
        ax.set_aspect("auto")

    fig.suptitle(f"Applied simulation x-shift = {shift_over_Lp:+.5f} $L_p$", y=1.02)
    plot2d_path = output_folder / "pressure_comparison_2d_after_x_shift.png"
    plt.savefig(plot2d_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Centerline plot
    x_line, p_sim_line = centerline_from_grid(
        X, Y, P_sim_shifted,
        y0=centerline_y,
        band_half_width=centerline_band_half_width,
        use_band=use_band_average_for_centerline,
    )
    _, p_dig_line = centerline_from_grid(
        X, Y, P_dig,
        y0=centerline_y,
        band_half_width=centerline_band_half_width,
        use_band=use_band_average_for_centerline,
    )
    _, u_dig_line = centerline_from_grid(
        X, Y, U_dig,
        y0=centerline_y,
        band_half_width=centerline_band_half_width,
        use_band=use_band_average_for_centerline,
    )

    centerline_df = pd.DataFrame({
        "x_over_Lp": x_line,
        "p_sim_shifted_over_pinf": p_sim_line,
        "p_digitized_over_pinf": p_dig_line,
        "digitization_uncertainty": u_dig_line,
        "difference_shifted_sim_minus_digitized": p_sim_line - p_dig_line,
        "absolute_difference": np.abs(p_sim_line - p_dig_line),
        "gap_beyond_digitization_uncertainty": np.maximum(np.abs(p_sim_line - p_dig_line) - u_dig_line, 0.0),
    })

    centerline_csv = output_folder / "pressure_centerline_comparison_after_x_shift.csv"
    centerline_df.to_csv(centerline_csv, index=False)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(8.5, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4]},
    )
    ax1, ax2 = axes

    ax1.fill_between(
        x_line,
        p_dig_line - u_dig_line,
        p_dig_line + u_dig_line,
        alpha=0.30,
        linewidth=0,
        label="Digitized uncertainty band",
    )
    ax1.plot(x_line, p_sim_line, linewidth=2.4, label="Shifted simulation")
    ax1.plot(x_line, p_dig_line, linewidth=2.4, linestyle="--", label="Digitized paper")
    ax1.set_ylabel(r"$P/P_\infty$")
    ax1.set_title(r"Centerline comparison after x-shift")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    delta_line = p_sim_line - p_dig_line
    ax2.plot(x_line, delta_line, linewidth=2.0, label=r"Shifted sim $-$ digitized")
    ax2.fill_between(
        x_line,
        -u_dig_line,
        u_dig_line,
        alpha=0.30,
        linewidth=0,
        label=r"$\pm$ digitization uncertainty",
    )
    ax2.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    ax2.set_xlabel(r"Normalized coordinate, $x/L_p$")
    ax2.set_ylabel(r"$\Delta P/P_\infty$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    centerline_plot = output_folder / "pressure_centerline_comparison_after_x_shift.png"
    plt.tight_layout()
    plt.savefig(centerline_plot, dpi=300, bbox_inches="tight")
    plt.show()

    print("\nShifted comparison statistics:")
    print(f"  mean signed difference: {np.nanmean(Delta):+.6f}")
    print(f"  mean absolute difference: {np.nanmean(AbsDelta):.6f}")
    print(f"  max absolute difference:  {np.nanmax(AbsDelta):.6f}")
    print(f"  fraction |difference| > digitization uncertainty: {np.mean(AbsDelta > U_dig):.3f}")

    print("\nSaved plots and line CSV:")
    print(f"  {plot2d_path}")
    print(f"  {centerline_csv}")
    print(f"  {centerline_plot}")


if __name__ == "__main__":
    main()
