#!/usr/bin/env python3
"""
apply_x_shift_to_simulation_csv_smart.py

Apply the x-shift found by diagnose_x_shift_smart.py to a Fluent/simulation
pressure CSV and generate comparison outputs against the digitized pressure field.

This script expects the diagnostic summary written by diagnose_x_shift_smart.py:

    results/<PROJECT>/<FILE_FOLDER>/x_shift_alignment/x_shift_diagnostic_summary.csv

Sign convention
---------------
The shift is applied to the SIMULATION coordinate:

    x_sim_shifted_over_Lp = x_sim_over_Lp + best_shift_over_Lp

and, in physical coordinates:

    x_shifted = x_original + best_shift_physical_m

Therefore:
  best_shift_over_Lp > 0  moves the simulation pressure field downstream/right.
  best_shift_over_Lp < 0  moves the simulation pressure field upstream/left.

Outputs
-------
1. shifted simulation CSV with physical and normalized coordinates
2. shifted 2D comparison CSV on common normalized grid
3. shifted centerline comparison CSV
4. shifted 2D pressure/difference figure
5. shifted centerline comparison figure
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

# Diagnostic summary from diagnose_x_shift_smart.py
alignment_folder = REPO_DIR / "results" / PROJECT / FILE_FOLDER / "x_shift_alignment"
shift_summary_csv = alignment_folder / "x_shift_diagnostic_summary.csv"

# Output folder
output_folder = REPO_DIR / "results" / PROJECT / FILE_FOLDER / "x_shift_applied"
os.makedirs(output_folder, exist_ok=True)

# Output files
output_shifted_sim_csv = output_folder / f"{FILE_FOLDER}_panel_pressure_sim_x_shifted.csv"
output_comparison_csv = output_folder / "pressure_field_comparison_after_x_shift.csv"
output_centerline_csv = output_folder / "pressure_centerline_comparison_after_x_shift.csv"
output_2d_figure = output_folder / "pressure_comparison_2d_after_x_shift.eps"
output_centerline_figure = output_folder / "pressure_centerline_comparison_after_x_shift.eps"

# Simulation pressure column
sim_pressure_column = "pressure"

# Fallback normalization values only used if summary CSV does not contain them.
# Ideally these come directly from the diagnostic summary, so diagnosis and apply
# scripts use exactly the same x_min, Lp, y_center, and p_inf.
p_inf_fallback = 48400.0
Lp_manual_fallback = None
y_center_manual_fallback = None

# Common comparison grid
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

# Optional: also save points outside original normalized x in shifted CSV.
# The shifted field can extend outside [0,1]. This is normal. For comparison
# plots, interpolation is evaluated only on [x_norm_min, x_norm_max].
keep_all_shifted_points = True


# ==================================================
# Helper functions
# ==================================================
def require_columns(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def read_shift_summary(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find shift summary CSV:\n  {path}\n"
            "Run diagnose_x_shift_smart.py first, or update shift_summary_csv."
        )

    summary = pd.read_csv(path)
    if len(summary) < 1:
        raise ValueError(f"Shift summary CSV is empty: {path}")

    row = summary.iloc[0]

    required = ["best_shift_over_Lp"]
    for c in required:
        if c not in summary.columns:
            raise ValueError(f"Shift summary is missing required column: {c}")

    out = {
        "best_shift_over_Lp": float(row["best_shift_over_Lp"]),
    }

    # Preferred fields written by diagnosis script.
    optional_fields = {
        "best_shift_physical_m": None,
        "x_min_original_m": None,
        "x_max_original_m": None,
        "Lp_m": None,
        "y_center_m": None,
        "p_inf_Pa": None,
    }

    for key in optional_fields:
        if key in summary.columns and pd.notna(row[key]):
            out[key] = float(row[key])
        else:
            out[key] = optional_fields[key]

    return out


def normalization_from_summary_or_data(sim_raw, shift_info):
    """Return x_min, x_max, Lp, y_center, p_inf using summary when available."""
    x_min_data = float(sim_raw["x"].min())
    x_max_data = float(sim_raw["x"].max())

    x_min = shift_info.get("x_min_original_m")
    x_max = shift_info.get("x_max_original_m")
    Lp = shift_info.get("Lp_m")
    y_center = shift_info.get("y_center_m")
    p_inf = shift_info.get("p_inf_Pa")

    if x_min is None or not np.isfinite(x_min):
        x_min = x_min_data
    if x_max is None or not np.isfinite(x_max):
        x_max = x_max_data

    if Lp is None or not np.isfinite(Lp) or Lp <= 0:
        if Lp_manual_fallback is None:
            Lp = x_max_data - x_min_data
        else:
            Lp = float(Lp_manual_fallback)

    if Lp <= 0:
        raise ValueError("Invalid Lp. Check x coordinates or summary CSV.")

    if y_center is None or not np.isfinite(y_center):
        if y_center_manual_fallback is None:
            y_center = 0.5 * (float(sim_raw["y"].min()) + float(sim_raw["y"].max()))
        else:
            y_center = float(y_center_manual_fallback)

    if p_inf is None or not np.isfinite(p_inf) or p_inf <= 0:
        p_inf = float(p_inf_fallback)

    return {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "Lp": float(Lp),
        "y_center": float(y_center),
        "p_inf": float(p_inf),
    }


def make_shifted_simulation_dataframe(sim_raw, norm_info, shift_info):
    """Create shifted simulation dataframe preserving original x,y,z,pressure."""
    df = sim_raw.copy()

    x_min = norm_info["x_min"]
    Lp = norm_info["Lp"]
    y_center = norm_info["y_center"]
    p_inf = norm_info["p_inf"]

    best_shift_over_Lp = shift_info["best_shift_over_Lp"]
    best_shift_physical_m = shift_info.get("best_shift_physical_m")
    if best_shift_physical_m is None or not np.isfinite(best_shift_physical_m):
        best_shift_physical_m = best_shift_over_Lp * Lp

    df["x_original"] = df["x"]
    df["x_shift_m"] = best_shift_physical_m
    df["x_shift_over_Lp"] = best_shift_over_Lp

    df["x_over_Lp_original"] = (df["x_original"] - x_min) / Lp
    df["x_over_Lp_shifted"] = df["x_over_Lp_original"] + best_shift_over_Lp
    df["x_shifted"] = df["x_original"] + best_shift_physical_m

    df["y_over_Lp"] = (df["y"] - y_center) / Lp
    df["p_over_pinf"] = df[sim_pressure_column] / p_inf

    # Make the main x column the shifted physical coordinate for downstream use.
    # The original x is preserved as x_original.
    df["x"] = df["x_shifted"]

    return df


def collapse_for_interpolation_shifted(df):
    """Average duplicate shifted x-y locations for grid interpolation."""
    return (
        df[["x_over_Lp_shifted", "y_over_Lp", "p_over_pinf"]]
        .groupby(["x_over_Lp_shifted", "y_over_Lp"], as_index=False)
        .mean()
    )


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


def print_info(title, dct):
    print(f"\n{title}")
    for k, v in dct.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.8e}")
        else:
            print(f"  {k}: {v}")


# ==================================================
# Main
# ==================================================
def main():
    print(f"Reading shift summary: {shift_summary_csv}")
    shift_info = read_shift_summary(shift_summary_csv)
    print_info("Shift information", shift_info)

    print(f"\nReading simulation CSV: {sim_csv}")
    sim_raw = pd.read_csv(sim_csv)
    require_columns(sim_raw, ["x", "y", "z", sim_pressure_column], "Simulation CSV")

    norm_info = normalization_from_summary_or_data(sim_raw, shift_info)
    print_info("Normalization used for shifted simulation", norm_info)

    shifted_df = make_shifted_simulation_dataframe(sim_raw, norm_info, shift_info)

    if not keep_all_shifted_points:
        shifted_df = shifted_df[
            (shifted_df["x_over_Lp_shifted"] >= x_norm_min)
            & (shifted_df["x_over_Lp_shifted"] <= x_norm_max)
        ].copy()

    shifted_df.to_csv(output_shifted_sim_csv, index=False)
    print(f"\nSaved shifted simulation CSV to:\n  {output_shifted_sim_csv}")

    # Prepare simulation for interpolation
    sim_interp_df = collapse_for_interpolation_shifted(shifted_df)

    print(f"\nReading digitized CSV: {dig_csv}")
    dig_df = pd.read_csv(dig_csv)
    require_columns(
        dig_df,
        ["x_over_Lp", "y_over_Lp", "pressure_ratio", "pressure_ratio_uncertainty"],
        "Digitized CSV",
    )
    dig_df = dig_df[
        ["x_over_Lp", "y_over_Lp", "pressure_ratio", "pressure_ratio_uncertainty"]
    ].copy()

    # Common grid in final normalized coordinates
    x_grid = np.linspace(x_norm_min, x_norm_max, nx)
    y_grid = np.linspace(y_norm_min, y_norm_max, ny)
    X, Y = np.meshgrid(x_grid, y_grid)

    P_sim = interpolate_to_grid(
        sim_interp_df,
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

    Delta = P_sim - P_dig
    AbsDelta = np.abs(Delta)
    GapBeyondUncertainty = np.maximum(AbsDelta - U_dig, 0.0)
    SignedGapBeyondUncertainty = np.sign(Delta) * GapBeyondUncertainty
    DeltaOverUncertainty = Delta / np.maximum(U_dig, 1e-12)

    print("\nComparison statistics after x-shift:")
    print(f"  Mean signed difference: {np.nanmean(Delta):.6f}")
    print(f"  Mean absolute difference: {np.nanmean(AbsDelta):.6f}")
    print(f"  Max absolute difference: {np.nanmax(AbsDelta):.6f}")
    print(f"  Mean gap beyond uncertainty: {np.nanmean(GapBeyondUncertainty):.6f}")
    print(f"  Max gap beyond uncertainty: {np.nanmax(GapBeyondUncertainty):.6f}")
    print(f"  Fraction with |difference| > uncertainty: {np.mean(AbsDelta > U_dig):.3f}")

    comparison_df = pd.DataFrame({
        "x_over_Lp": X.ravel(),
        "y_over_Lp": Y.ravel(),
        "p_shifted_sim_over_pinf": P_sim.ravel(),
        "p_digitized_over_pinf": P_dig.ravel(),
        "digitization_uncertainty": U_dig.ravel(),
        "difference_shifted_sim_minus_digitized": Delta.ravel(),
        "absolute_difference": AbsDelta.ravel(),
        "gap_beyond_digitization_uncertainty": GapBeyondUncertainty.ravel(),
        "signed_gap_beyond_digitization_uncertainty": SignedGapBeyondUncertainty.ravel(),
        "difference_over_uncertainty": DeltaOverUncertainty.ravel(),
        "applied_shift_over_Lp": shift_info["best_shift_over_Lp"],
        "applied_shift_physical_m": shift_info.get("best_shift_physical_m", shift_info["best_shift_over_Lp"] * norm_info["Lp"]),
    })
    comparison_df.to_csv(output_comparison_csv, index=False)
    print(f"Saved shifted comparison CSV to:\n  {output_comparison_csv}")

    # 2D plots
    if difference_limit_manual is None:
        diff_lim = np.nanpercentile(np.abs(Delta), 99)
        if not np.isfinite(diff_lim) or diff_lim <= 0:
            diff_lim = 1.0
    else:
        diff_lim = float(difference_limit_manual)

    gap_lim = np.nanpercentile(np.abs(SignedGapBeyondUncertainty), 99)
    if not np.isfinite(gap_lim) or gap_lim <= 0:
        gap_lim = diff_lim

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), constrained_layout=True)

    c1 = axes[0, 0].contourf(
        X, Y, P_sim,
        levels=np.linspace(pressure_plot_min, pressure_plot_max, 101),
        cmap=cmap_pressure,
        vmin=pressure_plot_min,
        vmax=pressure_plot_max,
        extend="both",
    )
    fig.colorbar(c1, ax=axes[0, 0])
    axes[0, 0].set_title(r"Shifted simulation: $P/P_\infty$")
    axes[0, 0].set_xlabel(r"$x/L_p$", fontsize=16)
    axes[0, 0].set_ylabel(r"$y/L_p$", fontsize=16)

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
    axes[0, 1].set_xlabel(r"$x/L_p$", fontsize=16)
    axes[0, 1].set_ylabel(r"$y/L_p$", fontsize=16)

    c3 = axes[1, 0].contourf(
        X, Y, Delta,
        levels=np.linspace(-diff_lim, diff_lim, 101),
        cmap=cmap_difference,
        vmin=-diff_lim,
        vmax=diff_lim,
        extend="both",
    )
    fig.colorbar(c3, ax=axes[1, 0])
    axes[1, 0].set_title(r"Difference: shifted simulation $-$ digitized")
    axes[1, 0].set_xlabel(r"$x/L_p$", fontsize=16)
    axes[1, 0].set_ylabel(r"$y/L_p$", fontsize=16)

    c4 = axes[1, 1].contourf(
        X, Y, SignedGapBeyondUncertainty,
        levels=np.linspace(-gap_lim, gap_lim, 101),
        cmap=cmap_difference,
        vmin=-gap_lim,
        vmax=gap_lim,
        extend="both",
    )
    fig.colorbar(c4, ax=axes[1, 1])
    axes[1, 1].set_title(r"Signed gap beyond digitization uncertainty")
    axes[1, 1].set_xlabel(r"$x/L_p$", fontsize=16)
    axes[1, 1].set_ylabel(r"$y/L_p$", fontsize=16)

    for ax in axes.ravel():
        ax.set_xlim(x_norm_min, x_norm_max)
        ax.set_ylim(y_norm_min, y_norm_max)
        ax.set_aspect("auto")

    plt.savefig(output_2d_figure, format='eps', dpi=300, bbox_inches="tight")
    print(f"Saved shifted 2D comparison figure to:\n  {output_2d_figure}")
    plt.show()

    # Centerline comparison
    x_line, p_sim_line = centerline_from_grid(
        X, Y, P_sim,
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

    delta_line = p_sim_line - p_dig_line
    gap_line = np.maximum(np.abs(delta_line) - u_dig_line, 0.0)

    centerline_df = pd.DataFrame({
        "x_over_Lp": x_line,
        "p_shifted_sim_over_pinf": p_sim_line,
        "p_digitized_over_pinf": p_dig_line,
        "digitization_uncertainty": u_dig_line,
        "difference_shifted_sim_minus_digitized": delta_line,
        "absolute_difference": np.abs(delta_line),
        "gap_beyond_digitization_uncertainty": gap_line,
        "applied_shift_over_Lp": shift_info["best_shift_over_Lp"],
        "applied_shift_physical_m": shift_info.get("best_shift_physical_m", shift_info["best_shift_over_Lp"] * norm_info["Lp"]),
    })
    centerline_df.to_csv(output_centerline_csv, index=False)
    print(f"Saved shifted centerline comparison CSV to:\n  {output_centerline_csv}")

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
        alpha=0.35,
        linewidth=0,
        label="Digitized uncertainty band",
        zorder=1,
    )
    ax1.plot(
        x_line,
        p_sim_line,
        linewidth=2.4,
        label="Shifted simulation",
        zorder=3,
    )
    ax1.plot(
        x_line,
        p_dig_line,
        linewidth=2.4,
        linestyle="--",
        label="Digitized paper",
        zorder=4,
    )
    ax1.set_ylabel(r"$P/P_\infty$", fontsize=16)
    ax1.set_title("Centerline comparison after x-shift")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    p_min_plot = np.nanmin([np.nanmin(p_sim_line), np.nanmin(p_dig_line - u_dig_line)])
    p_max_plot = np.nanmax([np.nanmax(p_sim_line), np.nanmax(p_dig_line + u_dig_line)])
    padding = 0.08 * (p_max_plot - p_min_plot)
    if np.isfinite(padding) and padding > 0:
        ax1.set_ylim(p_min_plot - padding, p_max_plot + padding)

    ax2.plot(
        x_line,
        delta_line,
        linewidth=2.0,
        label=r"Shifted sim $-$ digitized",
    )
    ax2.fill_between(
        x_line,
        -u_dig_line,
        u_dig_line,
        alpha=0.35,
        linewidth=0,
        label=r"$\pm$ digitization uncertainty",
    )
    ax2.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    ax2.set_xlabel(r"Normalized coordinate, $x/L_p$", fontsize=16)
    ax2.set_ylabel(r"$\Delta P/P_\infty$", fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    diff_abs_max = np.nanmax(np.abs(delta_line))
    unc_abs_max = np.nanmax(u_dig_line)
    diff_ylim = 1.15 * max(diff_abs_max, unc_abs_max)
    if np.isfinite(diff_ylim) and diff_ylim > 0:
        ax2.set_ylim(-diff_ylim, diff_ylim)

    plt.tight_layout()
    plt.savefig(output_centerline_figure, format='eps', dpi=300, bbox_inches="tight")
    print(f"Saved shifted centerline figure to:\n  {output_centerline_figure}")
    plt.show()

    print("\nDone.")
    print("\nKey output for Fluent/postprocessing use:")
    print(f"  {output_shifted_sim_csv}")
    print("\nMain shifted coordinate columns:")
    print("  x                          = shifted physical x-coordinate")
    print("  x_original                 = original physical x-coordinate")
    print("  x_over_Lp_shifted          = shifted normalized coordinate")
    print("  x_over_Lp_original         = original normalized coordinate")
    print("  y_over_Lp                  = normalized y-coordinate")
    print("  p_over_pinf                = pressure / p_inf")


if __name__ == "__main__":
    main()
