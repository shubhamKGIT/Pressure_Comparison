#!/usr/bin/env python3
"""
diagnose_x_shift_least_error.py

Estimate the streamwise x-shift needed to align a Fluent/simulation pressure field
with a digitized paper pressure field.

Sign convention
---------------
The fitted shift is applied to the SIMULATION normalized coordinate:

    x_sim_shifted_over_Lp = x_sim_over_Lp + best_shift_over_Lp

Therefore:
  best_shift_over_Lp > 0  moves the simulation pressure field downstream/right.
  best_shift_over_Lp < 0  moves the simulation pressure field upstream/left.

Outputs
-------
1. x_shift_diagnostic_summary.csv
2. x_shift_diagnostic_curve.csv
3. x_shift_centerline_before_after.png
4. x_shift_rmse_curve.png

The summary CSV is intended to be read by apply_x_shift_to_simulation_csv.py.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d
from scipy.optimize import minimize_scalar


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

output_folder = REPO_DIR / "results" / PROJECT / FILE_FOLDER / "x_shift_alignment"
os.makedirs(output_folder, exist_ok=True)

# Simulation pressure column
sim_pressure_column = "pressure"

# Reference pressure used to normalize simulation pressure
p_inf = 48400.0

# Optional manual panel length.
# If None, Lp = x_max - x_min from simulation data.
Lp_manual = None

# Optional manual y-center.
# If None, y_center = midpoint of simulation y range.
y_center_manual = None

# Common comparison grid for extracting centerline
nx = 600
ny = 250

x_norm_min = 0.0
x_norm_max = 1.0
y_norm_min = -0.25
y_norm_max = 0.25

# Centerline extraction
centerline_y = 0.0
centerline_band_half_width = 0.0025
use_band_average_for_centerline = True

# Shift search range in normalized x/Lp units.
# Example: +/- 0.15 means the code tests moving the simulation by +/- 15% of panel length.
shift_min_over_Lp = -0.20
shift_max_over_Lp = 0.20

# Restrict least-error fitting to a useful x-window.
# This avoids endpoints and regions where interpolation/extrapolation dominates.
fit_x_min = 0.05
fit_x_max = 0.95

# Optional: weight the error by digitization uncertainty.
# If True, the objective is mean(((sim-dig)/uncertainty)^2).
# If False, the objective is ordinary RMSE in P/P_inf.
use_uncertainty_weighting = False

# Optional: remove a constant pressure bias before fitting shift.
# Usually keep False if the offset is physical. Set True if you only want shape alignment.
allow_constant_pressure_bias = False

# Plot settings
pressure_plot_min = None
pressure_plot_max = None


# ==================================================
# Helper functions
# ==================================================
def require_columns(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def normalize_simulation_coordinates(df, p_inf, Lp_manual=None, y_center_manual=None):
    x_min = df["x"].min()
    x_max = df["x"].max()

    if Lp_manual is None:
        Lp = x_max - x_min
    else:
        Lp = float(Lp_manual)

    if Lp <= 0:
        raise ValueError("Invalid Lp. Check x coordinates or set Lp_manual.")

    if y_center_manual is None:
        y_center = 0.5 * (df["y"].min() + df["y"].max())
    else:
        y_center = float(y_center_manual)

    out = df.copy()
    out["x_over_Lp"] = (out["x"] - x_min) / Lp
    out["y_over_Lp"] = (out["y"] - y_center) / Lp
    out["p_over_pinf"] = out[sim_pressure_column] / p_inf

    info = {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "Lp": float(Lp),
        "y_center": float(y_center),
        "p_inf": float(p_inf),
    }
    return out, info


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


def safe_interp_function(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)

    # Remove duplicate x values by averaging
    tmp = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False).mean()
    x = tmp["x"].to_numpy()
    y = tmp["y"].to_numpy()

    if len(x) < 5:
        raise ValueError("Too few finite points for interpolation.")

    return interp1d(
        x,
        y,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )


def objective_for_shift(
    shift,
    x_eval,
    sim_interp,
    dig_interp,
    unc_interp,
    use_uncertainty_weighting=False,
    allow_constant_pressure_bias=False,
):
    # Since x_sim_shifted = x_sim + shift, the simulation value at final coordinate x
    # comes from the original simulation coordinate x - shift.
    p_sim_shifted = sim_interp(x_eval - shift)
    p_dig = dig_interp(x_eval)
    u_dig = unc_interp(x_eval)

    mask = np.isfinite(p_sim_shifted) & np.isfinite(p_dig)

    if use_uncertainty_weighting:
        mask = mask & np.isfinite(u_dig) & (u_dig > 0)

    if np.count_nonzero(mask) < 20:
        return np.inf

    residual = p_sim_shifted[mask] - p_dig[mask]

    if allow_constant_pressure_bias:
        if use_uncertainty_weighting:
            w = 1.0 / np.maximum(u_dig[mask], 1e-12) ** 2
            bias = np.sum(w * residual) / np.sum(w)
        else:
            bias = np.nanmean(residual)
        residual = residual - bias

    if use_uncertainty_weighting:
        residual = residual / np.maximum(u_dig[mask], 1e-12)

    return float(np.sqrt(np.nanmean(residual**2)))


# ==================================================
# Main
# ==================================================
def main():
    print(f"Reading simulation CSV: {sim_csv}")
    sim_raw = pd.read_csv(sim_csv)
    require_columns(sim_raw, ["x", "y", "z", sim_pressure_column], "Simulation CSV")

    # For interpolation, average duplicate x-y locations.
    sim_xy = (
        sim_raw[["x", "y", "z", sim_pressure_column]]
        .groupby(["x", "y"], as_index=False)
        .agg({sim_pressure_column: "mean", "z": "mean"})
    )

    sim_df, sim_info = normalize_simulation_coordinates(
        sim_xy,
        p_inf=p_inf,
        Lp_manual=Lp_manual,
        y_center_manual=y_center_manual,
    )

    print("\nSimulation normalization:")
    for k, v in sim_info.items():
        print(f"  {k}: {v:.8e}")

    print(f"\nReading digitized CSV: {dig_csv}")
    dig_df = pd.read_csv(dig_csv)
    require_columns(
        dig_df,
        ["x_over_Lp", "y_over_Lp", "pressure_ratio", "pressure_ratio_uncertainty"],
        "Digitized CSV",
    )

    # Common grid
    x_grid = np.linspace(x_norm_min, x_norm_max, nx)
    y_grid = np.linspace(y_norm_min, y_norm_max, ny)
    X, Y = np.meshgrid(x_grid, y_grid)

    P_sim = interpolate_to_grid(
        sim_df,
        x_col="x_over_Lp",
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

    sim_interp = safe_interp_function(x_line, p_sim_line)
    dig_interp = safe_interp_function(x_line, p_dig_line)
    unc_interp = safe_interp_function(x_line, u_dig_line)

    x_eval = np.linspace(fit_x_min, fit_x_max, 1000)

    rmse_before = objective_for_shift(
        0.0,
        x_eval,
        sim_interp,
        dig_interp,
        unc_interp,
        use_uncertainty_weighting=use_uncertainty_weighting,
        allow_constant_pressure_bias=allow_constant_pressure_bias,
    )

    result = minimize_scalar(
        objective_for_shift,
        bounds=(shift_min_over_Lp, shift_max_over_Lp),
        method="bounded",
        options={"xatol": 1e-6},
        args=(
            x_eval,
            sim_interp,
            dig_interp,
            unc_interp,
            use_uncertainty_weighting,
            allow_constant_pressure_bias,
        ),
    )

    best_shift = float(result.x)
    rmse_after = float(result.fun)
    best_shift_physical = best_shift * sim_info["Lp"]

    print("\nBest x-shift fit:")
    print(f"  best_shift_over_Lp: {best_shift:+.8f}")
    print(f"  best_shift_physical: {best_shift_physical:+.8e} m")
    print(f"  RMSE before shift: {rmse_before:.8f}")
    print(f"  RMSE after shift:  {rmse_after:.8f}")
    print("\nSign convention:")
    print("  x_sim_shifted_over_Lp = x_sim_over_Lp + best_shift_over_Lp")
    print("  positive shift moves simulation downstream/right on normalized plots.")

    # Diagnostic curve
    shifts = np.linspace(shift_min_over_Lp, shift_max_over_Lp, 401)
    rmse_vals = np.array([
        objective_for_shift(
            s,
            x_eval,
            sim_interp,
            dig_interp,
            unc_interp,
            use_uncertainty_weighting=use_uncertainty_weighting,
            allow_constant_pressure_bias=allow_constant_pressure_bias,
        )
        for s in shifts
    ])

    diagnostic_curve = pd.DataFrame({
        "shift_over_Lp": shifts,
        "shift_physical_m": shifts * sim_info["Lp"],
        "fit_error": rmse_vals,
    })
    diagnostic_curve_path = output_folder / "x_shift_diagnostic_curve.csv"
    diagnostic_curve.to_csv(diagnostic_curve_path, index=False)

    summary = pd.DataFrame([{
        "best_shift_over_Lp": best_shift,
        "best_shift_physical_m": best_shift_physical,
        "rmse_before_shift": rmse_before,
        "rmse_after_shift": rmse_after,
        "fit_x_min": fit_x_min,
        "fit_x_max": fit_x_max,
        "centerline_y_over_Lp": centerline_y,
        "centerline_band_half_width_over_Lp": centerline_band_half_width,
        "use_uncertainty_weighting": use_uncertainty_weighting,
        "allow_constant_pressure_bias": allow_constant_pressure_bias,
        "x_min_original_m": sim_info["x_min"],
        "x_max_original_m": sim_info["x_max"],
        "Lp_m": sim_info["Lp"],
        "y_center_m": sim_info["y_center"],
        "p_inf_Pa": sim_info["p_inf"],
    }])
    summary_path = output_folder / "x_shift_diagnostic_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Centerline before/after CSV
    p_sim_shifted_line = sim_interp(x_line - best_shift)
    line_df = pd.DataFrame({
        "x_over_Lp": x_line,
        "p_sim_original_over_pinf": p_sim_line,
        "p_sim_shifted_over_pinf": p_sim_shifted_line,
        "p_digitized_over_pinf": p_dig_line,
        "digitization_uncertainty": u_dig_line,
        "difference_original_sim_minus_digitized": p_sim_line - p_dig_line,
        "difference_shifted_sim_minus_digitized": p_sim_shifted_line - p_dig_line,
    })
    line_csv_path = output_folder / "x_shift_centerline_before_after.csv"
    line_df.to_csv(line_csv_path, index=False)

    # Plot RMSE curve
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(shifts, rmse_vals, linewidth=2.0)
    ax.axvline(best_shift, color="k", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"Simulation shift, $\Delta x/L_p$")
    if use_uncertainty_weighting:
        ax.set_ylabel("Weighted RMSE")
    else:
        ax.set_ylabel(r"RMSE in $P/P_\infty$")
    ax.set_title("Least-error x-shift diagnostic")
    ax.grid(True, alpha=0.3)
    ax.annotate(
        f"best shift = {best_shift:+.5f} Lp\n= {best_shift_physical:+.4e} m",
        xy=(best_shift, rmse_after),
        xytext=(0.05, 0.95),
        textcoords="axes fraction",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        arrowprops=dict(arrowstyle="->"),
    )
    rmse_fig_path = output_folder / "x_shift_rmse_curve.png"
    plt.tight_layout()
    plt.savefig(rmse_fig_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Plot centerline before and after
    fig, axes = plt.subplots(
        2, 1,
        figsize=(9, 7),
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
    ax1.plot(x_line, p_dig_line, "--", linewidth=2.2, label="Digitized paper")
    ax1.plot(x_line, p_sim_line, linewidth=1.8, alpha=0.7, label="Simulation, original")
    ax1.plot(x_line, p_sim_shifted_line, linewidth=2.5, label="Simulation, shifted")
    ax1.set_ylabel(r"$P/P_\infty$")
    ax1.set_title(r"Centerline comparison before/after x-shift")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(x_line, p_sim_line - p_dig_line, linewidth=1.8, alpha=0.7, label="Original sim - digitized")
    ax2.plot(x_line, p_sim_shifted_line - p_dig_line, linewidth=2.3, label="Shifted sim - digitized")
    ax2.fill_between(x_line, -u_dig_line, u_dig_line, alpha=0.25, linewidth=0, label=r"$\pm$ digitization uncertainty")
    ax2.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    ax2.set_xlabel(r"Final normalized coordinate, $x/L_p$")
    ax2.set_ylabel(r"$\Delta P/P_\infty$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig_path = output_folder / "x_shift_centerline_before_after.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {diagnostic_curve_path}")
    print(f"  {line_csv_path}")
    print(f"  {rmse_fig_path}")
    print(f"  {fig_path}")


if __name__ == "__main__":
    main()
