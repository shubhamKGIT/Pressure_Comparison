#!/usr/bin/env python3
"""
diagnose_x_shift_smart.py

Estimate the streamwise x-shift needed to align a Fluent/simulation pressure field
with a digitized paper pressure field, using smarter shift diagnostics than plain
least-squares over the full centerline.

This is a drop-in replacement for diagnose_x_shift_least_error.py.

Why this version exists
-----------------------
A plain least-squares fit can overcompensate when the pressure rise / shock
impingement region has a very large gradient. Small x-errors in that region
produce large pressure residuals, so ordinary RMSE can choose a shift that
aligns the shock jump but degrades the plateau/recovery region.

This script supports several methods:

1. least_squares
   Original behavior: ordinary RMSE over selected fitting points.

2. robust_weighted
   Uses soft-L1 robust loss and downweights high-gradient digitized-pressure
   regions. Recommended default.

3. feature_midrise
   Finds the pressure-rise midpoint in simulation and digitized centerline
   curves, then computes shift = x_midrise_digitized - x_midrise_simulation.

4. hybrid_feature_then_robust
   Uses feature_midrise as an initial estimate, then performs robust weighted
   optimization only in a local window around that feature-based estimate.
   Recommended when ordinary least squares visibly over-shifts.

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
3. x_shift_centerline_before_after.csv
4. x_shift_centerline_before_after.png
5. x_shift_error_curve.png
6. x_shift_method_comparison.csv

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

# ==================================================
# Shift fitting controls
# ==================================================
# Recommended options:
#   "robust_weighted"             robust loss + high-gradient downweighting
#   "hybrid_feature_then_robust"  feature midpoint initial guess + local robust search
#   "feature_midrise"             feature-based shift only
#   "least_squares"               original ordinary RMSE behavior
SHIFT_METHOD = "hybrid_feature_then_robust"

# Global shift search range in normalized x/Lp units.
shift_min_over_Lp = -0.20
shift_max_over_Lp = 0.20

# For hybrid_feature_then_robust, search only near the feature-based shift.
# This helps prevent the optimizer from jumping to an overcompensated solution.
hybrid_local_search_half_width = 0.035

# Fitting windows used for least_squares, robust_weighted, and hybrid methods.
# These exclude the immediate high-gradient pressure-rise zone by default.
# Modify these based on your plot.
fit_windows = [
    (0.30, 0.355),   # pre-shock / approach region
    (0.42, 0.90),    # post-shock plateau and recovery region
]
points_per_window = 500

# Fallback continuous window if you set USE_FIT_WINDOWS = False.
USE_FIT_WINDOWS = True
fit_x_min = 0.05
fit_x_max = 0.95

# Optional uncertainty weighting.
# This weights residuals by digitization uncertainty, in addition to robust/high-gradient logic.
# Keep False initially unless your digitization uncertainty is reliable and nonzero.
use_uncertainty_weighting = False

# Optional: remove a constant pressure bias before fitting shift.
# Usually keep False if the pressure offset is physical.
allow_constant_pressure_bias = False

# Robust weighted fitting controls
# Larger alpha downweights the high-gradient shock-rise region more strongly.
gradient_downweight_alpha = 8.0

# soft_l1_scale is a pressure-ratio residual scale.
# Residuals much larger than this behave more like L1 than L2.
soft_l1_scale = 0.025

# For feature_midrise.
# The algorithm estimates upstream baseline and peak using these windows, then finds
# the location where pressure reaches baseline + midpoint_fraction*(peak-baseline).
feature_upstream_window = (0.05, 0.32)
feature_peak_window = (0.40, 0.60)
midpoint_fraction = 0.50

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


def make_x_eval_from_windows(fit_windows, points_per_window=500):
    parts = []
    for a, b in fit_windows:
        if b <= a:
            raise ValueError(f"Invalid fit window: {(a, b)}")
        parts.append(np.linspace(a, b, points_per_window))
    return np.unique(np.concatenate(parts))


def soft_l1_loss(residual, scale):
    r = residual / max(scale, 1e-12)
    return 2.0 * (np.sqrt(1.0 + r**2) - 1.0)


def gradient_weights(x_valid, p_valid, alpha=8.0):
    if len(x_valid) < 5:
        return np.ones_like(p_valid)

    dpdx = np.gradient(p_valid, x_valid)
    max_grad = np.nanmax(np.abs(dpdx))

    if not np.isfinite(max_grad) or max_grad <= 0:
        return np.ones_like(p_valid)

    grad_norm = np.abs(dpdx) / max_grad
    return 1.0 / (1.0 + alpha * grad_norm)


def compute_feature_midrise_shift(x_line, p_sim_line, p_dig_line):
    """
    Estimate shift from the midpoint of pressure rise.

    Returns:
        shift = x_midrise_digitized - x_midrise_simulation
    because x_sim_shifted = x_sim + shift.
    """
    def midrise_location(x, p, label):
        x = np.asarray(x, dtype=float)
        p = np.asarray(p, dtype=float)
        mask = np.isfinite(x) & np.isfinite(p)
        x = x[mask]
        p = p[mask]

        if len(x) < 10:
            raise ValueError(f"Too few points for {label} midrise detection.")

        up_mask = (x >= feature_upstream_window[0]) & (x <= feature_upstream_window[1])
        pk_mask = (x >= feature_peak_window[0]) & (x <= feature_peak_window[1])

        if np.count_nonzero(up_mask) < 5:
            raise ValueError(f"Upstream feature window too sparse for {label}.")
        if np.count_nonzero(pk_mask) < 5:
            raise ValueError(f"Peak feature window too sparse for {label}.")

        p_base = float(np.nanmedian(p[up_mask]))
        p_peak = float(np.nanmax(p[pk_mask]))
        p_target = p_base + midpoint_fraction * (p_peak - p_base)

        # Search for first upward crossing in a broad region near expected shock rise.
        search_mask = (x >= feature_upstream_window[1]) & (x <= feature_peak_window[1])
        xs = x[search_mask]
        ps = p[search_mask]

        if len(xs) < 5:
            raise ValueError(f"Feature search window too sparse for {label}.")

        # Find first crossing from below to above target.
        q = ps - p_target
        crossing_indices = np.where((q[:-1] <= 0.0) & (q[1:] >= 0.0))[0]

        if len(crossing_indices) == 0:
            # Fallback: closest point to the target.
            idx = int(np.nanargmin(np.abs(q)))
            return float(xs[idx]), p_base, p_peak, p_target

        i = int(crossing_indices[0])
        x0, x1 = xs[i], xs[i + 1]
        q0, q1 = q[i], q[i + 1]

        if abs(q1 - q0) < 1e-12:
            x_cross = 0.5 * (x0 + x1)
        else:
            x_cross = x0 - q0 * (x1 - x0) / (q1 - q0)

        return float(x_cross), p_base, p_peak, p_target

    x_mid_sim, sim_base, sim_peak, sim_target = midrise_location(x_line, p_sim_line, "simulation")
    x_mid_dig, dig_base, dig_peak, dig_target = midrise_location(x_line, p_dig_line, "digitized")

    shift = x_mid_dig - x_mid_sim
    details = {
        "x_midrise_sim": x_mid_sim,
        "x_midrise_digitized": x_mid_dig,
        "feature_midrise_shift_over_Lp": shift,
        "sim_baseline": sim_base,
        "sim_peak": sim_peak,
        "sim_midrise_pressure": sim_target,
        "digitized_baseline": dig_base,
        "digitized_peak": dig_peak,
        "digitized_midrise_pressure": dig_target,
    }
    return float(shift), details


def objective_for_shift(
    shift,
    x_eval,
    sim_interp,
    dig_interp,
    unc_interp,
    method="least_squares",
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

    x_valid = x_eval[mask]
    p_dig_valid = p_dig[mask]
    u_valid = u_dig[mask]
    residual = p_sim_shifted[mask] - p_dig_valid

    if allow_constant_pressure_bias:
        if use_uncertainty_weighting:
            w = 1.0 / np.maximum(u_valid, 1e-12) ** 2
            bias = np.sum(w * residual) / np.sum(w)
        else:
            bias = np.nanmean(residual)
        residual = residual - bias

    if method == "least_squares":
        if use_uncertainty_weighting:
            residual = residual / np.maximum(u_valid, 1e-12)
        return float(np.sqrt(np.nanmean(residual**2)))

    if method in {"robust_weighted", "hybrid_feature_then_robust"}:
        w_grad = gradient_weights(x_valid, p_dig_valid, alpha=gradient_downweight_alpha)

        if use_uncertainty_weighting:
            w_unc = 1.0 / np.maximum(u_valid, 1e-12) ** 2
            w_unc = w_unc / np.nanmean(w_unc)
        else:
            w_unc = np.ones_like(w_grad)

        weights = w_grad * w_unc
        loss = soft_l1_loss(residual, scale=soft_l1_scale)
        return float(np.sqrt(np.nanmean(weights * loss)) * soft_l1_scale)

    raise ValueError(f"Unknown objective method: {method}")


def optimize_shift(method, bounds, x_eval, sim_interp, dig_interp, unc_interp):
    result = minimize_scalar(
        objective_for_shift,
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-6},
        args=(
            x_eval,
            sim_interp,
            dig_interp,
            unc_interp,
            method,
            use_uncertainty_weighting,
            allow_constant_pressure_bias,
        ),
    )
    return float(result.x), float(result.fun)


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

    if USE_FIT_WINDOWS:
        x_eval = make_x_eval_from_windows(fit_windows, points_per_window=points_per_window)
    else:
        x_eval = np.linspace(fit_x_min, fit_x_max, 1000)

    # Always compute reference methods for comparison.
    rmse_before = objective_for_shift(
        0.0,
        x_eval,
        sim_interp,
        dig_interp,
        unc_interp,
        method="least_squares",
        use_uncertainty_weighting=use_uncertainty_weighting,
        allow_constant_pressure_bias=allow_constant_pressure_bias,
    )

    robust_before = objective_for_shift(
        0.0,
        x_eval,
        sim_interp,
        dig_interp,
        unc_interp,
        method="robust_weighted",
        use_uncertainty_weighting=use_uncertainty_weighting,
        allow_constant_pressure_bias=allow_constant_pressure_bias,
    )

    least_squares_shift, least_squares_error = optimize_shift(
        "least_squares",
        (shift_min_over_Lp, shift_max_over_Lp),
        x_eval,
        sim_interp,
        dig_interp,
        unc_interp,
    )

    robust_shift, robust_error = optimize_shift(
        "robust_weighted",
        (shift_min_over_Lp, shift_max_over_Lp),
        x_eval,
        sim_interp,
        dig_interp,
        unc_interp,
    )

    feature_shift, feature_details = compute_feature_midrise_shift(x_line, p_sim_line, p_dig_line)
    feature_shift_clipped = float(np.clip(feature_shift, shift_min_over_Lp, shift_max_over_Lp))
    feature_error = objective_for_shift(
        feature_shift_clipped,
        x_eval,
        sim_interp,
        dig_interp,
        unc_interp,
        method="robust_weighted",
        use_uncertainty_weighting=use_uncertainty_weighting,
        allow_constant_pressure_bias=allow_constant_pressure_bias,
    )

    hybrid_lower = max(shift_min_over_Lp, feature_shift_clipped - hybrid_local_search_half_width)
    hybrid_upper = min(shift_max_over_Lp, feature_shift_clipped + hybrid_local_search_half_width)
    hybrid_shift, hybrid_error = optimize_shift(
        "hybrid_feature_then_robust",
        (hybrid_lower, hybrid_upper),
        x_eval,
        sim_interp,
        dig_interp,
        unc_interp,
    )

    method_rows = [
        {
            "method": "least_squares",
            "shift_over_Lp": least_squares_shift,
            "shift_physical_m": least_squares_shift * sim_info["Lp"],
            "fit_error": least_squares_error,
            "notes": "ordinary RMSE over selected fit points",
        },
        {
            "method": "robust_weighted",
            "shift_over_Lp": robust_shift,
            "shift_physical_m": robust_shift * sim_info["Lp"],
            "fit_error": robust_error,
            "notes": "soft-L1 loss with high-gradient downweighting",
        },
        {
            "method": "feature_midrise",
            "shift_over_Lp": feature_shift_clipped,
            "shift_physical_m": feature_shift_clipped * sim_info["Lp"],
            "fit_error": feature_error,
            "notes": "aligns pressure-rise midpoint locations",
        },
        {
            "method": "hybrid_feature_then_robust",
            "shift_over_Lp": hybrid_shift,
            "shift_physical_m": hybrid_shift * sim_info["Lp"],
            "fit_error": hybrid_error,
            "notes": "feature midpoint initial estimate with local robust search",
        },
    ]
    method_comparison = pd.DataFrame(method_rows)

    if SHIFT_METHOD == "least_squares":
        best_shift = least_squares_shift
        best_error = least_squares_error
        objective_method_for_curve = "least_squares"
    elif SHIFT_METHOD == "robust_weighted":
        best_shift = robust_shift
        best_error = robust_error
        objective_method_for_curve = "robust_weighted"
    elif SHIFT_METHOD == "feature_midrise":
        best_shift = feature_shift_clipped
        best_error = feature_error
        objective_method_for_curve = "robust_weighted"
    elif SHIFT_METHOD == "hybrid_feature_then_robust":
        best_shift = hybrid_shift
        best_error = hybrid_error
        objective_method_for_curve = "robust_weighted"
    else:
        raise ValueError(f"Unknown SHIFT_METHOD: {SHIFT_METHOD}")

    best_shift_physical = best_shift * sim_info["Lp"]

    print("\nShift method comparison:")
    print(method_comparison.to_string(index=False))

    print("\nSelected x-shift fit:")
    print(f"  SHIFT_METHOD: {SHIFT_METHOD}")
    print(f"  best_shift_over_Lp: {best_shift:+.8f}")
    print(f"  best_shift_physical: {best_shift_physical:+.8e} m")
    print(f"  least-squares RMSE before shift: {rmse_before:.8f}")
    print(f"  robust error before shift:       {robust_before:.8f}")
    print(f"  selected fit error after shift: {best_error:.8f}")
    print("\nSign convention:")
    print("  x_sim_shifted_over_Lp = x_sim_over_Lp + best_shift_over_Lp")
    print("  positive shift moves simulation downstream/right on normalized plots.")

    # Diagnostic curve for selected objective type.
    shifts = np.linspace(shift_min_over_Lp, shift_max_over_Lp, 401)
    error_vals = np.array([
        objective_for_shift(
            s,
            x_eval,
            sim_interp,
            dig_interp,
            unc_interp,
            method=objective_method_for_curve,
            use_uncertainty_weighting=use_uncertainty_weighting,
            allow_constant_pressure_bias=allow_constant_pressure_bias,
        )
        for s in shifts
    ])

    diagnostic_curve = pd.DataFrame({
        "shift_over_Lp": shifts,
        "shift_physical_m": shifts * sim_info["Lp"],
        "fit_error": error_vals,
        "objective_method": objective_method_for_curve,
    })
    diagnostic_curve_path = output_folder / "x_shift_diagnostic_curve.csv"
    diagnostic_curve.to_csv(diagnostic_curve_path, index=False)

    method_comparison_path = output_folder / "x_shift_method_comparison.csv"
    method_comparison.to_csv(method_comparison_path, index=False)

    # Summary includes both selected result and method details.
    summary_dict = {
        "selected_shift_method": SHIFT_METHOD,
        "best_shift_over_Lp": best_shift,
        "best_shift_physical_m": best_shift_physical,
        "selected_fit_error_after_shift": best_error,
        "least_squares_rmse_before_shift": rmse_before,
        "robust_error_before_shift": robust_before,
        "least_squares_shift_over_Lp": least_squares_shift,
        "robust_weighted_shift_over_Lp": robust_shift,
        "feature_midrise_shift_over_Lp": feature_shift_clipped,
        "hybrid_feature_then_robust_shift_over_Lp": hybrid_shift,
        "shift_min_over_Lp": shift_min_over_Lp,
        "shift_max_over_Lp": shift_max_over_Lp,
        "use_fit_windows": USE_FIT_WINDOWS,
        "fit_windows": str(fit_windows),
        "fit_x_min": fit_x_min,
        "fit_x_max": fit_x_max,
        "centerline_y_over_Lp": centerline_y,
        "centerline_band_half_width_over_Lp": centerline_band_half_width,
        "use_uncertainty_weighting": use_uncertainty_weighting,
        "allow_constant_pressure_bias": allow_constant_pressure_bias,
        "gradient_downweight_alpha": gradient_downweight_alpha,
        "soft_l1_scale": soft_l1_scale,
        "feature_upstream_window": str(feature_upstream_window),
        "feature_peak_window": str(feature_peak_window),
        "midpoint_fraction": midpoint_fraction,
        "x_min_original_m": sim_info["x_min"],
        "x_max_original_m": sim_info["x_max"],
        "Lp_m": sim_info["Lp"],
        "y_center_m": sim_info["y_center"],
        "p_inf_Pa": sim_info["p_inf"],
    }
    summary_dict.update(feature_details)

    summary = pd.DataFrame([summary_dict])
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

    # Plot diagnostic error curve.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(shifts, error_vals, linewidth=2.0, label=f"{objective_method_for_curve} error")
    ax.axvline(best_shift, color="k", linestyle="--", linewidth=1.3, label="selected shift")
    ax.axvline(least_squares_shift, color="0.5", linestyle=":", linewidth=1.3, label="least-squares shift")
    ax.axvline(robust_shift, color="0.3", linestyle="-.", linewidth=1.3, label="robust shift")
    ax.axvline(feature_shift_clipped, color="0.6", linestyle="--", linewidth=1.0, label="feature-midrise shift")

    if SHIFT_METHOD == "hybrid_feature_then_robust":
        ax.axvspan(hybrid_lower, hybrid_upper, alpha=0.12, label="hybrid local search range")

    ax.set_xlabel(r"Simulation shift, $\Delta x/L_p$")
    ax.set_ylabel("Fit error")
    ax.set_title("Smart x-shift diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.annotate(
        f"selected = {best_shift:+.5f} Lp\n= {best_shift_physical:+.4e} m",
        xy=(best_shift, best_error),
        xytext=(0.05, 0.95),
        textcoords="axes fraction",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        arrowprops=dict(arrowstyle="->"),
    )
    error_fig_path = output_folder / "x_shift_error_curve.png"
    plt.tight_layout()
    plt.savefig(error_fig_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Plot centerline before and after.
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

    # Mark feature locations for transparency.
    ax1.axvline(feature_details["x_midrise_digitized"], color="0.4", linestyle=":", linewidth=1.0, label="digitized midrise")
    ax1.axvline(feature_details["x_midrise_sim"] + best_shift, color="0.2", linestyle="-.", linewidth=1.0, label="shifted sim midrise")

    ax1.set_ylabel(r"$P/P_\infty$", fontsize=18)
    ax1.set_title(r"Centerline comparison before/after smart x-shift", fontsize=18)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=12)

    ax2.plot(x_line, p_sim_line - p_dig_line, linewidth=1.8, alpha=0.7, label="Original sim - digitized")
    ax2.plot(x_line, p_sim_shifted_line - p_dig_line, linewidth=2.3, label="Shifted sim - digitized")
    ax2.fill_between(x_line, -u_dig_line, u_dig_line, alpha=0.25, linewidth=0, label=r"$\pm$ digitization uncertainty")
    ax2.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)

    # Shade the fitting windows so you can verify what was used.
    if USE_FIT_WINDOWS:
        for a, b in fit_windows:
            ax2.axvspan(a, b, alpha=0.08)

    ax2.set_xlabel(r"Final normalized coordinate, $x/L_p$", fontsize=18)
    ax2.set_ylabel(r"$\Delta P/P_\infty$", fontsize=18)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=12)

    fig_path = output_folder / "x_shift_centerline_before_after.eps"
    plt.tight_layout()
    plt.savefig(fig_path, format='eps', dpi=300, bbox_inches="tight")
    plt.show()

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {diagnostic_curve_path}")
    print(f"  {method_comparison_path}")
    print(f"  {line_csv_path}")
    print(f"  {error_fig_path}")
    print(f"  {fig_path}")


if __name__ == "__main__":
    main()
