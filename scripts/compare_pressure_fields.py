import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path
import os
# ==================================================
# User inputs
# ==================================================
FILE_FOLDER = "case3"
REPO_DIR = Path(__file__).parent.parent
PROJECT = "SBLI_challenge"
SIM_CSV = "panel_pressure_sim.csv"
DIG_CSV = "digitized_pressure_field_2023Avia.csv"

project_dir = REPO_DIR/ "csv_files"/ PROJECT
# Simulation / Fluent CSV
sim_csv_filename = SIM_CSV
sim_csv = project_dir/ FILE_FOLDER/ sim_csv_filename

# Digitized paper CSv
dig_csv = project_dir/ FILE_FOLDER/ DIG_CSV

# Setting up where all outputs go
output_folder = REPO_DIR/ "results" / PROJECT/ FILE_FOLDER
if not os.path.exists(output_folder): 
    os.makedirs(output_folder)
# Output files
output_comparison_csv = output_folder/"pressure_field_comparison.csv"
output_2d_figure = output_folder/"pressure_comparison_2d.png"
output_centerline_figure = output_folder/"pressure_centerline_comparison.png"
output_centerline_csv = output_folder/"pressure_centerline_comparison.csv"

# Simulation pressure column
sim_pressure_column = "pressure"

# Reference pressure used to normalize simulation pressure
# Use the same P_inf as the paper.
p_inf = 48400.0

# Optional manual panel length.
# If None, Lp = x_max - x_min from simulation data.
Lp_manual = None

# Optional manual y-center.
# If None, y_center = midpoint of simulation y range.
y_center_manual = None

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
# This is in y/Lp units.
# For exactly y/Lp = 0, use nearest instead of band averaging below.

use_band_average_for_centerline = True

# Plot settings
cmap_pressure = "jet"
cmap_difference = "bwr"

# For pressure plots
pressure_plot_min = 0.7
pressure_plot_max = 1.7

# For difference plot.
# If None, script sets symmetric limits from data.
difference_limit_manual = None

# ==================================================
# Helper functions
# ==================================================

def require_columns(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def normalize_simulation_coordinates(df, p_inf, Lp_manual=None, y_center_manual=None):
    """
    Converts simulation x,y,pressure to:
        x_over_Lp, y_over_Lp, p_over_pinf
    """

    x_min = df["x"].min()
    x_max = df["x"].max()

    if Lp_manual is None:
        Lp = x_max - x_min
    else:
        Lp = Lp_manual

    if Lp <= 0:
        raise ValueError("Invalid Lp. Check x coordinates or set Lp_manual.")

    if y_center_manual is None:
        y_center = 0.5 * (df["y"].min() + df["y"].max())
    else:
        y_center = y_center_manual

    df = df.copy()

    df["x_over_Lp"] = (df["x"] - x_min) / Lp
    df["y_over_Lp"] = (df["y"] - y_center) / Lp
    df["p_over_pinf"] = df[sim_pressure_column] / p_inf

    info = {
        "x_min": x_min,
        "x_max": x_max,
        "Lp": Lp,
        "y_center": y_center,
        "p_inf": p_inf,
    }

    return df, info


def interpolate_to_grid(df, x_col, y_col, value_col, X, Y, method="linear"):
    """
    Interpolate scattered data to a target grid.
    Fills holes with nearest-neighbor interpolation.
    """

    points = (df[x_col].values, df[y_col].values)
    values = df[value_col].values

    Z = griddata(points, values, (X, Y), method=method)

    Z_nearest = griddata(points, values, (X, Y), method="nearest")

    Z = np.where(np.isnan(Z), Z_nearest, Z)

    return Z


def centerline_from_grid(X, Y, Z, y0=0.0, band_half_width=0.0025, use_band=True):
    """
    Extract line data along x at y/Lp = y0.
    If use_band=True, averages values within a band around y0.
    If use_band=False, uses nearest grid row.
    """

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
# Read simulation data
# ==================================================

sim_df = pd.read_csv(sim_csv)
require_columns(sim_df, ["x", "y", "z", sim_pressure_column], "Simulation CSV")

# Remove duplicate x-y locations by averaging pressure
sim_df = (
    sim_df[["x", "y", "z", sim_pressure_column]]
    .groupby(["x", "y"], as_index=False)[sim_pressure_column]
    .mean()
)

sim_df, sim_info = normalize_simulation_coordinates(
    sim_df,
    p_inf=p_inf,
    Lp_manual=Lp_manual,
    y_center_manual=y_center_manual
)

print("\nSimulation normalization:")
for k, v in sim_info.items():
    print(f"  {k}: {v:.8e}")

print(f"Simulation P/P_inf min: {sim_df['p_over_pinf'].min():.6f}")
print(f"Simulation P/P_inf max: {sim_df['p_over_pinf'].max():.6f}")

# ==================================================
# Read digitized data
# ==================================================

dig_df = pd.read_csv(dig_csv)

require_columns(
    dig_df,
    ["x_over_Lp", "y_over_Lp", "pressure_ratio", "pressure_ratio_uncertainty"],
    "Digitized CSV"
)

dig_df = dig_df[
    ["x_over_Lp", "y_over_Lp", "pressure_ratio", "pressure_ratio_uncertainty"]
].copy()

print("\nDigitized data:")
print(f"Digitized P/P_inf min: {dig_df['pressure_ratio'].min():.6f}")
print(f"Digitized P/P_inf max: {dig_df['pressure_ratio'].max():.6f}")
print(f"Digitized uncertainty median: {dig_df['pressure_ratio_uncertainty'].median():.6f}")
print(f"Digitized uncertainty 95th percentile: {dig_df['pressure_ratio_uncertainty'].quantile(0.95):.6f}")

# ==================================================
# Create common comparison grid
# ==================================================

x_grid = np.linspace(x_norm_min, x_norm_max, nx)
y_grid = np.linspace(y_norm_min, y_norm_max, ny)

X, Y = np.meshgrid(x_grid, y_grid)

# ==================================================
# Interpolate both fields onto common grid
# ==================================================

P_sim = interpolate_to_grid(
    sim_df,
    x_col="x_over_Lp",
    y_col="y_over_Lp",
    value_col="p_over_pinf",
    X=X,
    Y=Y
)

P_dig = interpolate_to_grid(
    dig_df,
    x_col="x_over_Lp",
    y_col="y_over_Lp",
    value_col="pressure_ratio",
    X=X,
    Y=Y
)

U_dig = interpolate_to_grid(
    dig_df,
    x_col="x_over_Lp",
    y_col="y_over_Lp",
    value_col="pressure_ratio_uncertainty",
    X=X,
    Y=Y
)

# ==================================================
# Difference and uncertainty-aware gap
# ==================================================

Delta = P_sim - P_dig

AbsDelta = np.abs(Delta)

# Part of the difference that exceeds digitization uncertainty
GapBeyondUncertainty = np.maximum(AbsDelta - U_dig, 0.0)

# Signed version:
# positive means simulation is higher than digitized beyond uncertainty
# negative means simulation is lower than digitized beyond uncertainty
SignedGapBeyondUncertainty = np.sign(Delta) * GapBeyondUncertainty

# z-score-like comparison
# |z| > 1 means difference exceeds 1-sigma digitization uncertainty
epsilon = 1e-12
DeltaOverUncertainty = Delta / np.maximum(U_dig, epsilon)

print("\nComparison statistics:")
print(f"Mean signed difference: {np.nanmean(Delta):.6f}")
print(f"Mean absolute difference: {np.nanmean(AbsDelta):.6f}")
print(f"Max absolute difference: {np.nanmax(AbsDelta):.6f}")
print(f"Mean gap beyond uncertainty: {np.nanmean(GapBeyondUncertainty):.6f}")
print(f"Max gap beyond uncertainty: {np.nanmax(GapBeyondUncertainty):.6f}")
print(f"Fraction of grid with |difference| > uncertainty: {np.mean(AbsDelta > U_dig):.3f}")

# ==================================================
# Save comparison CSV
# ==================================================

comparison_df = pd.DataFrame({
    "x_over_Lp": X.ravel(),
    "y_over_Lp": Y.ravel(),
    "p_sim_over_pinf": P_sim.ravel(),
    "p_digitized_over_pinf": P_dig.ravel(),
    "digitization_uncertainty": U_dig.ravel(),
    "difference_sim_minus_digitized": Delta.ravel(),
    "absolute_difference": AbsDelta.ravel(),
    "gap_beyond_digitization_uncertainty": GapBeyondUncertainty.ravel(),
    "signed_gap_beyond_digitization_uncertainty": SignedGapBeyondUncertainty.ravel(),
    "difference_over_uncertainty": DeltaOverUncertainty.ravel(),
})

comparison_df.to_csv(output_comparison_csv, index=False)
print(f"\nSaved comparison CSV to: {output_comparison_csv}")

# ==================================================
# 2D comparison plots
# ==================================================

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

# Simulation pressure
c1 = axes[0, 0].contourf(
    X, Y, P_sim,
    levels=np.linspace(pressure_plot_min, pressure_plot_max, 101),
    cmap=cmap_pressure,
    vmin=pressure_plot_min,
    vmax=pressure_plot_max,
    extend="both"
)
fig.colorbar(c1, ax=axes[0, 0])
axes[0, 0].set_title(r"Simulation: $P/P_\infty$")
axes[0, 0].set_xlabel(r"$x/L_p$", fontsize=18)
axes[0, 0].set_ylabel(r"$y/L_p$", fontsize=18)

# Digitized pressure
c2 = axes[0, 1].contourf(
    X, Y, P_dig,
    levels=np.linspace(pressure_plot_min, pressure_plot_max, 101),
    cmap=cmap_pressure,
    vmin=pressure_plot_min,
    vmax=pressure_plot_max,
    extend="both"
)
fig.colorbar(c2, ax=axes[0, 1])
axes[0, 1].set_title(r"Digitized paper: $P/P_\infty$")
axes[0, 1].set_xlabel(r"$x/L_p$", fontsize=18)
axes[0, 1].set_ylabel(r"$y/L_p$", fontsize=18)

# Difference
c3 = axes[1, 0].contourf(
    X, Y, Delta,
    levels=np.linspace(-diff_lim, diff_lim, 101),
    cmap=cmap_difference,
    vmin=-diff_lim,
    vmax=diff_lim,
    extend="both"
)
fig.colorbar(c3, ax=axes[1, 0])
axes[1, 0].set_title(r"Difference: simulation $-$ digitized")
axes[1, 0].set_xlabel(r"$x/L_p$", fontsize=18)
axes[1, 0].set_ylabel(r"$y/L_p$", fontsize=18)

# Gap beyond uncertainty
c4 = axes[1, 1].contourf(
    X, Y, SignedGapBeyondUncertainty,
    levels=np.linspace(-gap_lim, gap_lim, 101),
    cmap=cmap_difference,
    vmin=-gap_lim,
    vmax=gap_lim,
    extend="both"
)
fig.colorbar(c4, ax=axes[1, 1])
axes[1, 1].set_title(r"Signed gap beyond digitization uncertainty")
axes[1, 1].set_xlabel(r"$x/L_p$", fontsize=18)
axes[1, 1].set_ylabel(r"$y/L_p$", fontsize=18)

for ax in axes.ravel():
    ax.set_xlim(x_norm_min, x_norm_max)
    ax.set_ylim(y_norm_min, y_norm_max)
    ax.set_aspect("auto")

plt.savefig(output_2d_figure, dpi=300, bbox_inches="tight")
print(f"Saved 2D comparison figure to: {output_2d_figure}")
plt.show()

# ==================================================
# Centerline comparison
# ==================================================

x_line, p_sim_line = centerline_from_grid(
    X, Y, P_sim,
    y0=centerline_y,
    band_half_width=centerline_band_half_width,
    use_band=use_band_average_for_centerline
)

_, p_dig_line = centerline_from_grid(
    X, Y, P_dig,
    y0=centerline_y,
    band_half_width=centerline_band_half_width,
    use_band=use_band_average_for_centerline
)

_, u_dig_line = centerline_from_grid(
    X, Y, U_dig,
    y0=centerline_y,
    band_half_width=centerline_band_half_width,
    use_band=use_band_average_for_centerline
)

delta_line = p_sim_line - p_dig_line
gap_line = np.maximum(np.abs(delta_line) - u_dig_line, 0.0)

centerline_df = pd.DataFrame({
    "x_over_Lp": x_line,
    "p_sim_over_pinf": p_sim_line,
    "p_digitized_over_pinf": p_dig_line,
    "digitization_uncertainty": u_dig_line,
    "difference_sim_minus_digitized": delta_line,
    "absolute_difference": np.abs(delta_line),
    "gap_beyond_digitization_uncertainty": gap_line,
})

centerline_df.to_csv(output_centerline_csv, index=False)
print(f"Saved centerline comparison CSV to: {output_centerline_csv}")

# --------------------------------------------------
# Plot centerline comparison with clearer uncertainty band
# --------------------------------------------------

print("\nCenterline uncertainty statistics:")
print(f"  min uncertainty:    {np.nanmin(u_dig_line):.6f}")
print(f"  median uncertainty: {np.nanmedian(u_dig_line):.6f}")
print(f"  max uncertainty:    {np.nanmax(u_dig_line):.6f}")

fig, axes = plt.subplots(
    2, 1,
    figsize=(8.5, 6.5),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1.4]}
)

ax1 = axes[0]
ax2 = axes[1]

# -----------------------------
# Top plot: pressure comparison
# -----------------------------
ax1.fill_between(
    x_line,
    p_dig_line - u_dig_line,
    p_dig_line + u_dig_line,
    color="orange",
    alpha=0.35,
    linewidth=0,
    label="Digitized uncertainty band",
    zorder=1
)

ax1.plot(
    x_line,
    p_sim_line,
    color="tab:blue",
    linewidth=2.4,
    label="Simulation",
    zorder=3
)

ax1.plot(
    x_line,
    p_dig_line,
    color="tab:orange",
    linewidth=2.4,
    linestyle="--",
    label="Digitized paper",
    zorder=4
)

ax1.set_ylabel(r"$P/P_\infty$", fontsize=18)
ax1.set_title(r"Centerline Pressure Comparison at $y/L_p = 0$")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="best")

# Set y-limits only around pressure curves, not the difference curve
p_min_plot = np.nanmin([
    np.nanmin(p_sim_line),
    np.nanmin(p_dig_line - u_dig_line)
])
p_max_plot = np.nanmax([
    np.nanmax(p_sim_line),
    np.nanmax(p_dig_line + u_dig_line)
])

padding = 0.08 * (p_max_plot - p_min_plot)
ax1.set_ylim(p_min_plot - padding, p_max_plot + padding)

# -----------------------------
# Bottom plot: difference
# -----------------------------
delta_line = p_sim_line - p_dig_line

ax2.plot(
    x_line,
    delta_line,
    color="tab:green",
    linewidth=2.0,
    label=r"Difference: sim $-$ digitized"
)

# Digitization uncertainty envelope around zero
ax2.fill_between(
    x_line,
    -u_dig_line,
    u_dig_line,
    color="gray",
    alpha=0.35,
    linewidth=0,
    label=r"$\pm$ digitization uncertainty"
)

ax2.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)

ax2.set_xlabel(r"$x/L_p$", fontsize=18)
ax2.set_ylabel(r"$\Delta P/P_\infty$", fontsize=18)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="best")

# Optional: scale bottom axis around difference
diff_abs_max = np.nanmax(np.abs(delta_line))
unc_abs_max = np.nanmax(u_dig_line)
diff_ylim = 1.15 * max(diff_abs_max, unc_abs_max)

if diff_ylim > 0:
    ax2.set_ylim(-diff_ylim, diff_ylim)

plt.tight_layout()
plt.savefig(output_centerline_figure, dpi=300, bbox_inches="tight")
print(f"Saved centerline comparison figure to: {output_centerline_figure}")
plt.show()