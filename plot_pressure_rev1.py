import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path

# --------------------------------------------------
# User inputs
# --------------------------------------------------
input_csv = Path("panel_pressure_clean_columns.csv")

output_figure = Path("pressure_contour_normalized.png")
output_csv = Path("pressure_contour_normalized_data.csv")

pressure_column = "pressure"

# Reference pressure for normalization
# Use your freestream/static reference pressure here.
# Example from your earlier setup: p_inf = 48400 Pa
p_inf = 48400.0

# Optional manual panel length.
# If None, Lp = x_max - x_min from data.
Lp_manual = None

# Grid resolution for smooth contour
nx = 400
ny = 200

# Colorbar range.
# Use fixed values if you want direct comparison to paper.
use_fixed_color_limits = True
p_over_pinf_min = 0.7
p_over_pinf_max = 1.7

# Plot y-limits in normalized coordinates.
# Set to None to use full data range.
y_norm_limits = (-0.25, 0.25)

# Colormap similar to Fluent-style rainbow
cmap_name = "jet"

# --------------------------------------------------
# Read data
# --------------------------------------------------
df = pd.read_csv(input_csv)

required_cols = ["x", "y", "z", pressure_column]
missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# Keep required columns only
df = df[["x", "y", "z", pressure_column]].copy()

# Remove duplicate x-y points by averaging pressure
df = (
    df.groupby(["x", "y"], as_index=False)[pressure_column]
    .mean()
)

# --------------------------------------------------
# Normalize coordinates
# --------------------------------------------------
x_min = df["x"].min()
x_max = df["x"].max()

if Lp_manual is None:
    Lp = x_max - x_min
else:
    Lp = Lp_manual

if Lp <= 0:
    raise ValueError("Invalid panel length Lp. Check x coordinates.")

# x/Lp from 0 to 1
df["x_over_Lp"] = (df["x"] - x_min) / Lp

# Choose y center and normalize y/Lp
y_center = 0.5 * (df["y"].min() + df["y"].max())
df["y_over_Lp"] = (df["y"] - y_center) / Lp

# Normalize pressure
df["p_over_pinf"] = df[pressure_column] / p_inf

print(f"x_min = {x_min:.8e}")
print(f"x_max = {x_max:.8e}")
print(f"Lp    = {Lp:.8e}")
print(f"y_center = {y_center:.8e}")
print(f"p_inf = {p_inf:.8e}")

print(f"P/P_inf min = {df['p_over_pinf'].min():.6f}")
print(f"P/P_inf max = {df['p_over_pinf'].max():.6f}")

# Save normalized data
df.to_csv(output_csv, index=False)
print(f"Saved normalized data to: {output_csv}")

# --------------------------------------------------
# Create interpolation grid
# --------------------------------------------------
xi = np.linspace(df["x_over_Lp"].min(), df["x_over_Lp"].max(), nx)
yi = np.linspace(df["y_over_Lp"].min(), df["y_over_Lp"].max(), ny)

X, Y = np.meshgrid(xi, yi)

P = griddata(
    points=(df["x_over_Lp"].values, df["y_over_Lp"].values),
    values=df["p_over_pinf"].values,
    xi=(X, Y),
    method="linear"
)

# Fill remaining holes using nearest interpolation
P_nearest = griddata(
    points=(df["x_over_Lp"].values, df["y_over_Lp"].values),
    values=df["p_over_pinf"].values,
    xi=(X, Y),
    method="nearest"
)

P = np.where(np.isnan(P), P_nearest, P)

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.0))

if use_fixed_color_limits:
    levels = np.linspace(p_over_pinf_min, p_over_pinf_max, 101)
    contour = ax.contourf(
        X, Y, P,
        levels=levels,
        cmap=cmap_name,
        vmin=p_over_pinf_min,
        vmax=p_over_pinf_max,
        extend="both"
    )
else:
    contour = ax.contourf(
        X, Y, P,
        levels=101,
        cmap=cmap_name
    )

# Optional: add very light contour lines
# ax.contour(X, Y, P, levels=20, colors="k", linewidths=0.2, alpha=0.25)

cbar = fig.colorbar(contour, ax=ax, pad=0.035)
cbar.set_label(r"$P_{\mathrm{RANS}}/P_\infty$", rotation=90)

if use_fixed_color_limits:
    cbar.set_ticks(np.linspace(p_over_pinf_min, p_over_pinf_max, 6))

ax.set_xlabel(r"$x/L_p$")
ax.set_ylabel(r"$y/L_p$")

ax.set_xlim(0.0, 1.0)

if y_norm_limits is not None:
    ax.set_ylim(y_norm_limits[0], y_norm_limits[1])

# This makes the panel look like the paper snapshot rather than geometrically equal
ax.set_aspect("auto")

plt.tight_layout()
plt.savefig(output_figure, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {output_figure}")

plt.show()