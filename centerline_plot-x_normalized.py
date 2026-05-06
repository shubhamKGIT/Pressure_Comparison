import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# User inputs
# --------------------------------------------------
input_csv = Path("panel_pressure_clean_columns.csv")        # change this
output_figure = Path("pressure_centerline_x_norm.png")
output_csv = Path("pressure_centerline_x_norm.csv")

pressure_column = "pressure"

# If you know the exact centerline y value, set it here.
# Otherwise leave as None and the script uses (y_min + y_max) / 2.
target_y = None

# Tolerance for selecting points near centerline.
# If None, script selects the nearest available y location.
y_tolerance = None

# Pressure scaling
pressure_scale = 0.001    # use 0.001 if pressure is in Pa and you want kPa
pressure_label = "Pressure [kPa]"  # use "Pressure [kPa]" if pressure_scale = 0.001

# --------------------------------------------------
# Read data
# --------------------------------------------------
df = pd.read_csv(input_csv)

required_cols = ["x", "y", "z", pressure_column]
missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# --------------------------------------------------
# Find centerline y
# --------------------------------------------------
if target_y is None:
    target_y = 0.5 * (df["y"].min() + df["y"].max())

print(f"Requested centerline y = {target_y:.8e}")

# --------------------------------------------------
# Select centerline data
# --------------------------------------------------
if y_tolerance is None:
    # Choose nearest available y value
    unique_y = df["y"].drop_duplicates().sort_values()
    nearest_y = unique_y.iloc[(unique_y - target_y).abs().argmin()]

    centerline_df = df[df["y"] == nearest_y].copy()

    print(f"Nearest available y used = {nearest_y:.8e}")
    print(f"Difference from requested y = {abs(nearest_y - target_y):.8e}")

else:
    centerline_df = df[
        (df["y"] >= target_y - y_tolerance) &
        (df["y"] <= target_y + y_tolerance)
    ].copy()

    print(f"Using y tolerance = ±{y_tolerance:.8e}")

    # If multiple points exist for same x within tolerance, average pressure
    centerline_df = (
        centerline_df
        .groupby("x", as_index=False)[pressure_column]
        .mean()
    )

if centerline_df.empty:
    raise ValueError(
        "No points found along the requested centerline. "
        "Increase y_tolerance or check the y coordinates."
    )

# --------------------------------------------------
# Sort by x
# --------------------------------------------------
centerline_df = centerline_df.sort_values("x").reset_index(drop=True)

# --------------------------------------------------
# Normalize x by panel length L_p
# --------------------------------------------------
x_min = centerline_df["x"].min()
x_max = centerline_df["x"].max()

L_p = x_max - x_min

if L_p <= 0:
    raise ValueError("Invalid panel length L_p. Check x coordinates.")

centerline_df["x_over_Lp"] = (centerline_df["x"] - x_min) / L_p

print(f"x_min = {x_min:.8e}")
print(f"x_max = {x_max:.8e}")
print(f"L_p   = {L_p:.8e}")

# --------------------------------------------------
# Save extracted and normalized centerline data
# --------------------------------------------------
centerline_df.to_csv(output_csv, index=False)
print(f"Saved extracted centerline data to: {output_csv}")

# --------------------------------------------------
# Plot pressure vs normalized x
# --------------------------------------------------
x_plot = centerline_df["x_over_Lp"]
p_plot = centerline_df[pressure_column] * pressure_scale

plt.figure(figsize=(8, 5))
plt.plot(x_plot, p_plot, linewidth=2)

plt.xlabel(r"$x/L_p$")
plt.ylabel(pressure_label)
plt.title(r"Centerline Pressure Along Panel, $y = y_c$")
plt.grid(True, alpha=0.3)

plt.xlim(0.0, 1.0)

plt.tight_layout()
plt.savefig(output_figure, dpi=300, bbox_inches="tight")

print(f"Saved plot to: {output_figure}")

plt.show()