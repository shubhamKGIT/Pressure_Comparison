import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# User inputs
# --------------------------------------------------
REPO_DIR = Path(__file__).parent.parent
PROJECT = "2023Avaition"
IN_CSV = "panel_pressure_clean_columns.csv"
OUT_CSV  = "pressure_centerline_x.csv"
OUT_FIG = "pressure_centerline_x.png"

csv_folder = REPO_DIR/ "csv_files"/ PROJECT
input_csv = csv_folder/ IN_CSV      # change this
output_csv = csv_folder / OUT_CSV

results_folder = REPO_DIR/ "results" / PROJECT
output_figure = results_folder/ OUT_FIG



pressure_column = "pressure"

# If you know the exact centerline y value, set it here.
# Otherwise leave as None and the script uses (y_min + y_max) / 2.
target_y = None

# Tolerance for selecting points near centerline.
# If None, script selects the nearest available y location.
y_tolerance = None

# Unit scaling for plots
x_scale = 1000.0        # use 1000.0 if x is in m and you want mm
pressure_scale = 0.001 # use 0.001 if pressure is in Pa and you want kPa

x_label = "x [mm]"
pressure_label = "Pressure [kPa]"

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
    # choose nearest available y value
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

    # If multiple points exist for same x within tolerance, average them
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
centerline_df = centerline_df.sort_values("x")

# Save extracted centerline data
centerline_df.to_csv(output_csv, index=False)
print(f"Saved extracted centerline data to: {output_csv}")

# --------------------------------------------------
# Plot pressure vs x
# --------------------------------------------------
x_plot = centerline_df["x"] * x_scale
p_plot = centerline_df[pressure_column] * pressure_scale

plt.figure(figsize=(8, 5))
plt.plot(x_plot, p_plot, linewidth=2)

plt.xlabel(x_label)
plt.ylabel(pressure_label)
plt.title("Centerline Pressure Along x-direction")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_figure, dpi=300, bbox_inches="tight")

print(f"Saved plot to: {output_figure}")

plt.show()