import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from pathlib import Path

# --------------------------------------------------
# User inputs
# --------------------------------------------------
input_csv = Path("panel_pressure_clean_columns.csv")   # change to your file name
n_levels = 100                                  # number of contour levels
cmap_name = "jet"                               # Fluent-like rainbow colormap
save_figure = True
output_figure = Path("pressure_contour_xy.png")

# --------------------------------------------------
# Read data
# --------------------------------------------------
df = pd.read_csv(input_csv)

# Keep only needed columns
df = df[["x", "y", "z", "pressure"]].copy()

# Optional: verify z is approximately constant
z_min = df["z"].min()
z_max = df["z"].max()
print(f"z range: {z_min:.6e} to {z_max:.6e}")

if abs(z_max - z_min) > 1e-10:
    print("Warning: z is not strictly constant. Plot will still use x-y projection.")

# Optional: remove duplicate x-y pairs by averaging pressure
df = df.groupby(["x", "y"], as_index=False)["pressure"].mean()

# Extract columns
x = df["x"].values
y = df["y"].values
p = df["pressure"].values

# --------------------------------------------------
# Create triangulation for contour plot
# --------------------------------------------------
triang = tri.Triangulation(x, y)

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

contour = ax.tricontourf(triang, p, levels=n_levels, cmap=cmap_name)
ax.tricontour(triang, p, levels=20, colors="k", linewidths=0.3, alpha=0.4)

cbar = fig.colorbar(contour, ax=ax)
cbar.set_label("Pressure")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Pressure Contour on x-y Surface")
ax.set_aspect("equal", adjustable="box")

plt.tight_layout()

# Save figure if desired
if save_figure:
    plt.savefig(output_figure, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {output_figure}")

plt.show()