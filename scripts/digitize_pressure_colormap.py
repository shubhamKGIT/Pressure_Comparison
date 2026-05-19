import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter1d
import pathlib
import os

# ==================================================
# User inputs
# ==================================================
REPO_DIR = pathlib.Path(__file__).parent.parent
PROJECT = "2023_Aviation"
OUT_CSV = "digitized_pressure_field_2023Avia.csv"
IMAGE = "paper_pressure_4deg_2023Aviation.png"
OUT_PNG = "digitized_pressure_field_2023Avia.png"

paper_image_folder = "snapshots_from_paper"
snapshot_folder = REPO_DIR / paper_image_folder
snapshot_to_digitize = snapshot_folder/ IMAGE
results_folder = REPO_DIR/ "results" / PROJECT

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

image_path = snapshot_to_digitize.resolve()
output_folder = Path(REPO_DIR/ "csv_files"/ PROJECT)
output_csv = output_folder/ OUT_CSV

output_preview = results_folder / OUT_PNG

# Output grid size
nx = 400
ny = 200

# Axis limits from the paper figure
x_min_phys = 0.0
x_max_phys = 1.0

y_min_phys = -0.25
y_max_phys = 0.25

# Colorbar value limits from the paper figure
p_min = 0.7
p_max = 1.7

# Image bit depth assumption for normal PNG/JPG screenshots
bits_per_channel = 8

# Set True for interactive clicking calibration
interactive = True

# If interactive = False, manually enter pixel coordinates here.
# Pixel coordinates are in image coordinates: x to right, y downward.
#
# plot_corners:
#   lower-left, lower-right, upper-left, upper-right
#
# colorbar_points:
#   bottom of colorbar, top of colorbar
#
manual_plot_corners = {
    "lower_left":  (175, 269),
    "lower_right": (588, 269),
    "upper_left":  (175, 56),
    "upper_right": (588, 56),
}

manual_colorbar_points = {
    "bottom": (620, 268),
    "top":    (620, 57),
}

# Smooth sampled colorbar to reduce tick/text/noise effects
smooth_colorbar = True
smooth_window = 5

# ==================================================
# Helper functions
# ==================================================
def load_rgb_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float64) / 255.0
    return arr


def click_points(image, title, labels):
    """
    Interactively click points on image.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(image)
    ax.set_title(title)
    ax.set_axis_off()

    clicked = []

    for label in labels:
        print(f"Click: {label}")
        pts = plt.ginput(1, timeout=-1)
        if len(pts) == 0:
            raise RuntimeError("No point selected.")
        x, y = pts[0]
        clicked.append((x, y))
        ax.plot(x, y, "wo", markeredgecolor="k")
        ax.text(x + 5, y + 5, label, color="white",
                bbox=dict(facecolor="black", alpha=0.6))
        plt.draw()

    plt.close(fig)
    return clicked

def bilinear_map(u, v, corners):
    """
    Map normalized coordinates u, v in [0,1] to image pixel coordinates.

    corners:
        lower_left, lower_right, upper_left, upper_right

    u = 0 left, u = 1 right
    v = 0 bottom, v = 1 top

    Returns:
        xpix, ypix arrays with same shape as u and v
    """

    ll = np.array(corners["lower_left"], dtype=float)
    lr = np.array(corners["lower_right"], dtype=float)
    ul = np.array(corners["upper_left"], dtype=float)
    ur = np.array(corners["upper_right"], dtype=float)

    # Compute x-pixel map
    xpix = (
        (1 - u) * (1 - v) * ll[0]
        + u * (1 - v) * lr[0]
        + (1 - u) * v * ul[0]
        + u * v * ur[0]
    )

    # Compute y-pixel map
    ypix = (
        (1 - u) * (1 - v) * ll[1]
        + u * (1 - v) * lr[1]
        + (1 - u) * v * ul[1]
        + u * v * ur[1]
    )

    return xpix, ypix


def sample_image_nearest(image, xpix, ypix):
    """
    Sample RGB image using nearest-neighbor lookup.
    """
    h, w, _ = image.shape

    xi = np.rint(xpix).astype(int)
    yi = np.rint(ypix).astype(int)

    xi = np.clip(xi, 0, w - 1)
    yi = np.clip(yi, 0, h - 1)

    return image[yi, xi, :]


def sample_colorbar(image, bottom_xy, top_xy, n_samples=None):
    """
    Sample RGB values along the colorbar centerline.

    bottom_xy corresponds to p_min.
    top_xy corresponds to p_max.
    """
    bottom_xy = np.array(bottom_xy, dtype=float)
    top_xy = np.array(top_xy, dtype=float)

    length_pix = np.linalg.norm(top_xy - bottom_xy)

    if n_samples is None:
        n_samples = int(max(100, np.ceil(length_pix)))

    s = np.linspace(0.0, 1.0, n_samples)

    # s = 0 bottom, s = 1 top
    pts = bottom_xy[None, :] * (1 - s[:, None]) + top_xy[None, :] * s[:, None]

    colors = sample_image_nearest(image, pts[:, 0], pts[:, 1])

    if smooth_colorbar:
        colors = uniform_filter1d(colors, size=smooth_window, axis=0, mode="nearest")

    values = p_min + s * (p_max - p_min)

    return values, colors, length_pix


def estimate_uncertainty(
    nearest_color_distance,
    colorbar_values,
    colorbar_colors,
    bits=8
):
    """
    Estimate uncertainty in extracted scalar values.

    Components:
    1. Colorbar scalar resolution due to finite colorbar sampling.
    2. RGB quantization uncertainty due to finite image bit depth.
    3. Color matching uncertainty based on nearest-neighbor color residual.

    This is an approximate uncertainty, not a rigorous experimental error.
    """
    # Scalar step from finite colorbar discretization
    dv_colorbar = np.mean(np.abs(np.diff(colorbar_values)))

    # 8-bit RGB quantization step in normalized color space
    dq = 1.0 / (2**bits - 1)

    # Estimate local gradient: color distance per scalar value
    dcolor = np.linalg.norm(np.diff(colorbar_colors, axis=0), axis=1)
    dvalue = np.abs(np.diff(colorbar_values))

    # Avoid division by zero
    valid = dvalue > 0
    gradients = dcolor[valid] / dvalue[valid]

    # Use robust median gradient
    median_grad = np.median(gradients[gradients > 1e-12])

    if not np.isfinite(median_grad) or median_grad <= 0:
        median_grad = 1.0

    # Color quantization uncertainty converted to scalar uncertainty
    # sqrt(3) because RGB has three channels
    rgb_quant_color_unc = np.sqrt(3.0) * dq / 2.0
    dv_quant = rgb_quant_color_unc / median_grad

    # Color mismatch uncertainty converted to scalar uncertainty
    dv_match = nearest_color_distance / median_grad

    # Combine uncertainties
    dv_total = np.sqrt(dv_colorbar**2 + dv_quant**2 + dv_match**2)

    return dv_total, {
        "dv_colorbar": dv_colorbar,
        "dv_quant_typical": dv_quant,
        "median_color_gradient": median_grad,
    }


# ==================================================
# Main script
# ==================================================
image = load_rgb_image(image_path)

if interactive:
    labels = ["lower_left", "lower_right", "upper_left", "upper_right"]
    pts = click_points(
        image,
        "Click plot corners: lower-left, lower-right, upper-left, upper-right",
        labels
    )

    plot_corners = dict(zip(labels, pts))

    cb_labels = ["bottom", "top"]
    cb_pts = click_points(
        image,
        "Click colorbar points: bottom value, top value",
        cb_labels
    )

    colorbar_points = dict(zip(cb_labels, cb_pts))

else:
    plot_corners = manual_plot_corners
    colorbar_points = manual_colorbar_points

print("Selected plot corners:")
for k, val in plot_corners.items():
    print(f"  {k}: {val}")

print("Selected colorbar points:")
for k, val in colorbar_points.items():
    print(f"  {k}: {val}")
    
# --------------------------------------------------
# Sample colorbar and build color-to-value mapping
# --------------------------------------------------
colorbar_values, colorbar_colors, colorbar_length_pix = sample_colorbar(
    image,
    colorbar_points["bottom"],
    colorbar_points["top"]
)

tree = cKDTree(colorbar_colors)

print(f"Sampled colorbar length: {colorbar_length_pix:.1f} pixels")
print(f"Number of colorbar samples: {len(colorbar_values)}")
print(f"Colorbar value range: {p_min} to {p_max}")


# --------------------------------------------------
# Create output physical grid
# --------------------------------------------------
x_vals = np.linspace(x_min_phys, x_max_phys, nx)
y_vals = np.linspace(y_min_phys, y_max_phys, ny)

X_phys, Y_phys = np.meshgrid(x_vals, y_vals)

# Convert physical coordinates to normalized plot coordinates
u = (X_phys - x_min_phys) / (x_max_phys - x_min_phys)
v = (Y_phys - y_min_phys) / (y_max_phys - y_min_phys)

# Convert normalized coordinates to image pixel coordinates
X_pix, Y_pix = bilinear_map(u, v, plot_corners)

# Sample image colors from plot area
plot_colors = sample_image_nearest(image, X_pix, Y_pix)

# Flatten for color matching
plot_colors_flat = plot_colors.reshape(-1, 3)

# Find nearest colorbar color for each plot pixel
distances, indices = tree.query(plot_colors_flat, k=1)

P_flat = colorbar_values[indices]
P = P_flat.reshape(ny, nx)

color_dist = distances.reshape(ny, nx)


# --------------------------------------------------
# Estimate uncertainty
# --------------------------------------------------
uncert_flat, unc_info = estimate_uncertainty(
    nearest_color_distance=distances,
    colorbar_values=colorbar_values,
    colorbar_colors=colorbar_colors,
    bits=bits_per_channel
)

P_unc = uncert_flat.reshape(ny, nx)

print("\nApproximate uncertainty information:")
print(f"  colorbar scalar resolution uncertainty: ±{unc_info['dv_colorbar']:.5f}")
print(f"  RGB quantization typical uncertainty:   ±{unc_info['dv_quant_typical']:.5f}")
print(f"  median color gradient:                  {unc_info['median_color_gradient']:.5f}")
print(f"  median total uncertainty:               ±{np.median(P_unc):.5f}")
print(f"  95th percentile uncertainty:            ±{np.percentile(P_unc, 95):.5f}")
print(f"  max uncertainty:                        ±{np.max(P_unc):.5f}")


# --------------------------------------------------
# Save CSV
# --------------------------------------------------
out_df = pd.DataFrame({
    "x_over_Lp": X_phys.ravel(),
    "y_over_Lp": Y_phys.ravel(),
    "pressure_ratio": P.ravel(),
    "pressure_ratio_uncertainty": P_unc.ravel(),
    "color_distance_rgb": color_dist.ravel(),
    "x_pixel": X_pix.ravel(),
    "y_pixel": Y_pix.ravel(),
})

out_df.to_csv(output_csv, index=False)

print(f"\nSaved digitized data to: {output_csv}")
print(f"Number of rows: {len(out_df)}")


# --------------------------------------------------
# Preview plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.0))

levels = np.linspace(p_min, p_max, 101)

cont = ax.contourf(
    X_phys,
    Y_phys,
    P,
    levels=levels,
    cmap="jet",
    vmin=p_min,
    vmax=p_max,
    extend="both"
)

cbar = fig.colorbar(cont, ax=ax, pad=0.035)
cbar.set_label(r"$P/P_\infty$")

ax.set_xlabel(r"$x/L_p$")
ax.set_ylabel(r"$y/L_p$")
ax.set_xlim(x_min_phys, x_max_phys)
ax.set_ylim(y_min_phys, y_max_phys)
ax.set_title("Digitized Pressure Field from Image")

plt.tight_layout()
plt.savefig(output_preview, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved preview figure to: {output_preview}")