# X-shift alignment scripts

These scripts extend the existing simulated-vs-digitized pressure comparison workflow.

## Run order

1. Run:

```bash
python diagnose_x_shift_least_error.py
```

This estimates the streamwise shift that minimizes centerline pressure error.

2. Then run:

```bash
python apply_x_shift_to_simulation_csv.py
```

This applies the fitted shift to the raw simulation CSV and writes a shifted simulation CSV.

## Sign convention

The fitted shift is applied to the simulation coordinate:

```python
x_sim_shifted_over_Lp = x_sim_over_Lp + best_shift_over_Lp
```

So:

- positive shift moves the simulation field downstream/right,
- negative shift moves the simulation field upstream/left.

The physical shifted x-coordinate is:

```python
x_shifted = x_original + best_shift_over_Lp * Lp
```

## Main outputs

Diagnostic outputs:

```text
results/<PROJECT>/<FILE_FOLDER>/x_shift_alignment/x_shift_diagnostic_summary.csv
results/<PROJECT>/<FILE_FOLDER>/x_shift_alignment/x_shift_rmse_curve.png
results/<PROJECT>/<FILE_FOLDER>/x_shift_alignment/x_shift_centerline_before_after.png
```

Shifted simulation outputs:

```text
results/<PROJECT>/<FILE_FOLDER>/x_shifted_simulation/<FILE_FOLDER>_panel_pressure_sim_x_shifted.csv
results/<PROJECT>/<FILE_FOLDER>/x_shifted_simulation/pressure_comparison_2d_after_x_shift.png
results/<PROJECT>/<FILE_FOLDER>/x_shifted_simulation/pressure_centerline_comparison_after_x_shift.png
```

## Important settings

In the diagnostic script, check:

```python
shift_min_over_Lp = -0.20
shift_max_over_Lp = 0.20
fit_x_min = 0.05
fit_x_max = 0.95
use_uncertainty_weighting = False
allow_constant_pressure_bias = False
```

If only the shock location is misaligned, keep `allow_constant_pressure_bias = False`.
If you only want to align shape and ignore a constant pressure offset, set it to `True`.
