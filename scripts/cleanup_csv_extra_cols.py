import pandas as pd
from pathlib import Path

# -----------------------------
# User inputs
# -----------------------------
REPO_DIR = Path(__file__).parent.parent
PROJECT = "2023Avaition"
IN_CSV = "panel_pressure.csv"
OUT_CSV  = "panel_pressure_clean_columns.csv"

csv_folder = REPO_DIR/ "csv_files"/ PROJECT
input_csv = csv_folder/ IN_CSV      # change this
output_csv = csv_folder / OUT_CSV


# -----------------------------
# Read CSV
# -----------------------------
df = pd.read_csv(input_csv)

# Option 1: keep only the columns you want
columns_to_keep = ["x", "y", "z", "pressure"]
df_selected = df[columns_to_keep]

# -----------------------------
# Write cleaned CSV
# -----------------------------
df_selected.to_csv(output_csv, index=False)

print(f"Input columns: {list(df.columns)}")
print(f"Output columns: {list(df_selected.columns)}")
print(f"Saved cleaned CSV to: {output_csv}")