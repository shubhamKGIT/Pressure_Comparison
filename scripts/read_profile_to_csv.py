from pathlib import Path
import os
from utils import read_fluent_profile

# --------------------------------------------------
# User inputs
# --------------------------------------------------
REPO_DIR = Path(__file__).parent.parent
PROFILE_FILE_NAME = "panel_pressure_2023Avia_VisSol.prof"
PROJECT = "2023_Aviation"
OUTPUT_FILE = "panel_pressure_2023Avia_VisSol.csv"

# --------------------------------------------------
# Setting up Directories
# --------------------------------------------------
project_dir = REPO_DIR/ "csv_files"/ PROJECT
if not os.path.exists(project_dir):
    os.makedirs(project_dir)

profile_dir = REPO_DIR / "input_profiles"
profile_file = profile_dir / PROFILE_FILE_NAME
input_file = Path(profile_file)   # change this
output_file = Path(REPO_DIR/ "csv_files"/ PROJECT/ OUTPUT_FILE)   # change this

# --------------------------------------------------
# Convert to CSV
# --------------------------------------------------
df = read_fluent_profile(input_file)
df.to_csv(output_file, index=False)

print(f"Read variables: {list(df.columns)}")
print(f"Number of points: {len(df)}")
print(f"Saved CSV to: {output_file}")