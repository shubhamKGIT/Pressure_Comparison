from pathlib import Path
import os
from utils import read_fluent_profile

# --------------------------------------------------
# User inputs
# --------------------------------------------------
REPO_DIR = Path(__file__).parent.parent
PROJECT = "SBLI_challenge"
FILE_FOLDER =  "case3"
PROFILE_FILE_NAME = "panel_pressure_sim.prof"
OUTPUT_FILE = "panel_pressure_sim.csv"

# --------------------------------------------------
# Setting up Directories
# --------------------------------------------------
project_dir = REPO_DIR/ "csv_files"/ PROJECT # set the project directory for csv files to live
if not os.path.exists(project_dir):
    os.makedirs(project_dir)

profile_dir = REPO_DIR / "input_profiles"/ PROJECT/ FILE_FOLDER # set the profile folder
if not os.path.exists(profile_dir):
    os.makedirs(profile_dir)

profile_file = profile_dir / PROFILE_FILE_NAME    # get the profile file here

if not os.path.exists(project_dir):
    os.makedirs(project_dir)
output_full_filename =  OUTPUT_FILE 
output_folder = project_dir/ FILE_FOLDER 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_file = output_folder/output_full_filename  # change this
print(f"output in: {output_file.resolve()}")

# --------------------------------------------------
# Convert to CSV
# --------------------------------------------------
df = read_fluent_profile(profile_file)
df.to_csv(output_file, index=False)

print(f"Read variables: {list(df.columns)}")
print(f"Number of points: {len(df)}")
print(f"Saved CSV to: {output_file}")