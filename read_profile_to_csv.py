from pathlib import Path
import re
import pandas as pd

# --------------------------------------------------
# User inputs
# --------------------------------------------------
PROFILE_FILE = "4deg_steep_back_SST_steady_panel_pressure.prof"
input_file = Path(PROFILE_FILE)   # change this
OUTPUT_FILE = "panel_pressure.csv"
output_file = Path(OUTPUT_FILE)   # change this

# --------------------------------------------------
# Parser for Fluent profile format
# --------------------------------------------------
def read_fluent_profile(filename):
    """
    Reads a Fluent .prof profile file of the form:

    ((profile_name point N)
    (x
      ...
    )
    (y
      ...
    )
    (pressure
      ...
    )
    )

    Returns:
        pandas DataFrame with one column per profile variable.
    """

    data = {}
    current_var = None
    current_values = []

    numeric_pattern = re.compile(
        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    )

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            # Start of a variable block, e.g. "(x" or "(pressure"
            if line.startswith("(") and not line.startswith("(("):
                var_name = line[1:].strip()

                # ignore empty or malformed variable names
                if var_name:
                    current_var = var_name
                    current_values = []

                continue

            # End of a variable block
            if line == ")":
                if current_var is not None:
                    data[current_var] = current_values
                    current_var = None
                    current_values = []
                continue

            # Numeric value inside a variable block
            if current_var is not None:
                match = numeric_pattern.fullmatch(line)
                if match:
                    current_values.append(float(line))

    # Check column lengths
    lengths = {key: len(value) for key, value in data.items()}

    if not lengths:
        raise ValueError("No profile data found. Check the input file format.")

    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        raise ValueError(
            "Profile variables have different lengths:\n"
            + "\n".join(f"{k}: {v}" for k, v in lengths.items())
        )

    return pd.DataFrame(data)


# --------------------------------------------------
# Convert to CSV
# --------------------------------------------------
df = read_fluent_profile(input_file)
df.to_csv(output_file, index=False)

print(f"Read variables: {list(df.columns)}")
print(f"Number of points: {len(df)}")
print(f"Saved CSV to: {output_file}")