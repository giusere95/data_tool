import pandas as pd
import json

# -------- SETTINGS --------
TXT_FILE = "20260205_NEF77_Alpha_2025008.32_DOE_1300rpmx800Nm_Model_30k_Points.txt"       # your txt file name
PARQUET_FILE = "data.parquet"
UNITS_FILE = "units.json"
# --------------------------

print("Reading TXT file...")

# Read file (tab separated)
df = pd.read_csv(
    TXT_FILE,
    sep="\t",
    header=0,          # first row = column names
    dtype=str          # read everything as text first
)

print("Extracting units (row 2)...")

# Second row contains units
units_row = df.iloc[0]
units = units_row.to_dict()

# Remove the units row
df = df.iloc[1:].reset_index(drop=True)

print("Converting data to numeric...")

# Convert all columns to numeric (NaN handled automatically)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Saving Parquet...")
df.to_parquet(PARQUET_FILE, index=False)

print("Saving units separately...")
with open(UNITS_FILE, "w") as f:
    json.dump(units, f, indent=2)

print("Done!")
print("Rows:", len(df))
print("Columns:", len(df.columns))
