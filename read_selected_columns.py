import pandas as pd

PARQUET_FILE = "data.parquet"

# List the columns you want to load
# Column names are strings, e.g., "0", "10", "25"
columns_to_read = ["1", "2", "3"]

print("Reading selected columns...")

# Load only selected columns
df = pd.read_parquet(PARQUET_FILE, columns=columns_to_read)

print("Shape of loaded data:", df.shape)
print(df.head())