import pandas as pd

PARQUET_FILE = "data.parquet"

print("Reading only 3 columns...")

# Read only specific columns (by index)
df = pd.read_parquet(
    PARQUET_FILE,
    columns=["0", "1", "2"]
)

print("Shape:", df.shape)
print(df.head())