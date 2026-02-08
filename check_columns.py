import pandas as pd

#name of the parquet file
PARQUET_FILE="data.parquet"

#load the parquet file (all columns)
df = pd.read_parquet("data.parquet")

#print first 10 column names
print("First 10 columns:", df.columns[:10])

#print the shape of the table (rows, columns)
print("Shape:", df.shape)