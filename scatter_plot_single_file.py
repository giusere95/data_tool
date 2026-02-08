import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO

st.set_page_config(layout="wide")

# =========================
# CONFIG
# =========================
DATA_FOLDER = "Data"
os.makedirs(DATA_FOLDER, exist_ok=True)

st.title("Data Tool – Scatter & Converter")

# =========================
# FUNCTIONS
# =========================

def txt_to_dataframe(uploaded_file):
    """Convert TXT (tab separated) to DataFrame and extract units"""
    df_raw = pd.read_csv(uploaded_file, sep="\t", header=None)

    headers = df_raw.iloc[0].tolist()
    units = df_raw.iloc[1].tolist()

    df = df_raw.iloc[2:].reset_index(drop=True)
    df.columns = headers

    # Convert numeric where possible
    df = df.apply(pd.to_numeric, errors="ignore")

    return df, dict(zip(headers, units))


def dataframe_to_parquet_bytes(df):
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    return buffer


# =========================
# SIDEBAR – DATA SOURCES
# =========================
st.sidebar.header("Data Sources")

dataframes = []
column_units = {}

# ---- Existing Parquet files ----
parquet_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".parquet")]

if parquet_files:
    selected_parquets = st.sidebar.multiselect(
        "Select Parquet files",
        parquet_files
    )

    for f in selected_parquets:
        path = os.path.join(DATA_FOLDER, f)
        df_pq = pd.read_parquet(path)
        dataframes.append(df_pq)

# =========================
# TXT UPLOAD SECTION
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("Upload TXT")

uploaded_txt_files = st.sidebar.file_uploader(
    "Upload TXT files",
    type=["txt"],
    accept_multiple_files=True
)

uploaded_data = []

if uploaded_txt_files:
    for uploaded_file in uploaded_txt_files:
        df_txt, units_txt = txt_to_dataframe(uploaded_file)
        uploaded_data.append((uploaded_file.name, df_txt, units_txt))

# =========================
# MAIN AREA – PREVIEW & SAVE
# =========================
if uploaded_data:
    st.header("Uploaded TXT Preview")

    for name, df_txt, units_txt in uploaded_data:
        with st.expander(f"{name} ({len(df_txt)} rows)"):
            st.dataframe(df_txt.head(50))

            col1, col2 = st.columns(2)

            # Save to server
            with col1:
                if st.button(f"Save {name} as Parquet"):
                    save_name = name.replace(".txt", ".parquet")
                    path = os.path.join(DATA_FOLDER, save_name)
                    df_txt.to_parquet(path, index=False)
                    st.success(f"Saved to Data/{save_name}")

            # Download
            with col2:
                parquet_bytes = dataframe_to_parquet_bytes(df_txt)
                st.download_button(
                    label="Download Parquet",
                    data=parquet_bytes,
                    file_name=name.replace(".txt", ".parquet"),
                    mime="application/octet-stream"
                )

        # Add to active dataset
        dataframes.append(df_txt)
        column_units.update(units_txt)

# =========================
# STOP IF NO DATA
# =========================
if not dataframes:
    st.info("Upload TXT files or select Parquet files from sidebar.")
    st.stop()

# =========================
# MERGE DATA
# =========================
df = pd.concat(dataframes, ignore_index=True)

# =========================
# FILTER UI (Professional Compact)
# =========================
st.sidebar.markdown("---")
st.sidebar.header("Filters")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
filters = {}

for col in numeric_cols:
    with st.sidebar.expander(col, expanded=False):
        min_val = float(df[col].min())
        max_val = float(df[col].max())

        c1, c2 = st.columns(2)
        with c1:
            min_input = st.number_input(
                "Min",
                value=min_val,
                key=f"{col}_min"
            )
        with c2:
            max_input = st.number_input(
                "Max",
                value=max_val,
                key=f"{col}_max"
            )

        filters[col] = (min_input, max_input)

# Apply filters
for col, (min_v, max_v) in filters.items():
    df = df[(df[col] >= min_v) & (df[col] <= max_v)]

# =========================
# PLOT SECTION
# =========================
st.sidebar.markdown("---")
st.sidebar.header("Plot")

all_columns = df.columns.tolist()

x_col = st.sidebar.selectbox("X axis", all_columns)
y_col = st.sidebar.selectbox("Y axis", all_columns)

st.subheader("Scatter Plot")
st.write(f"Rows after filtering: {len(df)}")

st.scatter_chart(df[[x_col, y_col]])

# =========================
# DOWNLOAD FILTERED DATA
# =========================
st.markdown("---")
st.subheader("Export filtered data")

filtered_bytes = dataframe_to_parquet_bytes(df)

st.download_button(
    label="Download filtered Parquet",
    data=filtered_bytes,
    file_name="filtered.parquet",
    mime="application/octet-stream"
)
