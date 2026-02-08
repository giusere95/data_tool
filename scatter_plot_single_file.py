import streamlit as st
import pandas as pd
import plotly.express as px
import duckdb
import json
import os

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(layout="wide")
st.title("Data Scatter Explorer")

DATA_FOLDER = "Data"
UNITS_FILE = "column_units.json"

# =====================
# LOAD UNITS
# =====================
if os.path.exists(UNITS_FILE):
    with open(UNITS_FILE, "r") as f:
        column_units = json.load(f)
else:
    column_units = {}

# =====================
# LOAD PARQUET FILES
# =====================
parquet_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".parquet")]

selected_files = st.sidebar.multiselect(
    "Select file(s)",
    parquet_files
)

if not selected_files:
    st.info("Select at least one file.")
    st.stop()

# =====================
# LOAD DATA WITH DUCKDB
# =====================
con = duckdb.connect()
file_paths = [os.path.join(DATA_FOLDER, f) for f in selected_files]

query = " UNION ALL ".join([f"SELECT * FROM read_parquet('{p}')" for p in file_paths])
df = con.execute(query).fetch_df()

# Remove non-data rows if present
df = df.iloc[2:].reset_index(drop=True)

# Convert to numeric when possible
df = df.apply(pd.to_numeric, errors="ignore")

columns = list(df.columns)

# =====================
# SIDEBAR – FILTERS (COMPACT PROFESSIONAL UI)
# =====================
st.sidebar.markdown("## Filters")

selected_filter_columns = st.sidebar.multiselect(
    "Variables to filter",
    columns
)

filters = {}

for col in selected_filter_columns:
    col_min = pd.to_numeric(df[col], errors="coerce").min()
    col_max = pd.to_numeric(df[col], errors="coerce").max()

    c1, c2 = st.sidebar.columns(2)
    with c1:
        min_val = st.number_input(
            f"{col} min",
            value=float(col_min),
            key=f"{col}_min"
        )
    with c2:
        max_val = st.number_input(
            f"{col} max",
            value=float(col_max),
            key=f"{col}_max"
        )

    filters[col] = (min_val, max_val)

# Apply filters
df_filtered = df.copy()
for col, (min_val, max_val) in filters.items():
    df_filtered = df_filtered[
        (pd.to_numeric(df_filtered[col], errors="coerce") >= min_val) &
        (pd.to_numeric(df_filtered[col], errors="coerce") <= max_val)
    ]

# Create highlight column
df["__filtered__"] = df.index.isin(df_filtered.index)
df["__filtered__"] = df["__filtered__"].map({True: "Filtered", False: "Other"})

# =====================
# PLOT SETTINGS
# =====================
st.sidebar.markdown("## Plots")

num_plots = st.sidebar.number_input(
    "Number of plots",
    min_value=1,
    max_value=6,
    value=2
)

plots_per_row = st.sidebar.number_input(
    "Plots per row",
    min_value=1,
    max_value=3,
    value=2
)

# =====================
# DISPLAY PLOTS
# =====================
for row_start in range(0, num_plots, plots_per_row):

    cols_layout = st.columns(plots_per_row)

    for i in range(plots_per_row):
        plot_index = row_start + i
        if plot_index >= num_plots:
            break

        with cols_layout[i]:

            st.markdown(f"### Plot {plot_index + 1}")

            x_col = st.selectbox(
                "X axis",
                columns,
                key=f"x{plot_index}"
            )

            y_col = st.selectbox(
                "Y axis",
                columns,
                key=f"y{plot_index}"
            )

            # Units
            x_unit = column_units.get(x_col, "")
            y_unit = column_units.get(y_col, "")

            # Grid spacing (typed)
            st.markdown("Grid spacing")
            x_spacing = st.number_input(
                "X spacing",
                min_value=1.0,
                value=1000.0,
                key=f"xspace{plot_index}"
            )

            y_spacing = st.number_input(
                "Y spacing",
                min_value=1.0,
                value=1000.0,
                key=f"yspace{plot_index}"
            )

            # Scatter plot
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color="__filtered__",
                color_discrete_map={
                    "Filtered": "red",
                    "Other": "lightgray"
                }
            )

            # Axis titles with units
            fig.update_layout(
                xaxis_title=f"{x_col} ({x_unit})" if x_unit else x_col,
                yaxis_title=f"{y_col} ({y_unit})" if y_unit else y_col,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(color="black")
            )

            # Full numbers (no 6M)
            fig.update_xaxes(tickformat=",.0f")
            fig.update_yaxes(tickformat=",.0f")

            # Grid spacing
            fig.update_xaxes(
                dtick=x_spacing,
                showgrid=True,
                gridcolor="lightgray"
            )

            fig.update_yaxes(
                dtick=y_spacing,
                showgrid=True,
                gridcolor="lightgray"
            )

            st.plotly_chart(fig, use_container_width=True)
