import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide")

DATA_FOLDER = "Data"

st.title("Interactive Scatter Tool")

# ==============================
# Load parquet file
# ==============================
uploaded_parquet = st.file_uploader(
    "Upload Parquet file", type=["parquet"]
)

if uploaded_parquet is not None:
    df = pd.read_parquet(uploaded_parquet)
else:
    parquet_files = []
    if os.path.exists(DATA_FOLDER):
        parquet_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".parquet")]

    if not parquet_files:
        st.warning("Upload a parquet file or place files inside /Data")
        st.stop()

    selected_file = st.selectbox("Select file", parquet_files)
    df = pd.read_parquet(os.path.join(DATA_FOLDER, selected_file))

# ==============================
# Units extraction (row 2)
# ==============================
units_dict = {}
if len(df) > 1:
    units_row = df.iloc[1]
    for col in df.columns:
        unit = str(units_row[col])
        if unit != "-" and unit != "nan":
            units_dict[col] = unit
        else:
            units_dict[col] = ""

# Remove header rows if present
try:
    df = df.iloc[2:].reset_index(drop=True)
except:
    pass

# Convert numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Need at least two numeric columns")
    st.stop()

# ==============================
# Sidebar controls
# ==============================
st.sidebar.header("Layout")

num_plots = st.sidebar.number_input("Number of plots", 1, 6, 1)
plots_per_row = st.sidebar.number_input("Plots per row", 1, 3, 2)

# ==============================
# Filters (clean UI)
# ==============================
st.sidebar.header("Filters")

selected_filter_cols = st.sidebar.multiselect(
    "Select variables to filter",
    numeric_cols
)

filter_values = {}
active_filters = []

for col in selected_filter_cols:
    st.sidebar.markdown(f"**{col}**")
    row = st.sidebar.columns([1, 1, 1])

    enabled = row[0].checkbox("On", key=f"enable_{col}")

    if enabled:
        min_val = row[1].number_input(
            "min",
            value=float(df[col].min()),
            label_visibility="collapsed",
            key=f"{col}_min"
        )
        max_val = row[2].number_input(
            "max",
            value=float(df[col].max()),
            label_visibility="collapsed",
            key=f"{col}_max"
        )

        filter_values[col] = (min_val, max_val)
        active_filters.append(col)

# Apply filters
if active_filters:
    df_filtered = df.copy()
    for col, (mn, mx) in filter_values.items():
        df_filtered = df_filtered[
            (df_filtered[col] >= mn) & (df_filtered[col] <= mx)
        ]
    has_filter = True
else:
    df_filtered = pd.DataFrame()
    has_filter = False

# ==============================
# Plot grid
# ==============================
rows = int(np.ceil(num_plots / plots_per_row))
plot_index = 0

for r in range(rows):
    cols = st.columns(plots_per_row)

    for c in range(plots_per_row):
        if plot_index >= num_plots:
            break

        with cols[c]:
            st.markdown(f"### Plot {plot_index + 1}")

            x_col = st.selectbox(
                "X",
                numeric_cols,
                key=f"x_{plot_index}"
            )
            y_col = st.selectbox(
                "Y",
                numeric_cols,
                index=1,
                key=f"y_{plot_index}"
            )

            # ==============================
            # Grid spacing (safe)
            # ==============================
            x_min, x_max = df[x_col].min(), df[x_col].max()
            y_min, y_max = df[y_col].min(), df[y_col].max()

            default_spacing = (x_max - x_min) / 10 if (x_max - x_min) != 0 else 1.0

            spacing = st.number_input(
                "Grid spacing",
                min_value=0.000001,
                value=float(default_spacing),
                format="%f",
                key=f"spacing_{plot_index}"
            )

            if spacing <= 0 or np.isnan(spacing):
                spacing = default_spacing

            # Limit number of grid lines (performance protection)
            max_lines = 200

            def safe_ticks(min_v, max_v, step):
                if step <= 0:
                    return None
                count = (max_v - min_v) / step
                if count > max_lines:
                    step = (max_v - min_v) / max_lines
                return np.arange(min_v, max_v + step, step)

            x_ticks = safe_ticks(x_min, x_max, spacing)
            y_ticks = safe_ticks(y_min, y_max, spacing)

            # ==============================
            # Build figure
            # ==============================
            fig = go.Figure()

            # Base points (unfiltered only)
            if has_filter and not df_filtered.empty:
                df_base = df.drop(df_filtered.index)
            else:
                df_base = df

            fig.add_trace(
                go.Scatter(
                    x=df_base[x_col],
                    y=df_base[y_col],
                    mode="markers",
                    marker=dict(color="blue", size=6),
                    name="All",
                    opacity=0.5
                )
            )

            # Filtered points on top
            if has_filter and not df_filtered.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_filtered[x_col],
                        y=df_filtered[y_col],
                        mode="markers",
                        marker=dict(
                            color="red",
                            size=9,
                            line=dict(width=1, color="black")
                        ),
                        name="Filtered"
                    )
                )

            # ==============================
            # Layout
            # ==============================
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=400,
                margin=dict(l=40, r=20, t=40, b=40),
                xaxis=dict(
                    title=f"{x_col} ({units_dict.get(x_col,'')})",
                    tickmode="array" if x_ticks is not None else "auto",
                    tickvals=x_ticks,
                    showgrid=True,
                    gridcolor="lightgray",
                    griddash="solid"
                ),
                yaxis=dict(
                    title=f"{y_col} ({units_dict.get(y_col,'')})",
                    tickmode="array" if y_ticks is not None else "auto",
                    tickvals=y_ticks,
                    showgrid=True,
                    gridcolor="lightgray",
                    griddash="solid"
                ),
                legend=dict(orientation="h")
            )

            st.plotly_chart(fig, use_container_width=True)

        plot_index += 1
