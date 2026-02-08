import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from io import StringIO

st.set_page_config(layout="wide")

# ---------------------------
# Session storage
# ---------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "units" not in st.session_state:
    st.session_state.units = {}

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2 = st.tabs(["TXT → Parquet", "Scatter plots"])

# =====================================================
# TAB 1 — TXT upload and conversion
# =====================================================
with tab1:
    st.header("Upload TXT and convert to Parquet")

    uploaded_txt = st.file_uploader("Upload TXT file", type=["txt"])

    if uploaded_txt is not None:
        content = uploaded_txt.read().decode("utf-8")
        txt_buffer = StringIO(content)

        # Read header and units
        header = pd.read_csv(txt_buffer, sep="\t", nrows=1, header=None)
        txt_buffer.seek(0)
        units = pd.read_csv(txt_buffer, sep="\t", skiprows=1, nrows=1, header=None)

        txt_buffer.seek(0)
        df = pd.read_csv(txt_buffer, sep="\t", skiprows=2, header=None)

        df.columns = header.iloc[0].astype(str)

        # Convert numeric where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Save units
        unit_dict = {}
        for col, unit in zip(df.columns, units.iloc[0]):
            if str(unit) != "-" and str(unit) != "nan":
                unit_dict[col] = str(unit)

        st.session_state.df = df
        st.session_state.units = unit_dict

        # Save parquet locally (optional)
        parquet_name = uploaded_txt.name.replace(".txt", ".parquet")
        df.to_parquet(parquet_name, index=False)

        st.success(f"Converted and loaded: {parquet_name}")
        st.info("Now go to the Scatter plots tab")

# =====================================================
# TAB 2 — Scatter plots
# =====================================================
with tab2:
    st.header("Scatter plots")

    # Upload parquet directly
    uploaded_parquet = st.file_uploader("Upload Parquet file", type=["parquet"])

    if uploaded_parquet is not None:
        df = pd.read_parquet(uploaded_parquet)
        st.session_state.df = df
        st.session_state.units = {}  # no units available

    df = st.session_state.df
    units = st.session_state.units

    if df is None:
        st.warning("Load a TXT or Parquet file first")
        st.stop()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Need at least two numeric columns")
        st.stop()

    # =====================================================
    # SIDEBAR CONTROLS
    # =====================================================
    st.sidebar.header("Plots")

    n_plots = st.sidebar.number_input("Number of plots", 1, 6, 1)
    plots_per_row = st.sidebar.number_input("Plots per row", 1, 3, 2)

    # =====================================================
    # GLOBAL FILTERS — COMPACT UI
    # =====================================================
    st.sidebar.header("Global Filters")

    filter_values = {}
    active_filters = []

    for col in numeric_cols:
        row = st.sidebar.columns([1.5, 1, 1])

        enabled = row[0].checkbox(col, key=f"enable_{col}")

        if enabled:
            min_val = row[1].number_input(
                "min",
                value=float(df[col].min()),
                key=f"{col}_min",
                label_visibility="collapsed"
            )
            max_val = row[2].number_input(
                "max",
                value=float(df[col].max()),
                key=f"{col}_max",
                label_visibility="collapsed"
            )

            filter_values[col] = (min_val, max_val)
            active_filters.append(col)

    # Apply filters globally
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

    # =====================================================
    # PLOTS
    # =====================================================
    plot_index = 0

    for row_start in range(0, n_plots, plots_per_row):
        cols = st.columns(plots_per_row)

        for col_container in cols:
            if plot_index >= n_plots:
                break

            with col_container:
                st.subheader(f"Plot {plot_index + 1}")

                x_col = st.selectbox(
                    "X",
                    numeric_cols,
                    key=f"x_{plot_index}"
                )
                y_col = st.selectbox(
                    "Y",
                    numeric_cols,
                    index=1 if len(numeric_cols) > 1 else 0,
                    key=f"y_{plot_index}"
                )

                # Grid spacing (same for X and Y)
                spacing = st.number_input(
                    "Grid spacing",
                    value=float((df[x_col].max() - df[x_col].min()) / 10),
                    key=f"spacing_{plot_index}"
                )

                # Units
                x_label = f"{x_col} ({units.get(x_col, '')})"
                y_label = f"{y_col} ({units.get(y_col, '')})"

                fig = go.Figure()

                # Plot ALL data first (blue)
                fig.add_trace(go.Scattergl(
                    x=df[x_col],
                    y=df[y_col],
                    mode="markers",
                    marker=dict(color="blue", size=4),
                    name="All data"
                ))

                # Plot filtered data ON TOP (red)
                if has_filter and not df_filtered.empty:
                    fig.add_trace(go.Scattergl(
                        x=df_filtered[x_col],
                        y=df_filtered[y_col],
                        mode="markers",
                        marker=dict(color="red", size=6),
                        name="Filtered"
                    ))

                # Axis ranges
                x_min, x_max = df[x_col].min(), df[x_col].max()
                y_min, y_max = df[y_col].min(), df[y_col].max()

                # Grid values
                x_ticks = np.arange(x_min, x_max + spacing, spacing)
                y_ticks = np.arange(y_min, y_max + spacing, spacing)

                fig.update_layout(
                    template="plotly_white",
                    height=400,
                    xaxis=dict(
                        title=x_label,
                        tickmode="array",
                        tickvals=x_ticks,
                        showgrid=True,
                        gridcolor="lightgray",
                        griddash="solid",
                        tickformat=",.0f"
                    ),
                    yaxis=dict(
                        title=y_label,
                        tickmode="array",
                        tickvals=y_ticks,
                        showgrid=True,
                        gridcolor="lightgray",
                        griddash="solid",
                        tickformat=",.0f"
                    ),
                    margin=dict(l=40, r=20, t=40, b=40),
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

            plot_index += 1
