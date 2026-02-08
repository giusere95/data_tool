import pandas as pd
import streamlit as st
import plotly.express as px
import json
import math

st.set_page_config(page_title="Live Scatter Dashboard", layout="wide")

# -------- Sidebar: Data Selection --------
st.sidebar.title("Data Selection")
uploaded_file = st.sidebar.file_uploader("Select Parquet file", type=["parquet"])
units_file = st.sidebar.file_uploader("Select Units JSON file", type=["json"])

if uploaded_file is not None and units_file is not None:
    # Load data
    df = pd.read_parquet(uploaded_file)
    units = json.load(units_file)

    # -------- Sidebar: Filter Configuration --------
    st.sidebar.subheader("Filters")
    num_filters = st.sidebar.number_input("Number of filters", min_value=0, max_value=6, value=0)

    filters = {}
    for i in range(num_filters):
        col = st.sidebar.selectbox(f"Filter Column {i+1}", df.columns, key=f"filter_col_{i}")
        cond = st.sidebar.text_input(f"Condition for {col} (e.g., '> 50')", key=f"filter_val_{i}")
        if cond.strip():
            filters[col] = cond.strip()

    # Apply filters robustly
    df_filtered = df.copy()
    for col, cond in filters.items():
        numeric_col = pd.to_numeric(df_filtered[col], errors="coerce")
        try:
            mask = eval(f"numeric_col {cond}")
            df_filtered = df_filtered[mask]
        except Exception:
            st.warning(f"Could not apply filter '{cond}' to column '{col}'")

    # Color mask for plotting
    color_mask = pd.Series(df.index.isin(df_filtered.index)).map({True: "Filtered", False: "Other"})

    # -------- Sidebar: Scatter Plots Configuration --------
    st.sidebar.subheader("Scatter Plots Configuration")
    num_plots = st.sidebar.number_input("Number of scatter plots", min_value=1, max_value=6, value=2)

    # -------- Dashboard: Scatter Plots --------
    st.subheader("Scatter Plots Dashboard")

    plots_per_row = 2  # number of plots per row
    num_rows = math.ceil(num_plots / plots_per_row)

    plot_index = 0
    for row in range(num_rows):
        cols_in_row = min(plots_per_row, num_plots - plot_index)
        cols = st.columns(cols_in_row)

        for col_idx in range(cols_in_row):
            with cols[col_idx]:
                st.markdown(f"**Plot {plot_index+1}**")

                # X/Y selection
                x_col = st.selectbox(f"X-axis", df.columns, key=f"x{plot_index}")
                y_col = st.selectbox(f"Y-axis", df.columns, key=f"y{plot_index}")

                # Units for axes
                x_unit = units.get(x_col, "")
                y_unit = units.get(y_col, "")
                x_label = f"{x_col} ({x_unit})" if x_unit else x_col
                y_label = f"{y_col} ({y_unit})" if y_unit else y_col

                # -------- Per-plot Grid Spacing Number Inputs --------
                st.markdown("**Adjust Grid Line Spacing**")
                x_spacing = st.number_input(
                    "X-axis spacing",
                    min_value=1,
                    max_value=1000000,
                    value=1000,
                    step=1,
                    key=f"xspacing{plot_index}"
                )
                y_spacing = st.number_input(
                    "Y-axis spacing",
                    min_value=1,
                    max_value=1000000,
                    value=1000,
                    step=1,
                    key=f"yspacing{plot_index}"
                )

                # Scatter plot
                fig = px.scatter(
                    df, x=x_col, y=y_col,
                    color=color_mask,
                    color_discrete_map={"Filtered": "red", "Other": "blue"},
                    opacity=0.6,
                    title=f"{x_col} vs {y_col}",
                    width=500,   # fixed width
                    height=400   # fixed height
                )

                # Update layout: readable white background, axes, ticks, grid spacing
                fig.update_layout(
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    xaxis_type="linear",
                    yaxis_type="linear",
                    xaxis_tickformat=',',
                    yaxis_tickformat=',',
                    xaxis_dtick=x_spacing,   # spacing between X grid lines
                    yaxis_dtick=y_spacing,   # spacing between Y grid lines
                    xaxis_showgrid=True,
                    yaxis_showgrid=True,
                    xaxis_gridcolor="gray",
                    yaxis_gridcolor="gray",
                    xaxis_gridwidth=1,
                    yaxis_gridwidth=1,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis_linecolor="black",
                    yaxis_linecolor="black",
                    xaxis_tickcolor="black",
                    yaxis_tickcolor="black",
                    xaxis_tickfont=dict(color="black"),
                    yaxis_tickfont=dict(color="black"),
                    xaxis_title_font=dict(color="black"),
                    yaxis_title_font=dict(color="black"),
                    title_font=dict(color="black")
                )

                st.plotly_chart(fig, use_container_width=True)
                plot_index += 1
