import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

st.set_page_config(layout="wide")
st.title("Data Tool")

# =====================================================
# TABS
# =====================================================
tab_convert, tab_plot = st.tabs(["TXT → Parquet", "Scatter plots"])

# =====================================================
# TAB 1 — TXT to Parquet
# =====================================================
with tab_convert:
    st.header("Convert TXT to Parquet")

    uploaded_txt = st.file_uploader("Upload TXT file", type=["txt"])

    if uploaded_txt is not None:
        # Read tab-separated file
        df_raw = pd.read_csv(uploaded_txt, sep="\t", header=None)

        # First row = column names
        df_raw.columns = df_raw.iloc[0]

        # Keep first two rows (names + units)
        df_parquet = df_raw.copy()

        # Download parquet
        buffer = io.BytesIO()
        df_parquet.to_parquet(buffer, index=False)

        st.success("File ready (units preserved)")

        st.download_button(
            "Download Parquet",
            buffer.getvalue(),
            file_name="converted.parquet"
        )

# =====================================================
# TAB 2 — SCATTER TOOL
# =====================================================
with tab_plot:

    st.header("Interactive Scatter")

    uploaded_parquet = st.file_uploader(
        "Upload Parquet file",
        type=["parquet"],
        key="parquet_uploader"
    )

    if uploaded_parquet is None:
        st.info("Upload a parquet file")
        st.stop()

    df_raw = pd.read_parquet(uploaded_parquet)

    # =====================================================
    # Extract units from second row
    # =====================================================
    units_dict = {}
    if len(df_raw) > 1:
        units_row = df_raw.iloc[1]
        for col in df_raw.columns:
            val = str(units_row[col])
            if val != "-" and val.lower() != "nan":
                units_dict[col] = val
            else:
                units_dict[col] = ""

        # Actual data starts from row 3
        df = df_raw.iloc[2:].reset_index(drop=True)
    else:
        df = df_raw
        units_dict = {col: "" for col in df.columns}

    # Convert numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Need at least two numeric columns")
        st.stop()

    # =====================================================
    # Sidebar — layout
    # =====================================================
    st.sidebar.header("Layout")
    num_plots = st.sidebar.number_input("Number of plots", 1, 6, 1)
    plots_per_row = st.sidebar.number_input("Plots per row", 1, 3, 2)

    # =====================================================
    # Sidebar — filters
    # =====================================================
    st.sidebar.header("Filters")

    selected_filter_cols = st.sidebar.multiselect(
        "Select variables",
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
                (df_filtered[col] >= mn) &
                (df_filtered[col] <= mx)
            ]
        has_filter = True
    else:
        df_filtered = pd.DataFrame()
        has_filter = False

    # =====================================================
    # Plot grid
    # =====================================================
    rows = int(np.ceil(num_plots / plots_per_row))
    plot_index = 0

    for r in range(rows):
        cols = st.columns(plots_per_row)

        for c in range(plots_per_row):
            if plot_index >= num_plots:
                break

            with cols[c]:
                st.markdown(f"### Plot {plot_index + 1}")

                # Compact X/Y selection
                xy_row = st.columns(2)
                x_col = xy_row[0].selectbox(
                    "X",
                    numeric_cols,
                    key=f"x_{plot_index}"
                )
                y_col = xy_row[1].selectbox(
                    "Y",
                    numeric_cols,
                    index=1,
                    key=f"y_{plot_index}"
                )

                # Axis ranges
                x_min_data, x_max_data = df[x_col].min(), df[x_col].max()
                y_min_data, y_max_data = df[y_col].min(), df[y_col].max()

                # Spacing
                ctrl1 = st.columns(2)
                x_spacing = ctrl1[0].number_input(
                    "X spacing",
                    min_value=0.000001,
                    value=float((x_max_data - x_min_data) / 10 if x_max_data != x_min_data else 1.0),
                    key=f"xsp_{plot_index}"
                )
                y_spacing = ctrl1[1].number_input(
                    "Y spacing",
                    min_value=0.000001,
                    value=float((y_max_data - y_min_data) / 10 if y_max_data != y_min_data else 1.0),
                    key=f"ysp_{plot_index}"
                )

                # Decimals
                ctrl2 = st.columns(2)
                x_decimals = ctrl2[0].number_input(
                    "X decimals", 0, 10, 2,
                    key=f"xdec_{plot_index}"
                )
                y_decimals = ctrl2[1].number_input(
                    "Y decimals", 0, 10, 2,
                    key=f"ydec_{plot_index}"
                )

                # Axis start
                ctrl3 = st.columns(2)
                x_start = ctrl3[0].number_input(
                    "X start",
                    value=float(x_min_data),
                    key=f"xstart_{plot_index}"
                )
                y_start = ctrl3[1].number_input(
                    "Y start",
                    value=float(y_min_data),
                    key=f"ystart_{plot_index}"
                )

                # Safe ticks
                MAX_LINES = 200

                def safe_ticks(start, max_v, step):
                    if step <= 0:
                        return None
                    count = (max_v - start) / step
                    if count > MAX_LINES:
                        step = (max_v - start) / MAX_LINES
                    return np.arange(start, max_v + step, step)

                x_ticks = safe_ticks(x_start, x_max_data, x_spacing)
                y_ticks = safe_ticks(y_start, y_max_data, y_spacing)

                # =====================================================
                # Plot
                # =====================================================
                fig = go.Figure()

                # Base points
                if has_filter and not df_filtered.empty:
                    df_base = df.drop(df_filtered.index)
                else:
                    df_base = df

                fig.add_trace(go.Scatter(
                    x=df_base[x_col],
                    y=df_base[y_col],
                    mode="markers",
                    marker=dict(color="blue", size=6),
                    opacity=0.5,
                    name="All"
                ))

                # Filtered points (solid red, no border)
                if has_filter and not df_filtered.empty:
                    fig.add_trace(go.Scatter(
                        x=df_filtered[x_col],
                        y=df_filtered[y_col],
                        mode="markers",
                        marker=dict(color="red", size=9),
                        name="Filtered"
                    ))

                # Axis labels with units
                x_unit = units_dict.get(x_col, "")
                y_unit = units_dict.get(y_col, "")

                fig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    height=400,
                    margin=dict(l=40, r=20, t=40, b=40),
                    xaxis=dict(
                        title=f"{x_col} ({x_unit})" if x_unit else x_col,
                        tickmode="array" if x_ticks is not None else "auto",
                        tickvals=x_ticks,
                        tickformat=f".{int(x_decimals)}f",
                        showgrid=True,
                        gridcolor="lightgray"
                    ),
                    yaxis=dict(
                        title=f"{y_col} ({y_unit})" if y_unit else y_col,
                        tickmode="array" if y_ticks is not None else "auto",
                        tickvals=y_ticks,
                        tickformat=f".{int(y_decimals)}f",
                        showgrid=True,
                        gridcolor="lightgray"
                    ),
                    legend=dict(orientation="h")
                )

                st.plotly_chart(fig, use_container_width=True)

            plot_index += 1
