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
        df_raw = pd.read_csv(uploaded_txt, sep="\t", header=None)

        # First row = column names
        df_raw.columns = df_raw.iloc[0]

        buffer = io.BytesIO()
        df_raw.to_parquet(buffer, index=False)

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
    # Extract units (row 2)
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
    # Sidebar — Layout (compact)
    # =====================================================
    st.sidebar.header("Layout")

    colA, colB = st.sidebar.columns(2)
    num_plots = colA.number_input(
        "Plots", 1, 6, 1,
        key="num_plots",
        label_visibility="collapsed"
    )
    plots_per_row = colB.number_input(
        "Per row", 1, 3, 2,
        key="plots_row",
        label_visibility="collapsed"
    )

    # =====================================================
    # Sidebar — Filters
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
    # Sidebar — Download filtered data
    # =====================================================
    if has_filter and not df_filtered.empty:
        csv_buffer = df_filtered.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button(
            "Download filtered data",
            csv_buffer,
            file_name="filtered_data.csv"
        )

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

                # X/Y selection compact
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

                # Data ranges
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

                # Safe ticks
                MAX_LINES = 200

                def safe_ticks(min_v, max_v, step):
                    if step <= 0:
                        return None
                    count = (max_v - min_v) / step
                    if count > MAX_LINES:
                        step = (max_v - min_v) / MAX_LINES
                    return np.arange(min_v, max_v + step, step)

                x_ticks = safe_ticks(x_min_data, x_max_data, x_spacing)
                y_ticks = safe_ticks(y_min_data, y_max_data, y_spacing)

                # =====================================================
                # Plot
                # =====================================================
                fig = go.Figure()

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

                if has_filter and not df_filtered.empty:
                    fig.add_trace(go.Scatter(
                        x=df_filtered[x_col],
                        y=df_filtered[y_col],
                        mode="markers",
                        marker=dict(color="red", size=9),
                        name="Filtered"
                    ))

                x_unit = units_dict.get(x_col, "")
                y_unit = units_dict.get(y_col, "")

                fig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    height=400,
                    margin=dict(l=40, r=20, t=60, b=40),
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
                    legend=dict(
                        orientation="h",
                        y=1.05,
                        x=1,
                        xanchor="right"
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

                # =====================================================
                # Download buttons (PNG + HTML fallback)
                # =====================================================
                dcol1, dcol2 = st.columns(2)

                with dcol1:
                    try:
                        png_bytes = fig.to_image(format="png")
                        st.download_button(
                            "PNG",
                            data=png_bytes,
                            file_name=f"plot_{plot_index + 1}.png",
                            mime="image/png",
                            key=f"png_{plot_index}"
                        )
                    except Exception:
                        st.caption("PNG not available")

                with dcol2:
                    html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
                    st.download_button(
                        "HTML",
                        data=html_bytes,
                        file_name=f"plot_{plot_index + 1}.html",
                        mime="text/html",
                        key=f"html_{plot_index}"
                    )

            plot_index += 1
