import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import json

st.set_page_config(layout="wide")

# -----------------------------
# Folders & session state
# -----------------------------
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)
UNIT_FILE = os.path.join(DATA_FOLDER, "units.json")

if "df" not in st.session_state:
    st.session_state.df = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Tabs
tab_upload, tab_plot = st.tabs(["Upload / Convert", "Scatter Plot"])

# ==========================================================
# TAB 1 — Upload / Convert TXT
# ==========================================================
with tab_upload:
    st.header("Upload TXT and convert to Parquet")

    uploaded_txt = st.file_uploader(
        "Upload TXT files",
        type=["txt"],
        accept_multiple_files=True
    )

    if uploaded_txt and st.button("Convert to Parquet"):
        last_parquet_path = None
        units_dict = {}

        for file in uploaded_txt:
            try:
                df_raw = pd.read_csv(file, sep="\t", header=None)

                headers = df_raw.iloc[0].tolist()
                units = df_raw.iloc[1].tolist()
                df_data = df_raw.iloc[2:].reset_index(drop=True)
                df_data.columns = headers

                # Convert numeric columns
                df_data = df_data.apply(pd.to_numeric, errors="ignore")

                # Save parquet
                parquet_name = file.name.replace(".txt", ".parquet")
                parquet_path = os.path.join(DATA_FOLDER, parquet_name)
                df_data.to_parquet(parquet_path, index=False)

                # Save units
                for h, u in zip(headers, units):
                    units_dict[h] = u

                last_parquet_path = parquet_path
                st.success(f"{file.name} → {parquet_name}")

            except Exception as e:
                st.error(f"Error with {file.name}: {e}")

        if units_dict:
            with open(UNIT_FILE, "w", encoding="utf-8") as f:
                json.dump(units_dict, f, indent=2)

        if last_parquet_path:
            st.session_state.df = pd.read_parquet(last_parquet_path)
            st.session_state.current_file = os.path.basename(last_parquet_path)
            st.rerun()

    st.divider()
    st.subheader("Or upload a Parquet file directly")

    uploaded_parquet = st.file_uploader("Upload Parquet", type=["parquet"])
    if uploaded_parquet is not None:
        st.session_state.df = pd.read_parquet(uploaded_parquet)
        st.session_state.current_file = uploaded_parquet.name
        st.rerun()


# ==========================================================
# TAB 2 — Scatter Plot
# ==========================================================
with tab_plot:
    st.header("Scatter Plot")

    df = st.session_state.df

    # Load units
    units_dict = {}
    if os.path.exists(UNIT_FILE):
        with open(UNIT_FILE, "r", encoding="utf-8") as f:
            units_dict = json.load(f)

    if df is None:
        st.info("Upload or load a dataset to start.")
    else:
        st.success(f"Loaded: {st.session_state.current_file}")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Need at least two numeric columns.")
        else:
            # -----------------------------
            # Sidebar — Layout
            # -----------------------------
            st.sidebar.header("Layout")
            n_plots = st.sidebar.number_input("Number of plots", 1, 6, 2)
            plots_per_row = st.sidebar.number_input("Plots per row", 1, 3, 2)

            # -----------------------------
            # Sidebar — Global Filters
            # -----------------------------
            st.sidebar.header("Global Filters")

            filter_cols = st.sidebar.multiselect("Columns to filter", numeric_cols)
            filter_values = {}

            for col in filter_cols:
                min_val = st.sidebar.number_input(
                    f"{col} min", value=float(df[col].min()), key=f"{col}_min"
                )
                max_val = st.sidebar.number_input(
                    f"{col} max", value=float(df[col].max()), key=f"{col}_max"
                )
                filter_values[col] = (min_val, max_val)

            # Apply filters
            if filter_cols:
                df_filtered = df.copy()
                for col, (mn, mx) in filter_values.items():
                    df_filtered = df_filtered[
                        (df_filtered[col] >= mn) & (df_filtered[col] <= mx)
                    ]
                has_filter = True
            else:
                df_filtered = pd.DataFrame()
                has_filter = False

            # -----------------------------
            # Download filtered data
            # -----------------------------
            if has_filter and not df_filtered.empty:
                st.sidebar.header("Export")
                st.sidebar.download_button(
                    "Download filtered (CSV)",
                    df_filtered.to_csv(index=False),
                    file_name="filtered_data.csv",
                    mime="text/csv",
                )

                st.sidebar.download_button(
                    "Download filtered (Parquet)",
                    df_filtered.to_parquet(index=False),
                    file_name="filtered_data.parquet",
                )

            # -----------------------------
            # Sidebar — Axis spacing per plot
            # -----------------------------
            st.sidebar.header("Axis spacing per plot")

            x_spacing_list = []
            y_spacing_list = []

            for i in range(n_plots):
                st.sidebar.markdown(f"**Plot {i+1}**")
                x_spacing_list.append(
                    st.sidebar.number_input(
                        "X spacing",
                        min_value=1.0,
                        value=1.0,
                        key=f"xspace_{i}",
                    )
                )
                y_spacing_list.append(
                    st.sidebar.number_input(
                        "Y spacing",
                        min_value=1.0,
                        value=1.0,
                        key=f"yspace_{i}",
                    )
                )

            # -----------------------------
            # Plot grid layout
            # -----------------------------
            rows = []
            for i in range(n_plots):
                if i % plots_per_row == 0:
                    rows.append(st.columns(plots_per_row))

                col = rows[i // plots_per_row][i % plots_per_row]

                with col:
                    st.subheader(f"Plot {i+1}")

                    x_col = st.selectbox(
                        "X axis",
                        numeric_cols,
                        key=f"x_{i}",
                    )
                    y_col = st.selectbox(
                        "Y axis",
                        numeric_cols,
                        key=f"y_{i}",
                    )

                    # Axis labels with units
                    x_label = x_col
                    y_label = y_col

                    if x_col in units_dict and units_dict[x_col] not in ["-", "", None]:
                        x_label += f" ({units_dict[x_col]})"
                    if y_col in units_dict and units_dict[y_col] not in ["-", "", None]:
                        y_label += f" ({units_dict[y_col]})"

                    # Base vs filtered separation
                    if has_filter and not df_filtered.empty:
                        df_base = df.drop(df_filtered.index)
                    else:
                        df_base = df

                    fig = go.Figure()

                    # Base points (blue)
                    fig.add_trace(
                        go.Scatter(
                            x=df_base[x_col],
                            y=df_base[y_col],
                            mode="markers",
                            marker=dict(color="blue", opacity=0.35, size=7),
                            name="All"
                        )
                    )

                    # Filtered points (red ON TOP)
                    if has_filter and not df_filtered.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=df_filtered[x_col],
                                y=df_filtered[y_col],
                                mode="markers",
                                marker=dict(color="red", size=9),
                                name="Filtered"
                            )
                        )

                    # -----------------------------
                    # Vertical grid lines (SOLID)
                    # spacing = X spacing
                    # -----------------------------
                    x_min = df[x_col].min()
                    x_max = df[x_col].max()
                    spacing = x_spacing_list[i]

                    if spacing > 0:
                        x_lines = []
                        val = x_min - (x_min % spacing)
                        while val <= x_max:
                            x_lines.append(val)
                            val += spacing

                        for xv in x_lines:
                            fig.add_vline(
                                x=xv,
                                line_width=1,
                                line_color="lightgray"
                            )

                    # Layout
                    fig.update_layout(
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        height=500,
                        xaxis=dict(
                            title=x_label,
                            dtick=x_spacing_list[i],
                            tickformat="g"
                        ),
                        yaxis=dict(
                            title=y_label,
                            dtick=y_spacing_list[i],
                            tickformat="g"
                        ),
                        margin=dict(l=40, r=10, t=40, b=40),
                    )

                    st.plotly_chart(fig, use_container_width=True)
