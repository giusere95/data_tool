import streamlit as st
import pandas as pd
import plotly.express as px
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
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

# Tabs
tab_upload, tab_plot = st.tabs(["Upload / Convert", "Scatter Plot"])

# -----------------------------
# TAB 1 — Upload / Convert TXT
# -----------------------------
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

        # Save units json
        if units_dict:
            with open(UNIT_FILE, "w", encoding="utf-8") as f:
                json.dump(units_dict, f, ensure_ascii=False, indent=2)

        if last_parquet_path:
            st.session_state.df = pd.read_parquet(last_parquet_path)
            st.session_state.current_file = os.path.basename(last_parquet_path)
            st.session_state.active_tab = 1
            st.rerun()

    st.divider()
    st.subheader("Or upload a Parquet file directly")
    uploaded_parquet = st.file_uploader("Upload Parquet", type=["parquet"])
    if uploaded_parquet is not None:
        st.session_state.df = pd.read_parquet(uploaded_parquet)
        st.session_state.current_file = uploaded_parquet.name
        st.session_state.active_tab = 1
        st.rerun()

# -----------------------------
# TAB 2 — Scatter Plot
# -----------------------------
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
        st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("Need at least two numeric columns.")
        else:
            # -----------------------------
            # Sidebar settings
            # -----------------------------
            st.sidebar.header("Plot Configuration")
            n_plots = st.sidebar.number_input("Number of plots", min_value=1, max_value=6, value=2)
            plots_per_row = st.sidebar.number_input("Plots per row", min_value=1, max_value=3, value=2)

            # Global filters
            st.sidebar.header("Global Filters")
            filter_cols = st.sidebar.multiselect("Columns to filter", numeric_cols)
            filter_values = {}
            for col in filter_cols:
                min_val = st.sidebar.number_input(f"{col} min", value=float(df[col].min()))
                max_val = st.sidebar.number_input(f"{col} max", value=float(df[col].max()))
                filter_values[col] = (min_val, max_val)

            # Per-plot axis/grid spacing
            st.sidebar.header("Per-plot Axis/Grid Spacing")
            x_spacing_list = []
            y_spacing_list = []
            vertical_spacing_list = []
            for i in range(n_plots):
                x_spacing_list.append(st.sidebar.number_input(f"X spacing Plot {i+1}", min_value=1, value=1))
                y_spacing_list.append(st.sidebar.number_input(f"Y spacing Plot {i+1}", min_value=1, value=1))
                vertical_spacing_list.append(st.sidebar.number_input(f"Vertical grid spacing Plot {i+1}", min_value=1, value=1))

            # -----------------------------
            # Apply global filter
            # -----------------------------
            if filter_cols:
                df_filtered = df.copy()
                for col, (mn, mx) in filter_values.items():
                    df_filtered = df_filtered[(df_filtered[col] >= mn) & (df_filtered[col] <= mx)]
                has_filter = True
            else:
                df_filtered = pd.DataFrame()
                has_filter = False

            # -----------------------------
            # Create plots
            # -----------------------------
            rows = []
            for i in range(n_plots):
                if i % plots_per_row == 0:
                    row = st.columns(plots_per_row)
                    rows.append(row)

                row_idx = i // plots_per_row
                col_idx = i % plots_per_row
                col = rows[row_idx][col_idx]

                with col:
                    st.subheader(f"Plot {i+1}")
                    x_col = st.selectbox(f"X axis (Plot {i+1})", numeric_cols, key=f"x_{i}")
                    y_col = st.selectbox(f"Y axis (Plot {i+1})", numeric_cols, key=f"y_{i}")

                    # -----------------------------
                    # Separate base and filtered points
                    # -----------------------------
                    if has_filter and not df_filtered.empty:
                        df_base = df.drop(df_filtered.index)
                    else:
                        df_base = df.copy()

                    # Base scatter
                    fig = px.scatter(
                        df_base,
                        x=x_col,
                        y=y_col,
                        opacity=0.4,
                        color_discrete_sequence=["blue"],
                        labels={
                            x_col: f"{x_col} ({units_dict.get(x_col, '')})",
                            y_col: f"{y_col} ({units_dict.get(y_col, '')})"
                        },
                    )

                    # Filtered points on top
                    if has_filter and not df_filtered.empty:
                        fig.add_scatter(
                            x=df_filtered[x_col],
                            y=df_filtered[y_col],
                            mode="markers",
                            name="Filtered",
                            marker=dict(color="red", size=10),
                        )

                    # Add vertical grid lines
                    x_min, x_max = df[x_col].min(), df[x_col].max()
                    x_lines = list(range(int(x_min), int(x_max)+1, vertical_spacing_list[i]))
                    for x in x_lines:
                        fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="lightgray")

                    # Layout
                    fig.update_layout(
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        height=500,
                        xaxis=dict(tickformat="d", dtick=x_spacing_list[i]),
                        yaxis=dict(tickformat="d", dtick=y_spacing_list[i])
                    )

                    st.plotly_chart(fig, use_container_width=True)
