import streamlit as st
import pandas as pd
import plotly.express as px
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
                # Read TXT
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

                # Store units
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

        # Automatically switch to Scatter tab with last converted file
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

    # Load parquet from Data folder (optional)
    parquet_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".parquet")]

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_file = st.selectbox(
            "Load from project Data folder",
            ["None"] + parquet_files
        )

    with col2:
        if selected_file != "None" and st.button("Load file"):
            path = os.path.join(DATA_FOLDER, selected_file)
            st.session_state.df = pd.read_parquet(path)
            st.session_state.current_file = selected_file
            st.rerun()

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
            colx, coly = st.columns(2)

            with colx:
                x_col = st.selectbox("X axis", numeric_cols)

            with coly:
                y_col = st.selectbox("Y axis", numeric_cols)

            # Filtering UI
            st.subheader("Filter")

            filter_col = st.selectbox("Filter column", ["None"] + numeric_cols)

            df_filtered = df

            if filter_col != "None":
                fcol1, fcol2 = st.columns(2)
                with fcol1:
                    min_val = st.number_input(
                        "Min value", value=float(df[filter_col].min())
                    )
                with fcol2:
                    max_val = st.number_input(
                        "Max value", value=float(df[filter_col].max())
                    )

                df_filtered = df[(df[filter_col] >= min_val) & (df[filter_col] <= max_val)]

            # Scatter plot
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                opacity=0.4,
                labels={
                    x_col: f"{x_col} ({units_dict.get(x_col,'')})",
                    y_col: f"{y_col} ({units_dict.get(y_col,'')})"
                },
            )

            fig.add_scatter(
                x=df_filtered[x_col],
                y=df_filtered[y_col],
                mode="markers",
                name="Filtered",
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=600,
                xaxis=dict(tickformat="d"),
                yaxis=dict(tickformat="d")
            )

            st.plotly_chart(fig, use_container_width=True)
