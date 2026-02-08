import streamlit as st
import pandas as pd
import os
import io
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")

st.title("Data Tool")

# ============================
# Folders (for local + cloud)
# ============================
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# ============================
# Tabs
# ============================
tab1, tab2 = st.tabs(["TXT → Parquet Converter", "Scatter Plot Viewer"])

# ============================================================
# TAB 1 — TXT CONVERTER (NO PLOTS HERE)
# ============================================================
with tab1:
    st.header("Upload TXT and convert to Parquet")

    txt_files = st.file_uploader(
        "Upload one or more TXT files",
        type=["txt"],
        accept_multiple_files=True
    )

    if txt_files:
        for file in txt_files:
            st.write(f"Processing: {file.name}")

            # Read TXT
            try:
                df = pd.read_csv(file, sep=None, engine="python")

                # Save parquet
                parquet_name = file.name.replace(".txt", ".parquet")
                parquet_path = os.path.join(DATA_FOLDER, parquet_name)
                df.to_parquet(parquet_path, index=False)

                st.success(f"Saved as {parquet_name}")

            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

    st.info("Converted files are stored and available in the Scatter Plot tab.")


# ============================================================
# TAB 2 — SCATTER PLOTS
# ============================================================
with tab2:
    st.header("Scatter Plot Viewer")

    # --- Option 1: Upload parquet directly ---
    uploaded_parquet = st.file_uploader(
        "Upload a Parquet file",
        type=["parquet"]
    )

    df = None

    if uploaded_parquet is not None:
        df = pd.read_parquet(uploaded_parquet)
        st.success("Using uploaded Parquet file")

    else:
        # --- Option 2: Load from data folder ---
        parquet_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".parquet")]

        if parquet_files:
            selected_file = st.selectbox("Or select a file from server", parquet_files)
            df = pd.read_parquet(os.path.join(DATA_FOLDER, selected_file))
        else:
            st.warning("No parquet files available. Upload one or convert a TXT first.")

    # ============================================================
    # Plot section
    # ============================================================
    if df is not None:

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Need at least two numeric columns.")
        else:
            st.subheader("Scatter Plot")

            col1, col2 = st.columns(2)

            with col1:
                x_col = st.selectbox("X axis", numeric_cols)

            with col2:
                y_col = st.selectbox("Y axis", numeric_cols)

            # Simple filtering UI
            st.markdown("### Filter")

            filter_col = st.selectbox("Column to filter", numeric_cols)

            min_val = float(df[filter_col].min())
            max_val = float(df[filter_col].max())

            col_min, col_max = st.columns(2)

            with col_min:
                user_min = st.number_input("Min value", value=min_val)

            with col_max:
                user_max = st.number_input("Max value", value=max_val)

            df_filtered = df[(df[filter_col] >= user_min) & (df[filter_col] <= user_max)]

            # Highlight filtered points
            df["Selection"] = "Other"
            df.loc[df_filtered.index, "Selection"] = "Filtered"

            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color="Selection",
                template="plotly_white"
            )

            fig.update_layout(
                height=600,
                xaxis=dict(tickformat="d"),
                yaxis=dict(tickformat="d")
            )

            st.plotly_chart(fig, use_container_width=True)
