import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import io

st.set_page_config(layout="wide")
st.title("Data Tool")

# -----------------------------
# Helpers
# -----------------------------
def extract_units(columns):
    units = {}
    clean_cols = []
    for col in columns:
        match = re.match(r"(.+?)\s*\((.+?)\)", col)
        if match:
            name = match.group(1).strip()
            unit = match.group(2).strip()
            units[name] = unit
            clean_cols.append(name)
        else:
            units[col] = ""
            clean_cols.append(col)
    return clean_cols, units


def read_txt_with_units(uploaded_file):
    content = uploaded_file.read().decode("utf-8")
    lines = content.strip().split("\n")

    header = lines[0].split("\t")
    clean_cols, units = extract_units(header)

    data = "\n".join(lines[1:])
    df = pd.read_csv(io.StringIO(data), sep="\t", names=clean_cols)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, units


def generate_ticks(start, max_val, spacing):
    if spacing <= 0:
        return None
    return np.arange(start, max_val + spacing, spacing)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["TXT → Parquet", "Plot Data"])

df = None
units = {}

# -----------------------------
# TAB 1 - TXT conversion
# -----------------------------
with tab1:
    st.header("Convert TXT to Parquet")

    txt_file = st.file_uploader("Upload TXT file", type=["txt"])

    if txt_file is not None:
        df, units = read_txt_with_units(txt_file)

        st.success("TXT loaded successfully")
        st.write(df.head())

        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            "Download Parquet",
            buffer,
            file_name="data.parquet",
            mime="application/octet-stream"
        )

        # Keep data available for plotting without re-upload
        st.session_state["df"] = df
        st.session_state["units"] = units


# -----------------------------
# TAB 2 - Plotting
# -----------------------------
with tab2:
    st.header("Plot Data")

    uploaded_parquet = st.file_uploader("Upload Parquet file", type=["parquet"])

    if uploaded_parquet is not None:
        df = pd.read_parquet(uploaded_parquet)
        st.session_state["df"] = df
        st.session_state["units"] = {c: "" for c in df.columns}

    if "df" in st.session_state:
        df = st.session_state["df"]
        units = st.session_state.get("units", {c: "" for c in df.columns})

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Need at least two numeric columns")
            st.stop()

        # -----------------------------
        # Sidebar layout
        # -----------------------------
        st.sidebar.header("Plot Settings")

        # Compact row: number of plots + plots per row
        colA, colB = st.sidebar.columns(2)
        n_plots = colA.number_input("Number of plots", 1, 12, 1)
        plots_per_row = colB.number_input("Plots per row", 1, 4, 1)

        # Axis selection (compact)
        c1, c2 = st.sidebar.columns(2)
        x_col = c1.selectbox("X axis", numeric_cols)
        y_col = c2.selectbox("Y axis", numeric_cols, index=1)

        # Units display
        x_unit = units.get(x_col, "")
        y_unit = units.get(y_col, "")

        # Axis spacing
        c3, c4 = st.sidebar.columns(2)
        x_spacing = c3.number_input("X spacing", value=1.0)
        y_spacing = c4.number_input("Y spacing", value=1.0)

        # Axis decimals
        c5, c6 = st.sidebar.columns(2)
        x_decimals = c5.number_input("X decimals", 0, 6, 2)
        y_decimals = c6.number_input("Y decimals", 0, 6, 2)

        # Axis start
        c7, c8 = st.sidebar.columns(2)
        x_start = c7.number_input("X start", value=float(df[x_col].min()))
        y_start = c8.number_input("Y start", value=float(df[y_col].min()))

        # -----------------------------
        # Filtering
        # -----------------------------
        st.sidebar.header("Filter")

        filter_col = st.sidebar.selectbox("Select filter column", ["None"] + numeric_cols)

        mask = np.ones(len(df), dtype=bool)

        if filter_col != "None":
            fmin, fmax = st.sidebar.slider(
                "Min / Max",
                float(df[filter_col].min()),
                float(df[filter_col].max()),
                (
                    float(df[filter_col].min()),
                    float(df[filter_col].max())
                )
            )

            enable_filter = st.sidebar.checkbox("Enable filter", value=True)

            if enable_filter:
                mask = (df[filter_col] >= fmin) & (df[filter_col] <= fmax)

        # -----------------------------
        # Plot grid
        # -----------------------------
        rows = int(np.ceil(n_plots / plots_per_row))
        fig, axes = plt.subplots(rows, plots_per_row, figsize=(6 * plots_per_row, 4 * rows))

        if n_plots == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for i in range(n_plots):
            ax = axes[i]

            # Unfiltered (blue)
            ax.scatter(
                df.loc[~mask, x_col],
                df.loc[~mask, y_col],
                s=10,
                alpha=0.5
            )

            # Filtered (red on top, no border)
            ax.scatter(
                df.loc[mask, x_col],
                df.loc[mask, y_col],
                s=20,
                color="red",
                alpha=0.9,
                zorder=3
            )

            # Axis ticks using start value
            x_ticks = generate_ticks(x_start, df[x_col].max(), x_spacing)
            y_ticks = generate_ticks(y_start, df[y_col].max(), y_spacing)

            if x_ticks is not None:
                ax.set_xticks(x_ticks)
                ax.set_xlim(x_start, df[x_col].max())

            if y_ticks is not None:
                ax.set_yticks(y_ticks)
                ax.set_ylim(y_start, df[y_col].max())

            # Decimal formatting
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda val, _: f"{val:.{x_decimals}f}")
            )
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda val, _: f"{val:.{y_decimals}f}")
            )

            # Labels with units
            xlabel = f"{x_col} ({x_unit})" if x_unit else x_col
            ylabel = f"{y_col} ({y_unit})" if y_unit else y_col

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # Continuous grid
            ax.grid(True, linestyle="-", alpha=0.4)

        # Remove empty axes
        for j in range(n_plots, len(axes)):
            fig.delaxes(axes[j])

        st.pyplot(fig)
