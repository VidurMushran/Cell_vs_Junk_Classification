#!/usr/bin/env python3
import os, glob, math, base64, html
import streamlit as st
import pandas as pd
import numpy as np
import h5py
from io import BytesIO
from PIL import Image
import streamlit.components.v1 as components

# -----------------------------
# Display helpers (EXACT PyQt composite)
# -----------------------------
def channels_to_rgb8bit(img):
    a = img.astype(np.float32)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"Expected HWC with >=3 channels, got {a.shape}")

    rgb = a[..., [1, 2, 0]]  # TRITC, CY5, DAPI
    if a.shape[2] > 3:
        rgb = rgb + a[..., 3:4]  # add FITC to all

    rgb[rgb > 65535.0] = 65535.0
    rgb[rgb < 0.0] = 0.0
    return (rgb // 256.0).astype(np.uint8)

def load_h5_scores_and_index(h5_path):
    df = pd.read_hdf(h5_path, "features")
    if "model_score" not in df.columns:
        if "confidence_score" in df.columns:
            df["model_score"] = df["confidence_score"].astype(np.float32)
        else:
            raise ValueError("No model_score or confidence_score present.")
    scores = df["model_score"].astype(np.float32).values
    return df, scores, len(df)

def get_images_by_indices(h5_path, indices):
    imgs = []
    with h5py.File(h5_path, "r") as f:
        X = f["images"]
        for i in indices:
            raw = X[int(i)]
            imgs.append(channels_to_rgb8bit(raw))
    return imgs

# -----------------------------
# HTML gallery helpers
# -----------------------------
def np_img_to_b64_png(img_rgb8):
    """numpy HxWx3 uint8 -> base64 PNG string"""
    pil_img = Image.fromarray(img_rgb8)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def safe_get(row, keys, default="NA"):
    """Try multiple possible column names."""
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    return default

def render_gallery_column(
    title,
    h5_path,
    df,
    indices_sorted,
    page,
    page_size,
    n_cols,
    scores_full,
    thr,
    score_col="model_score",
):
    st.markdown(f"### {title}")
    total = len(indices_sorted)
    if total == 0:
        st.info("No cells in this bucket.")
        return

    total_pages = math.ceil(total / page_size)
    page = min(max(page, 0), max(total_pages - 1, 0))
    start = page * page_size
    end = min(start + page_size, total)

    st.caption(f"Showing {start+1}–{end} / {total} (page {page+1}/{total_pages})")

    page_indices = indices_sorted[start:end]
    imgs = get_images_by_indices(h5_path, page_indices)

    cells_html = []
    for img, idx in zip(imgs, page_indices):
        row = df.iloc[int(idx)]
        score = float(row[score_col]) if score_col in row else float(scores_full[int(idx)])
        pred = int(score >= thr)

        slide_id = safe_get(row, ["slide_id", "Slide_ID", "slide"])
        frame_id = safe_get(row, ["frame_id", "Frame_ID", "frame"])
        cell_id  = safe_get(row, ["cell_id", "Cell_ID", "cell"])
        cell_x   = safe_get(row, ["cell_x", "x", "X"])
        cell_y   = safe_get(row, ["cell_y", "y", "Y"])

        tooltip = (
            f"idx: {int(idx)}\n"
            f"score: {score:.4f}\n"
            f"pred: {pred}\n"
            f"slide_id: {slide_id}\n"
            f"frame_id: {frame_id}\n"
            f"cell_id: {cell_id}\n"
            f"cell_x: {cell_x}\n"
            f"cell_y: {cell_y}"
        )
        tooltip_esc = html.escape(tooltip)

        b64 = np_img_to_b64_png(img)
        cells_html.append(
            f"""
            <div class="cell">
              <img src="data:image/png;base64,{b64}"
                   title="{tooltip_esc}" />
            </div>
            """
        )

    # CSS + grid in one HTML payload
    grid_html = f"""
    <style>
    .gallery-grid {{
        display: grid;
        grid-template-columns: repeat({int(n_cols)}, 1fr);
        gap: 0px;
        margin: 0;
        padding: 0;
        width: 100%;
    }}
    .gallery-grid .cell {{
        margin: 0;
        padding: 0;
        line-height: 0;
    }}
    .gallery-grid img {{
        width: 100%;
        height: auto;
        display: block;
    }}
    </style>

    <div class="gallery-grid">
      {''.join(cells_html)}
    </div>
    """

    # Estimate height so Streamlit allocates enough space.
    # Each row is ~ (column width) tall; use a safe px guess.
    n_imgs = len(imgs)
    n_rows = math.ceil(n_imgs / int(n_cols))
    approx_row_px = 140  # tweak if you want bigger/smaller vertical space
    height_px = max(200, n_rows * approx_row_px)

    components.html(grid_html, height=height_px, scrolling=True)


# -----------------------------
# NEW: zero-gap CSS (place once near top of app)
# -----------------------------
st.markdown(
    """
    <style>
    .gallery-grid {
        display: grid;
        gap: 0px;              /* NO padding between tiles */
        margin: 0;
        padding: 0;
        width: 100%;
    }
    .gallery-grid .cell {
        margin: 0;
        padding: 0;
        line-height: 0;        /* remove vertical whitespace */
    }
    .gallery-grid img {
        width: 100%;
        height: auto;
        display: block;        /* removes inline-image gaps */
        image-rendering: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Streamlit App (rest of your file stays same)
# -----------------------------
st.set_page_config(page_title="Junk vs Rare Threshold Explorer", layout="wide")
st.title("Threshold Explorer (pick a file → sort by score → split into junk/nonjunk)")

h5_dir = st.text_input("Directory containing HDF5s", value="/mnt/deepstore/Vidur/Junk Classification/data_model_labels/unannotated")
if not os.path.isdir(h5_dir):
    st.info("Enter a valid directory.")
    st.stop()

all_files = sorted(glob.glob(os.path.join(h5_dir, "*.hdf5")))
if not all_files:
    st.info("No .hdf5 files found here.")
    st.stop()

selected = st.selectbox(
    "Select an HDF5 to browse",
    all_files,
    format_func=lambda p: os.path.basename(p)
)

try:
    df, scores, N = load_h5_scores_and_index(selected)
except Exception as e:
    st.error(f"Could not read {os.path.basename(selected)}: {e}")
    st.stop()

order_desc = np.argsort(scores)[::-1]

thr = st.slider("Threshold for junk (positive)", 0.0, 1.0, 0.5, 0.01)
top_n = st.number_input("Top-N by score to consider (per file)",
                        min_value=50, max_value=int(N),
                        value=min(500, int(N)), step=50)
page_size = st.slider("Cells per page (per column)", 20, 200, 80, 20)
n_cols = st.slider("Columns per gallery (fewer = bigger tiles)", 2, 12, 5)

order_desc = order_desc[:int(top_n)]

pred_top = (scores[order_desc] >= thr).astype(np.uint8)
junk_indices = order_desc[pred_top == 1]
cell_indices = order_desc[pred_top == 0]

if "page_junk" not in st.session_state:
    st.session_state.page_junk = 0
if "page_cell" not in st.session_state:
    st.session_state.page_cell = 0

c1, c2, c3 = st.columns(3)
c1.metric("Total cells in file", int(N))
c2.metric("Top-N shown", int(top_n))
c3.metric("Pred junk in Top-N", int(len(junk_indices)))

st.divider()

pc1, pc2, pc3, pc4 = st.columns([1,1,1,1])
with pc1:
    if st.button("⬅ Prev junk"):
        st.session_state.page_junk = max(0, st.session_state.page_junk - 1)
with pc2:
    if st.button("Next junk ➡"):
        st.session_state.page_junk += 1
with pc3:
    if st.button("⬅ Prev nonjunk"):
        st.session_state.page_cell = max(0, st.session_state.page_cell - 1)
with pc4:
    if st.button("Next nonjunk ➡"):
        st.session_state.page_cell += 1

left, right = st.columns(2)

with left:
    render_gallery_column(
        title=f"Predicted JUNK (score ≥ {thr:.2f})",
        h5_path=selected,
        df=df,
        indices_sorted=junk_indices,
        page=st.session_state.page_junk,
        page_size=page_size,
        n_cols=n_cols,
        scores_full=scores,
        thr=thr,
    )

with right:
    render_gallery_column(
        title=f"Predicted CELLS / NONJUNK (score < {thr:.2f})",
        h5_path=selected,
        df=df,
        indices_sorted=cell_indices,
        page=st.session_state.page_cell,
        page_size=page_size,
        n_cols=n_cols,
        scores_full=scores,
        thr=thr,
    )

st.divider()

baseline_thr = 0.5
pred_now = (scores >= thr).astype(np.uint8)
pred_base = (scores >= baseline_thr).astype(np.uint8)
switched = np.where(pred_now != pred_base)[0]

with st.expander("Switchers vs baseline @0.5 (this file only)"):
    st.write(f"Switched count: {int(len(switched))}")
    if len(switched) > 0:
        sw_df = df.iloc[switched].copy()
        sw_df["idx"] = switched
        sw_df["pred@thr"] = pred_now[switched]
        sw_df["pred@0.5"] = pred_base[switched]
        st.dataframe(
            sw_df[["idx","model_score","pred@thr","pred@0.5"]].sort_values("model_score", ascending=False),
            use_container_width=True
        )
