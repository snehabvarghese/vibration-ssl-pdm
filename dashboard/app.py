"""
STEP 12: Streamlit machine-health dashboard.

WHAT IT SHOWS
-------------
For a selected (or uploaded) vibration signal:
  * NORMAL / ANOMALOUS status from the EDR-based health monitor
  * predicted fault class + classification confidence
  * EDR value and per-segment anomaly score
  * anomaly trend along the signal
  * the embedding map with the current signal highlighted

WHERE THE DATA COMES FROM
-------------------------
Everything is loaded from artefacts produced by the pipeline:
  checkpoints/ssl_encoder.pt         the frozen encoder
  checkpoints/health_monitor.joblib  the fitted healthy baseline + thresholds
  results/embeddings/embeddings.npz  cached embeddings for the map
A small classifier is fitted once, on the training split embeddings, and
cached in the Streamlit session.

PHASE 2 HOOK
------------
`analyse_signal()` takes a raw 1-D array plus its sampling rate and does the
whole raw -> preprocess -> embed -> score chain. A live MQTT/serial reader
only has to call that same function with a fresh buffer; nothing else in this
file is file-specific.

RUN
---
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config import CFG, DATA_RAW_DIR
from core.analysis import (
    analyse_signal,
    get_classifier,
    get_embeddings,
    get_encoder,
    get_map,
    get_monitor,
)
from preprocessing.cwru_manifest import select_records
from preprocessing.loader import load_signal

st.set_page_config(page_title="Machine Health Monitor", layout="wide")


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def sidebar() -> Tuple[str, object]:
    st.sidebar.title("Machine Health Monitor")
    st.sidebar.caption("Phase 1 -- offline dataset. Phase 2 replaces this "
                       "source with a live sensor stream.")
    st.session_state["clf_name"] = st.sidebar.selectbox(
        "Downstream classifier", CFG.downstream.classifiers, index=0
    )
    mode = st.sidebar.radio("Signal source", ["Dataset recording", "Upload a .mat / .npy file"])
    if mode == "Dataset recording":
        records = [r for r in select_records() if (DATA_RAW_DIR / f"{r.file_id}.mat").exists()]
        if not records:
            st.sidebar.error("No raw recordings found. Run "
                             "`python -m preprocessing.download_cwru --all`.")
            return mode, None
        options = {f"{r.file_id} -- {r.label()} @ {r.load_hp} hp": r for r in records}
        choice = st.sidebar.selectbox("Recording", list(options))
        return mode, options[choice]
    return mode, st.sidebar.file_uploader("Vibration file", type=["mat", "npy"])


def read_upload(upload) -> Optional[Tuple[np.ndarray, int]]:
    import io
    import scipy.io as sio
    from preprocessing.loader import _resample_to

    # CWRU normal files (97-100) are recorded at 48 kHz, while faults are 12 kHz
    is_normal_cwru = any(upload.name.startswith(p) for p in ["97", "98", "99", "100"])
    default_rate = 48_000 if is_normal_cwru else 12_000

    rate = st.sidebar.number_input("Source Sampling Rate (Hz)",
                                   min_value=1000, value=default_rate, step=1000,
                                   help="CWRU Normal baseline files (97-100) are recorded at 48,000 Hz; fault files at 12,000 Hz.")
    
    if upload.name.endswith(".npy"):
        sig = np.load(io.BytesIO(upload.getvalue())).astype(float).ravel()
    else:
        mat = sio.loadmat(io.BytesIO(upload.getvalue()))
        keys = [k for k in mat if isinstance(k, str) and k.endswith("_time")] or [
            k for k, v in mat.items() if isinstance(k, str) and not k.startswith("__") and getattr(v, "size", 0) > 1000
        ]
        if not keys:
            st.error("No 1-D signal variable found in that .mat file.")
            return None
        
        # Pre-select _DE_time if present, as model was trained on Drive-End accelerometer
        de_keys = [k for k in keys if k.endswith("_DE_time")]
        default_idx = keys.index(de_keys[0]) if de_keys else 0
        key = st.sidebar.selectbox("Signal variable (Channel)", keys, index=default_idx)
        sig = np.asarray(mat[key]).astype(float).ravel()

    # Resample to the target model frequency (12,000 Hz) if needed
    if rate != CFG.dataset.sampling_rate:
        sig = _resample_to(sig, src_rate=int(rate), dst_rate=CFG.dataset.sampling_rate)

    return sig, CFG.dataset.sampling_rate


def main() -> None:
    mode, source = sidebar()
    st.title("Machine Health Monitor")

    if source is None:
        st.info("Select a dataset recording or upload a vibration file to begin.")
        return

    if mode == "Dataset recording":
        loaded = load_signal(source)
        signal, rate = loaded.signal, loaded.sampling_rate
        true_label = loaded.label
        header = f"Recording {loaded.record_id}  |  {loaded.load_hp} hp  |  {rate} Hz"
    else:
        read = read_upload(source)
        if read is None:
            return
        signal, rate = read
        true_label = None
        header = f"Uploaded: {source.name}  |  {rate} Hz  |  {len(signal)} samples"

    st.caption(header)
    clf_name = st.session_state.get("clf_name", "svm_rbf")
    result = analyse_signal(signal, rate, clf_name=clf_name)
    if result is None:
        st.error("Signal too short to produce a single analysis window.")
        return

    # ---- status row ----
    c1, c2, c3, c4 = st.columns(4)
    if "anomalous" in result:
        status = "ANOMALOUS" if result["anomalous"] else "NORMAL"
        c1.metric("Status", status)
        c1.markdown(
            f"<span style='color:{'#C44E52' if result['anomalous'] else '#55A868'};"
            f"font-weight:600'>{'above' if result['anomalous'] else 'below'} EDR threshold</span>",
            unsafe_allow_html=True,
        )
        c3.metric("EDR (nats/s)", f"{result['edr_mean']:.2f}",
                  delta=f"threshold {result['edr_threshold']:.2f}", delta_color="off")
        c4.metric("Segments flagged", f"{result['fraction_segments_flagged']*100:.0f}%")
    else:
        c1.warning("Health monitor not fitted -- run `python -m anomaly_detection.anomaly_detector`")
    c2.metric("Predicted fault", str(result["predicted_class"]),
              delta=f"{result['confidence']*100:.1f}% confidence", delta_color="off")

    if true_label is not None:
        ok = true_label == result["predicted_class"]
        st.caption(f"True dataset label: **{true_label}** "
                   f"({'matches' if ok else 'does NOT match'} the prediction). "
                   "Shown for verification only; the model is not given the label.")

    # ---- waveform ----
    st.subheader("Vibration signal")
    n_show = min(len(signal), int(2 * rate))
    st.line_chart(pd.DataFrame({"amplitude": signal[:n_show]}), height=180)
    st.caption(f"first {n_show/rate:.1f} s of {len(signal)/rate:.1f} s")

    # ---- trend ----
    if "edr_scores" in result:
        st.subheader("Health / anomaly trend")
        scores = np.asarray(result["edr_scores"])
        df = pd.DataFrame({
            "EDR": scores,
            "threshold": np.full(len(scores), result["edr_threshold"]),
        })
        st.line_chart(df, height=240)
        st.caption("EDR = KL divergence of the current embedding window from the "
                   "healthy baseline, per second. Our approximation of the "
                   "reference paper's metric, not a reproduction of it.")

    # ---- embedding map ----
    st.subheader("Embedding space")
    xy, labels, classes, pca = get_map()
    current = pca.transform(np.asarray(result["embeddings"]))
    df_map = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1],
                           "class": [classes[i] for i in labels]})
    try:
        import altair as alt

        base = alt.Chart(df_map).mark_circle(size=18, opacity=0.5).encode(
            x="x", y="y", color=alt.Color("class:N", legend=alt.Legend(columns=2)),
            tooltip=["class"],
        )
        here = alt.Chart(pd.DataFrame({"x": current[:, 0], "y": current[:, 1]})).mark_point(
            size=70, color="black", shape="cross"
        ).encode(x="x", y="y")
        st.altair_chart((base + here).properties(height=420), use_container_width=True)
        st.caption("PCA of the self-supervised embeddings. Black crosses = the "
                   "segments of the selected signal.")
    except Exception:  # pragma: no cover - altair should ship with streamlit
        st.scatter_chart(df_map, x="x", y="y", color="class", height=420)

    # ---- probabilities ----
    st.subheader("Fault classification")
    probs = pd.Series(result["class_probabilities"]).sort_values(ascending=False)
    st.bar_chart(probs.head(8), height=240)
    st.caption(f"Classifier: {st.session_state['clf_name']} on frozen self-supervised "
               f"embeddings; probabilities averaged over {result['n_segments']} segments.")


if __name__ == "__main__":
    main()
