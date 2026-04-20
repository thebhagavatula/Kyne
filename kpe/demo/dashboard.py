import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Kyne KPE Verifier", layout="wide")

st.markdown("""
    <style>
    .verdict-match   { font-size:3rem; font-weight:900; color:#00e676; text-align:center; }
    .verdict-nomatch { font-size:3rem; font-weight:900; color:#ff1744; text-align:center; }
    </style>
""", unsafe_allow_html=True)

st.title("Kyne - Kinetic Digital Guardianship")
st.caption("Kinetic Perceptual Entropy - Motion Signature Verification")
st.divider()

# mock data to be replaced with real api response on day 4
t = np.linspace(0, 4 * np.pi, 100)
MOCK = {
    "match_id": 0,
    "confidence":   0.91,
    "verdict":      "MATCH",
    "ref_signal":   np.sin(t) + 0.3 * np.sin(3 * t) + np.random.normal(0, 0.03, 100),
    "query_signal": np.sin(t) + 0.3 * np.sin(3 * t) + np.random.normal(0, 0.18, 100),
    "dtw_path_x":   np.linspace(0, 99, 60).astype(int),
    "dtw_path_y":   np.clip(np.linspace(0, 99, 60).astype(int) + np.random.randint(-5, 5, 60), 0, 99),
}

# layout split into upload panel and result panel
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("Suspect Clip Upload")
    uploaded = st.file_uploader("Drop video (MP4 / AVI / MOV)", type=["mp4", "avi", "mov"])
    if uploaded:
        st.video(uploaded)
        st.button("Verify Clip", width='stretch')
    else:
        st.info("No file uploaded - showing mock demo results")

with col_result:
    st.subheader("Verification Result")
    verdict = MOCK["verdict"]
    conf    = MOCK["confidence"]

    # renders green match or red no match verdict
    if verdict == "MATCH":
        st.markdown("<div class='verdict-match'>MATCH</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='verdict-nomatch'>NO MATCH</div>", unsafe_allow_html=True)

    st.progress(conf, text=f"Confidence: {conf:.0%}")
    st.caption(f"Match ID: {MOCK['match_id']}")

    col_a, col_b = st.columns(2)
    col_a.metric("Confidence", f"{conf:.0%}")
    col_b.metric("Distortion", f"{1 - conf:.0%}")

st.divider()

# overlays reference signal and suspect signal on one chart
st.subheader("Motion Signature Waveform - S(t) vs S prime(t)")
st.caption("Reference broadcast overlaid with suspect clip")

wave_df = pd.DataFrame({
    "Reference S(t) broadcast":    MOCK["ref_signal"],
    "Suspect S prime(t) pirate clip": MOCK["query_signal"],
})
st.line_chart(wave_df, width='stretch')

st.divider()

# shows how suspect timeline maps to reference via dtw warping path
st.subheader("DTW Alignment Path")
st.caption("Diagonal means perfect sync between reference and suspect")

dtw_df = pd.DataFrame({
    "Reference frame index": MOCK["dtw_path_x"],
    "Suspect frame index":   MOCK["dtw_path_y"],
})
st.line_chart(dtw_df.set_index("Reference frame index"), width='stretch')