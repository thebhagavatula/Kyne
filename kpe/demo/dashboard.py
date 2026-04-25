import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import requests

st.set_page_config(page_title="Kyne KPE Verifier", layout="wide")

st.markdown("""
    <style>
    .verdict-match   { font-size:3rem; font-weight:900; color:#00e676; text-align:center; }
    .verdict-nomatch { font-size:3rem; font-weight:900; color:#ff1744; text-align:center; }
    </style>
""", unsafe_allow_html=True)

st.title("Kyne - Kinetic Digital Guardianship")
st.caption("Kinetic Perceptual Entropy - Motion Signature Verification - SDG 9")
st.divider()

# mock data used when no file is uploaded or api is offline
t = np.linspace(0, 4 * np.pi, 100)
MOCK = {
    "match_id":     0,
    "confidence":   0.91,
    "verdict":      "MATCH",
    "ref_signal":   np.sin(t) + 0.3 * np.sin(3 * t) + np.random.normal(0, 0.03, 100),
    "query_signal": np.sin(t) + 0.3 * np.sin(3 * t) + np.random.normal(0, 0.18, 100),
    "dtw_path_x":   np.linspace(0, 99, 60).astype(int),
    "dtw_path_y":   np.clip(np.linspace(0, 99, 60).astype(int) + np.random.randint(-5, 5, 60), 0, 99),
}

# sends video file to /verify and returns api response dict
# sends video file to /verify and returns api response dict
def call_verify_api(uploaded_file):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        resp = requests.post("http://127.0.0.1:8000/verify", files=files, timeout=30)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "backend offline - showing mock results"
    except requests.exceptions.Timeout:
        return None, "request timed out"
    except Exception as e:
        return None, str(e)

# layout split into upload panel and result panel
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("Suspect Clip Upload")
    uploaded = st.file_uploader("Drop video (MP4 / AVI / MOV)", type=["mp4", "avi", "mov"])
    if uploaded:
        st.video(uploaded)
        if st.button("Verify Clip", width="stretch"):
            with st.spinner("Analysing motion signature..."):
                api_result, error = call_verify_api(uploaded)
                if error:
                    st.warning(error)
                else:
                    st.session_state["result"] = api_result
    else:
        st.info("No file uploaded - showing mock demo results")

# uses session state result if available otherwise falls back to mock
result = st.session_state.get("result", MOCK)

with col_result:
    st.subheader("Verification Result")
    verdict = result.get("verdict", MOCK["verdict"])
    conf    = result.get("confidence", MOCK["confidence"])
    mid     = result.get("match_id", MOCK["match_id"])

    st.write("")
    st.write("")

    # renders green match or red no match verdict
    if verdict == "MATCH":
        st.markdown("<div class='verdict-match'>MATCH</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='verdict-nomatch'>NO MATCH</div>", unsafe_allow_html=True)

    st.write("")
    st.progress(conf, text=f"Confidence: {conf:.0%}")
    st.caption(f"Match ID: {mid}")
    st.write("")

    col_a, col_b = st.columns(2)
    col_a.metric("Confidence", f"{conf:.0%}")
    col_b.metric("Distortion", f"{1 - conf:.0%}")

st.divider()

# uses real signals from api if available otherwise falls back to mock
ref_signal   = list(result.get("ref_signal",   list(MOCK["ref_signal"])))
query_signal = list(result.get("query_signal", list(MOCK["query_signal"])))

n = max(len(ref_signal), len(query_signal))
ref_signal   = ref_signal   + [None] * (n - len(ref_signal))
query_signal = query_signal + [None] * (n - len(query_signal))

st.subheader("Motion Signature Waveform - S(t) vs S prime(t)")
st.caption("Reference broadcast overlaid with suspect clip")

wave_df = pd.DataFrame({
    "frame":  list(range(n)) + list(range(n)),
    "signal": ref_signal + query_signal,
    "source": ["Reference S(t)"] * n + ["Suspect S prime(t)"] * n,
})

wave_chart = alt.Chart(wave_df).mark_line().encode(
    x=alt.X("frame:Q", title="frame index"),
    y=alt.Y("signal:Q", title="motion magnitude"),
    color=alt.Color("source:N", scale=alt.Scale(
        domain=["Reference S(t)", "Suspect S prime(t)"],
        range=["#489fb5", "#ff8800"]
    )),
    strokeWidth=alt.value(2)
).properties(height=300)

st.altair_chart(wave_chart, use_container_width=True)

st.divider()

# shows dtw warping path between reference and suspect frames
st.subheader("DTW Alignment Path")
st.caption("Diagonal means perfect sync between reference and suspect")

dtw_df = pd.DataFrame({
    "reference frame": MOCK["dtw_path_x"],
    "suspect frame":   MOCK["dtw_path_y"],
})

dtw_chart = alt.Chart(dtw_df).mark_line(color="#489fb5", strokeWidth=2).encode(
    x=alt.X("reference frame:Q", title="reference frame index"),
    y=alt.Y("suspect frame:Q",   title="suspect frame index"),
).properties(height=300)

st.altair_chart(dtw_chart, use_container_width=True)