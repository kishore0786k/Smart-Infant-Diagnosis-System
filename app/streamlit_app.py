import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import joblib
import sounddevice as sd
import soundfile as sf
import pandas as pd
import altair as alt
import os
import time

# ------------------------------------------------------------
# Custom CSS Styling
# ------------------------------------------------------------
custom_css = """
<style>
/* Background gradient */
.stApp {
    background: linear-gradient(135deg, #e3f2fd 0%, #ffffff 100%);
}

/* Title styling */
h1 {
    text-align: center;
    background: linear-gradient(90deg, #1e88e5, #42a5f5);
    color: white !important;
    padding: 12px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    margin-bottom: 25px;
}

/* Box elements */
.css-1d391kg, .css-1lcbmhc {
    background-color: #ffffffcc !important;
    border-radius: 15px;
    padding: 20px;
}

/* Success message */
.stSuccess {
    background-color: #c8e6c9;
    padding: 15px;
    border-radius: 8px;
    border-left: 5px solid #2e7d32;
    font-size: 18px;
}

/* Buttons */
.stButton>button {
    background-color: #1e88e5;
    color: white;
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #1565c0;
    color: white;
}

/* Sidebar */
.css-1lcbmhc {
    background-color: #e3f2fd !important;
}

/* Audio player spacing */
.stAudio {
    margin-top: 10px;
    margin-bottom: 20px;
}

/* Chart padding */
.css-1kyxreq {
    padding: 15px !important;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------
MODEL_PATH = "../models/infant_cry_model.h5"
LABEL_PATH = "../models/label_encoder.pkl"
SCALER_PATH = "../models/feature_scaler.pkl"

# ------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------
st.markdown("<h1>🍼 Smart Infant Diagnosis System</h1>", unsafe_allow_html=True)

st.write(
    "Upload or record an infant cry sound to diagnose the emotional/physical state.\n"
    "This model is trained on **balanced & normalized MFCC features**."
)

# ------------------------------------------------------------
# Load model, labels, scaler
# ------------------------------------------------------------
@st.cache_resource
def load_components():
    model = tf.keras.models.load_model(MODEL_PATH)
    le = joblib.load(LABEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, le, scaler

model, le, scaler = load_components()
n_features = model.input_shape[1]

# ------------------------------------------------------------
# Feature extraction + prediction
# ------------------------------------------------------------
def predict_from_audio(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feat = np.mean(mfcc.T, axis=0).reshape(1, -1)

    feat = scaler.transform(feat)

    if feat.shape[1] != n_features:
        st.error(f"Feature mismatch. Expected {n_features}, got {feat.shape[1]}")
        return None

    probs = model.predict(feat)[0]
    pred_idx = np.argmax(probs)
    pred_label = le.classes_[pred_idx]
    conf = probs[pred_idx] * 100

    st.success(f"🩺 **Diagnosis: {pred_label.upper()}** ({conf:.2f}% confidence)")

    prob_df = pd.DataFrame({
        "Cry Type": le.classes_,
        "Confidence": probs * 100
    }).sort_values(by="Confidence", ascending=False)

    chart = (
        alt.Chart(prob_df)
        .mark_bar()
        .encode(
            x=alt.X("Cry Type", sort='-y'),
            y="Confidence",
            color=alt.Color("Cry Type", legend=None),
            tooltip=["Cry Type", "Confidence"]
        )
    )

    st.altair_chart(chart, use_container_width=True)

    return pred_label

# ------------------------------------------------------------
# Sidebar mode selector
# ------------------------------------------------------------
mode = st.sidebar.radio("Select Mode:", ["🎧 Upload Audio File", "🎙️ Live Recording"])

# ------------------------------------------------------------
# MODE 1: Upload Audio File
# ------------------------------------------------------------
if mode == "🎧 Upload Audio File":
    uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

    if uploaded_file:
        try:
            y, sr = librosa.load(uploaded_file, sr=22050)
            st.audio(uploaded_file, format="audio/wav")
            predict_from_audio(y, sr)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Please upload a .wav file.")

# ------------------------------------------------------------
# MODE 2: Live Recording
# ------------------------------------------------------------
else:
    duration = st.slider("Recording Duration (seconds)", 3, 10, 5)
    sample_rate = 22050

    if st.button("🎤 Start Recording"):
        st.info("Recording… make the cry sound now!")

        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()

        temp_path = "live_record.wav"
        sf.write(temp_path, recording, sample_rate)

        st.success("Recording complete!")
        st.audio(temp_path, format="audio/wav")

        try:
            y, sr = librosa.load(temp_path, sr=22050)

            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))

            predict_from_audio(y, sr)
        except Exception as e:
            st.error(f"Recording error: {e}")

        time.sleep(1)
        if os.path.exists(temp_path):
            os.remove(temp_path)
