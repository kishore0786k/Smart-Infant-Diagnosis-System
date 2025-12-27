import streamlit as st
import sounddevice as sd
import wavio
import librosa
import numpy as np
import tensorflow as tf
import joblib
import time
import os

# Load model and encoder
MODEL_PATH = '../models/infant_cry_model.h5'
LABEL_PATH = '../models/label_encoder.pkl'
model = tf.keras.models.load_model(MODEL_PATH)
le = joblib.load(LABEL_PATH)

st.set_page_config(page_title="Live Cry Analyzer", page_icon="🎤")
st.title("🎤 Live Infant Cry Recording & Diagnosis")
st.write("Record a few seconds of audio and analyze the cry in real-time using the trained model.")

# Record settings
duration = st.slider("Recording Duration (seconds)", 3, 10, 5)
sample_rate = 22050

if st.button("🎙️ Record Cry Sound"):
    st.info("Recording... Please make some sound (cry simulation or background noise).")
    st.write("Recording duration:", duration, "seconds")
    st.write("⏳ Please wait...")

    # Record from mic
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # wait till recording is finished
    file_path = "live_record.wav"
    wavio.write(file_path, recording, sample_rate, sampwidth=2)

    st.success("✅ Recording complete! Saved as live_record.wav.")
    st.audio(file_path, format='audio/wav')

    # Process audio for prediction
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feat = np.mean(mfcc.T, axis=0)

    # Add simulated biological features (to match trained model)
    hr = np.random.normal(140, 10, 1)
    temp = np.random.normal(37.0, 0.3, 1)
    feat_bio = np.concatenate([feat, hr, temp]).reshape(1, -1)

    # Predict
    pred = np.argmax(model.predict(feat_bio), axis=1)[0]
    predicted_label = le.classes_[pred]

    st.subheader(f"🩺 Predicted Diagnosis: **{predicted_label.upper()}**")
    st.caption("*(Model prediction based on live cry and simulated biological data)*")

    # Delete old recording to keep folder clean
    if os.path.exists(file_path):
        time.sleep(2)
        os.remove(file_path)
