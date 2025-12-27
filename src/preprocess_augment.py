import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import joblib
from random import choice, uniform
from collections import Counter
from sklearn.preprocessing import StandardScaler

# === CONFIGURATION ===
DATA_DIR = "data/Baby Cry Sence Dataset"
OUT_FILE = "data/features_balanced.pkl"
TARGET_MIN = 80  # samples per class minimum

# === Augmentation Functions ===
def add_noise(y):
    noise = np.random.randn(len(y))
    return y + 0.005 * noise

def pitch_shift(y, sr):
    n_steps = uniform(-2, 2)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def time_stretch_fixed(y, rate=1.1):
    """Librosa time stretch compatibility"""
    try:
        return librosa.effects.time_stretch(y=y, rate=rate)
    except TypeError:
        return librosa.effects.time_stretch(y, rate)

AUG_FUNCS = ['noise', 'pitch', 'stretch']

def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def augment_audio(y, sr):
    func = choice(AUG_FUNCS)
    if func == 'noise':
        return add_noise(y)
    elif func == 'pitch':
        return pitch_shift(y, sr)
    else:
        rate = uniform(0.8, 1.2)
        y2 = time_stretch_fixed(y, rate)
        if len(y2) > len(y):
            y2 = y2[:len(y)]
        else:
            y2 = np.pad(y2, (0, len(y)-len(y2)), mode='constant')
        return y2

# === MAIN FUNCTION ===
def build_dataset(target_min=TARGET_MIN):
    X, y = [], []

    for label in os.listdir(DATA_DIR):
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder):
            continue

        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        print(f"\nProcessing {label} ({len(files)} files)")

        for file in tqdm(files, desc=f"{label}"):
            path = os.path.join(folder, file)
            try:
                y_audio, sr = librosa.load(path, sr=22050)
                feat = extract_mfcc(y_audio, sr)
                X.append(feat)
                y.append(label)
            except Exception as e:
                print(f"Error in {file}: {e}")

        # Augment if too few
        if len(files) < target_min:
            needed = target_min - len(files)
            for _ in range(needed):
                base_file = choice(files)
                path = os.path.join(folder, base_file)
                try:
                    y_audio, sr = librosa.load(path, sr=22050)
                    y_aug = augment_audio(y_audio, sr)
                    feat = extract_mfcc(y_aug, sr)
                    X.append(feat)
                    y.append(label)
                except Exception as e:
                    print(f"Aug error in {label}: {e}")

    X = np.array(X)
    y = np.array(y)
    print("\nBefore normalization — Shape:", X.shape)

    # === Normalize ===
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/feature_scaler.pkl")

    print("✅ Normalization complete. Scaler saved as models/feature_scaler.pkl")
    print("Final Class Counts:", Counter(y))

    joblib.dump((X, y), OUT_FILE)
    print(f"✅ Balanced normalized features saved as {OUT_FILE}")
    return X, y


if __name__ == "__main__":
    build_dataset()
