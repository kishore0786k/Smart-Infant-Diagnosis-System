import librosa, numpy as np, os
from tqdm import tqdm
import joblib

def extract_features(base_dir):
    X, y = [], []
    for label in os.listdir(base_dir):
        folder = os.path.join(base_dir, label)
        if not os.path.isdir(folder):
            continue
        print(f"🔍 Processing folder: {label}")
        for file in tqdm(os.listdir(folder), desc=f"{label}"):
            if not file.endswith(".wav"):
                continue
            path = os.path.join(folder, file)
            try:
                y_audio, sr = librosa.load(path, sr=22050)
                mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
                feat = np.mean(mfcc.T, axis=0)
                X.append(feat)
                y.append(label)
            except Exception as e:
                print(f"⚠️ Error processing {file}: {e}")
    X, y = np.array(X), np.array(y)
    os.makedirs("data", exist_ok=True)
    joblib.dump((X, y), "data/features.pkl")
    print(f"✅ Feature extraction complete: {X.shape[0]} samples, {X.shape[1]} features each.")
    return X, y

if __name__ == "__main__":
    extract_features("data/Baby Cry Sence Dataset")
